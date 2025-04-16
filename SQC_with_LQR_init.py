"""
SAC with LQR Initialization for Portfolio Execution

This script extends the SAC pipeline by:
1. Computing the infinite-horizon LQR solution (P) offline for the quadratic part
   of the reward (ignoring |x| and alpha*(p+x) term) via the discrete Riccati equation.
2. Using P to generate a synthetic value-function V0(s)=s^T P_s s as targets.
3. Pre-training the SAC critic networks on (state, V0(state)) pairs to initialize them.
4. Fine-tuning with full SAC including |x| cost and alpha signal.

Key steps:
- `compute_lqr_P(rho, t_lambda, la, gamma)`: solve DARE for P_s on state=[alpha,p]
- `pretrain_critic(critic_net, P_s, num_samples, epochs)`: sample states, train critic to match V0(s)
- Integrate pretraining into main training loop before SAC updates.

Requires: torch, numpy, scipy
Run on GPU if available.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from scipy.linalg import solve_discrete_are

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- 1. Compute LQR P matrix ----
def compute_lqr_P(rho, t_lambda, la, gamma=0.99):
    """
    Solve the discrete-time algebraic Riccati equation for the system:
      state s=[alpha; p], dynamics s' = A s + B x
      A = [[rho, 0], [0, 1]], B = [[0], [1]]
    with stage cost (quadratic part only):
      cost = la * (p + x)^2 + 0.5 * t_lambda * x^2
           = s^T Q s + 2 s^T N x + 0.5 x^T R x
    We ignore the linear alpha*(p+x) and |x| term.
    Returns: P_s (2x2 matrix)
    """
    # System matrices
    A = np.array([[rho, 0.0], [0.0, 1.0]])
    B = np.array([[0.0], [1.0]])
    # Stage-cost matrices
    # Q = 2*la on p^2 term, alpha has no cost here
    Q = np.array([[0.0, 0.0], [0.0, 2.0 * la]])
    # Cross-term N(s,x), but standard DARE doesn't handle cross-terms.
    # We absorb 2la*p*x by augmenting R
    # Effective R = t_lambda + 2*la
    R = np.array([[t_lambda + 2.0 * la]])
    # Solve DARE: A^T P A - P - (A^T P B)(R + B^T P B)^{-1}(B^T P A) + Q = 0
    P = solve_discrete_are(A, B, Q, R)
    return P

# ---- 2. Pre-train critic ----
def pretrain_critic(critic_net, P_s, num_samples=5000, epochs=20, lr=1e-3):
    """
    Pre-train a critic network to match the quadratic value V0(s)=s^T P_s s.
    - critic_net: a PyTorch nn.Module taking (state, action) and returning Q(s,a), but we will use action=0
    - P_s: 2x2 matrix from compute_lqr_P
    """
    optimizer = optim.Adam(critic_net.parameters(), lr=lr)
    for ep in range(epochs):
        # Sample random states: alpha ~ N(0,1), p~Uniform(-1,1)
        alphas = np.random.randn(num_samples)
        ps = np.random.uniform(-1, 1, size=num_samples)
        # Constant parameters (rho,t_lambda,c,la) are ignored by critic net first two dims
        # Build full state as zeros for extras
        states = np.stack([alphas, ps], axis=1)
        # Compute target V0(s)
        Vs = np.einsum('ij,kj,ki->k', P_s, states, states)  # s^T P s
        # Convert to torch
        state_t = torch.tensor(np.concatenate([states,
                                               np.zeros((num_samples,4))], axis=1),
                               dtype=torch.float32, device=device)
        action_t = torch.zeros((num_samples,1), dtype=torch.float32, device=device)
        target_t = torch.tensor(Vs, dtype=torch.float32, device=device).unsqueeze(1)
        # Regression
        critic_net.train()
        pred = critic_net(state_t, action_t)
        loss = F.mse_loss(pred, target_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ep % 5 == 0:
            print(f"Pretraining epoch {ep}, loss={loss.item():.6f}")

# ---- Insert SAC pipeline classes here (PortfolioEnv, ReplayBuffer, Actor, Critic, SACAgent) ----
# [Reuse definitions from previous notebook]
# For brevity, assume `Critic` and `SACAgent` are defined as before.

# ---- 3. Main training with LQR init ----
def train_with_init(
    num_episodes=500,
    max_steps=100,
    start_steps=1000,
    batch_size=256,
    rho=0.9, t_lambda=10.0, la=1.0
):
    # 1) Compute P_s for init
    P_s = compute_lqr_P(rho, t_lambda, la)
    # 2) Create agent & pretrain critic
    agent = SACAgent(state_dim=6).to(device)
    print("Pretraining critic using LQR approximation...")
    pretrain_critic(agent.critic1, P_s)
    pretrain_critic(agent.critic2, P_s)
    # 3) Continue with full SAC training (randomizing parameters per episode)
    replay_buffer = ReplayBuffer()
    total_steps = 0
    for ep in range(1, num_episodes+1):
        # sample new task params
        rho = np.random.uniform(0.8, 0.99)
        t_lambda = np.random.uniform(1, 1000)
        c = np.random.uniform(0, 10)
        la = 1.0
        env = PortfolioEnv(rho, t_lambda, c, la)
        state = env.reset()
        ep_reward = 0.0
        for t in range(max_steps):
            if total_steps < start_steps:
                action = np.random.uniform(-1,1)
            else:
                action = agent.select_action(state)
            next_state, reward, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state)
            state = next_state
            ep_reward += reward
            total_steps += 1
            if len(replay_buffer) > batch_size:
                agent.update(replay_buffer, batch_size)
        print(f"Episode {ep} Reward {ep_reward:.2f}")
    return agent

# Example usage
def main():
    agent = train_with_init(num_episodes=200, max_steps=100)
    # Then simulate as before

if __name__ == '__main__':
    main()
