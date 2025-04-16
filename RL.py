'''Portfolio execution environment and SAC agent pipeline.

We model state s = [alpha, p, rho, t_lambda, c, la]
Dynamics:
  alpha_{t+1} = rho*alpha_t + sqrt(1-rho^2)*epsilon_t
  p_{t+1} = p_t + x_t
Reward: r = alpha*(p + x) - c*abs(x) - 0.5*t_lambda*x**2 - la*(p + x)**2

This script implements:
- PortfolioEnv: the environment
- ReplayBuffer: for off-policy learning
- Actor, Critic network classes
- SACAgent class: encapsulates SAC training
- main(): sampling episodes with random parameters, training loop
Hyperparameters and how to change are at the top under CONFIGURATION.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CONFIGURATION: change these to tune the agent and environment
STATE_DIM = 2 + 4  # alpha, p plus rho, t_lambda, c, la
HIDDEN_DIM = 128
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005  # for target network updates
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
START_STEPS = 10000
MAX_EPISODES = 1000
MAX_STEPS = 100
POLICY_UPDATES_PER_STEP = 1

# Parameter priors for sampling new tasks (rho, t_lambda, c, la)
RHO_RANGE = (0.8, 0.99)
TLAMBDA_RANGE = (0.1, 1.0)
C_RANGE = (1e-4, 1e-2)
LA_RANGE = (1e-4, 1e-2)

class PortfolioEnv:
    def __init__(self, rho, t_lambda, c, la):
        self.rho = rho
        self.t_lambda = t_lambda
        self.c = c
        self.la = la
        self.reset()

    def reset(self):
        self.alpha = np.random.randn()
        self.p = 0.0
        return self._get_state()

    def step(self, x):
        reward = (self.alpha * (self.p + x)
                  - self.c * abs(x)
                  - 0.5 * self.t_lambda * x**2
                  - self.la * (self.p + x)**2)
        self.p += x
        noise = np.random.randn()
        self.alpha = self.rho * self.alpha + np.sqrt(1 - self.rho**2) * noise
        return self._get_state(), reward, False

    def _get_state(self):
        return np.array([self.alpha, self.p, self.rho, self.t_lambda, self.c, self.la], dtype=np.float32)

class ReplayBuffer:
    def __init__(self, max_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (torch.tensor(states, dtype=torch.float32, device=DEVICE),
                torch.tensor(actions, dtype=torch.float32, device=DEVICE).unsqueeze(-1),
                torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(-1),
                torch.tensor(next_states, dtype=torch.float32, device=DEVICE))

    def __len__(self):
        return len(self.buffer)

class Critic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

class Actor(nn.Module):
    def __init__(self, state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, action_limit=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, 1)
        self.log_std_layer = nn.Linear(hidden_dim, 1)
        self.action_limit = action_limit

    def forward(self, state):
        x = self.net(state)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, state):
        mu, std = self(state)
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        x_tanh = torch.tanh(x)
        action = x_tanh if self.action_limit is None else x_tanh * self.action_limit
        log_prob = dist.log_prob(x) - torch.log(1 - x_tanh.pow(2) + 1e-6)
        return action, log_prob

class SACAgent:
    def __init__(self, state_dim=STATE_DIM, action_limit=None):
        self.actor = Actor(state_dim, HIDDEN_DIM, action_limit).to(DEVICE)
        self.critic1 = Critic(state_dim, HIDDEN_DIM).to(DEVICE)
        self.critic2 = Critic(state_dim, HIDDEN_DIM).to(DEVICE)
        self.target_critic1 = Critic(state_dim, HIDDEN_DIM).to(DEVICE)
        self.target_critic2 = Critic(state_dim, HIDDEN_DIM).to(DEVICE)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=LR_CRITIC)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=LR_CRITIC)
        self.log_alpha = torch.zeros(1, device=DEVICE, requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=LR_ACTOR)
        self.target_entropy = -1.0

    def select_action(self, state, evaluate=False):
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        if evaluate:
            mu, _ = self.actor(state_t)
            return torch.tanh(mu).item()
        action, _ = self.actor.sample(state_t)
        return action.detach().cpu().item()

    def update(self, buffer):
        state, action, reward, next_state = buffer.sample()
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            tgt_q1 = self.target_critic1(next_state, next_action)
            tgt_q2 = self.target_critic2(next_state, next_action)
            tgt_q = torch.min(tgt_q1, tgt_q2) - torch.exp(self.log_alpha) * next_log_prob
            q_target = reward + GAMMA * tgt_q
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        self.critic1_optim.zero_grad(); critic1_loss.backward(); self.critic1_optim.step()
        self.critic2_optim.zero_grad(); critic2_loss.backward(); self.critic2_optim.step()
        action_new, log_prob = self.actor.sample(state)
        q1_new = self.critic1(state, action_new)
        q2_new = self.critic2(state, action_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (torch.exp(self.log_alpha) * log_prob - q_new).mean()
        self.actor_optim.zero_grad(); actor_loss.backward(); self.actor_optim.step()
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad(); alpha_loss.backward(); self.alpha_optim.step()
        # Soft update
        for target, source in [(self.target_critic1, self.critic1),
                               (self.target_critic2, self.critic2)]:
            for t_param, s_param in zip(target.parameters(), source.parameters()):
                t_param.data.copy_(t_param.data * (1 - TAU) + s_param.data * TAU)

def main():
    agent = SACAgent(STATE_DIM)
    buffer = ReplayBuffer()
    total_steps = 0
    for episode in range(MAX_EPISODES):
        rho = np.random.uniform(*RHO_RANGE)
        t_lambda = np.random.uniform(*TLAMBDA_RANGE)
        c = np.random.uniform(*C_RANGE)
        la = np.random.uniform(*LA_RANGE)
        env = PortfolioEnv(rho, t_lambda, c, la)
        state = env.reset()
        ep_reward = 0.0
        for step in range(MAX_STEPS):
            if total_steps < START_STEPS:
                action = np.random.uniform(-1.0, 1.0)
            else:
                action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state)
            state = next_state
            ep_reward += reward
            total_steps += 1
            if len(buffer) > BATCH_SIZE:
                for _ in range(POLICY_UPDATES_PER_STEP):
                    agent.update(buffer)
            if done:
                break
        print(f'Episode {episode} Reward {ep_reward:.3f}')
    torch.save(agent.actor.state_dict(), 'sac_actor.pth')
    torch.save(agent.critic1.state_dict(), 'sac_critic1.pth')
    torch.save(agent.critic2.state_dict(), 'sac_critic2.pth')

if __name__ == '__main__':
    main()
