# dqn_portfolio.py – Deep‑Q‑Learning with structured Q(s,x) and richer network
"""python
Deep‑Q‑Learning for single‑asset trading with the *structured* quadratic‑plus‑|x| value
======================================================================================

Enhancements vs. first version
------------------------------
* **Richer network** – Residual MLP trunk (3×512) with LayerNorm → three heads
  for *(f, g, z_raw)*.  Still lightweight enough for a single GPU but models
  sharper nonlinearities in `(α, p, ρ, c, t_ℓ)`.
* **NoisyLinear** layers in the first two hidden blocks (Fortunato et al., 2018)
  for smarter exploration (optional – turn off via flag).
* **Policy roll‑out helper** – `evaluate_policy()` returns the *full* time series
  of rewards, positions, and alphas for fixed parameters.
* Same *structured* Q‑formula

```math
Q_θ(s,x)=f_θ(s) + g_θ(s)|x| − z_θ(s)x², \;  z_θ(s)=\text{softplus}(z^\text{raw}_θ(s)).
```

Copy this file into a module or notebook cell; train with `train_dqn`, then
call `evaluate_policy` with *any* fixed `(c,t_ℓ,ρ)` triplet.
"""

# -----------------------------------------------------------------------------
# Imports & device -------------------------------------------------------------
# -----------------------------------------------------------------------------

import math, random, pathlib
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Environment ------------------------------------------------------------------
# -----------------------------------------------------------------------------

@dataclass
class EnvParams:
    c: float; t_l: float; lam: float; rho: float; sigma_eps: float

@dataclass
class State:
    alpha: float; position: float

class TradingEnv:
    """Scalar, non‑vectorised environment suitable for DQN roll‑outs."""
    def __init__(self, params: EnvParams):
        self.p=params; self.state=State(0.,0.)
    def reset(self, alpha0=0., position0=0.):
        self.state=State(alpha0, position0);return self.state
    def step(self, action: float):
        s=self.state
        p_new=s.position+action
        r=s.alpha*p_new - self.p.c*abs(action) - 0.5*self.p.t_l*action**2 - self.p.lam*p_new**2
        eps=np.random.normal(0., self.p.sigma_eps)
        self.state=State(self.p.rho*s.alpha+eps, p_new)
        return self.state, r

# -----------------------------------------------------------------------------
# Replay buffer -----------------------------------------------------------------
# -----------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity:int): self.buf=deque(maxlen=capacity)
    def push(self, tr): self.buf.append(tr)
    def sample(self, bs:int):
        batch=random.sample(self.buf, bs)
        s,a,r,sp,d=map(np.array, zip(*batch))
        return (torch.tensor(s,  dtype=torch.float32, device=DEVICE),
                torch.tensor(a,  dtype=torch.float32, device=DEVICE),
                torch.tensor(r,  dtype=torch.float32, device=DEVICE),
                torch.tensor(sp, dtype=torch.float32, device=DEVICE),
                torch.tensor(d,  dtype=torch.float32, device=DEVICE))
    def __len__(self): return len(self.buf)

# -----------------------------------------------------------------------------
# Noisy layer for exploration ---------------------------------------------------
# -----------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet layer (σ_init=0.5)."""
    def __init__(self, in_f, out_f, sigma_init=.5):
        super().__init__()
        self.mu_w = nn.Parameter(torch.empty(out_f,in_f)); self.sigma_w=nn.Parameter(torch.empty(out_f,in_f))
        self.mu_b = nn.Parameter(torch.empty(out_f));     self.sigma_b=nn.Parameter(torch.empty(out_f))
        self.register_buffer('eps_w', torch.empty(out_f,in_f))
        self.register_buffer('eps_b', torch.empty(out_f))
        self.reset_parameters(sigma_init)
        self.sample_noise()
    def reset_parameters(self, sigma_init):
        bound=1/math.sqrt(self.mu_w.size(1))
        nn.init.uniform_(self.mu_w,-bound,bound); nn.init.uniform_(self.mu_b,-bound,bound)
        nn.init.constant_(self.sigma_w, sigma_init*bound); nn.init.constant_(self.sigma_b, sigma_init*bound)
    def f(self,x): return torch.sign(x)*torch.sqrt(torch.abs(x))
    def sample_noise(self):
        eps_in  = self.f(torch.randn(self.mu_w.size(1), device=self.mu_w.device))
        eps_out = self.f(torch.randn(self.mu_w.size(0), device=self.mu_w.device))
        self.eps_w.copy_(eps_out.ger(eps_in)); self.eps_b.copy_(eps_out)
    def forward(self,x):
        if self.training: w=self.mu_w+self.sigma_w*self.eps_w; b=self.mu_b+self.sigma_b*self.eps_b
        else: w=self.mu_w; b=self.mu_b
        return F.linear(x,w,b)

# -----------------------------------------------------------------------------
# Neural network – residual trunk, three heads ---------------------------------
# -----------------------------------------------------------------------------

class ResidBlock(nn.Module):
    def __init__(self, dim, noisy=True):
        super().__init__()
        lin = NoisyLinear if noisy else nn.Linear
        self.fc1=lin(dim,dim); self.ln1=nn.LayerNorm(dim)
        self.fc2=lin(dim,dim); self.ln2=nn.LayerNorm(dim)
    def forward(self,x):
        h=F.relu(self.ln1(self.fc1(x)))
        h=self.ln2(self.fc2(h))
        return F.relu(x+h)

class DQN(nn.Module):
    def __init__(self, state_dim=5, hidden=512, n_blocks=3, noisy=True):
        super().__init__()
        lin = NoisyLinear if noisy else nn.Linear
        self.inp = lin(state_dim, hidden)
        self.trunk = nn.Sequential(*[ResidBlock(hidden,noisy) for _ in range(n_blocks)])
        self.head_f = lin(hidden,1)
        self.head_g = lin(hidden,1)
        self.head_z = lin(hidden,1)
    def forward(self,s):
        x=F.relu(self.inp(s))
        x=self.trunk(x)
        f=self.head_f(x).squeeze(-1)
        g=self.head_g(x).squeeze(-1)
        z=F.softplus(self.head_z(x).squeeze(-1))+1e-4
        return f,g,z

# -----------------------------------------------------------------------------
# Q helper ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

def q_from_components(f,g,z,action_grid):
    a=torch.as_tensor(action_grid,dtype=torch.float32,device=f.device)
    return f.unsqueeze(1)+g.unsqueeze(1)*torch.abs(a)+(-z.unsqueeze(1)*(a**2))

# -----------------------------------------------------------------------------
# Training loop ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def train_dqn(*,env_params:EnvParams,episodes:int=800,steps_per_episode:int=250,
              action_grid:np.ndarray=np.linspace(-1,1,41),gamma:float=.99,
              buffer_capacity:int=100_000,batch_size:int=256,lr:float=2e-4,
              target_update:int=500,epsilon_start:float=1.,epsilon_final:float=.05,
              epsilon_decay:int=20_000,save_path:pathlib.Path|str="./checkpoints/dqn_portfolio.pt",
              noisy=True):
    save_path=pathlib.Path(save_path); save_path.parent.mkdir(parents=True,exist_ok=True)
    env=TradingEnv(env_params)
    policy=DQN(noisy=noisy).to(DEVICE); target=DQN(noisy=noisy).to(DEVICE)
    target.load_state_dict(policy.state_dict()); target.eval()
    opt=torch.optim.Adam(policy.parameters(), lr=lr)
    buf=ReplayBuffer(buffer_capacity)
    eps=epsilon_start; decay_rate=(epsilon_start-epsilon_final)/epsilon_decay
    gstep=0
    for ep in range(episodes):
        s=env.reset(alpha0=np.random.normal(),position0=0.)
        for _ in range(steps_per_episode):
            gstep+=1
            state_vec=np.array([s.position,s.alpha,env.p.rho,env.p.c,env.p.t_l],dtype=np.float32)
            if noisy: policy.apply(lambda m: isinstance(m,NoisyLinear) and m.sample_noise())
            if np.random.rand()<eps:
                act=float(np.random.choice(action_grid))
            else:
                with torch.no_grad():
                    f,g,z=policy(torch.tensor(state_vec,device=DEVICE).unsqueeze(0))
                    qvals=q_from_components(f,g,z,action_grid)
                    act=float(action_grid[int(qvals.argmax())])
            s_next,r=env.step(act)
            buf.push((state_vec,act,r,np.array([s_next.position,s_next.alpha,env.p.rho,env.p.c,env.p.t_l],dtype=np.float32),0.))
            s=s_next
            if len(buf)>=batch_size:
                sb,ab,rb,spb,db=buf.sample(batch_size)
                f,g,z=policy(sb); q_pred=f+g*torch.abs(ab)-z*ab**2
                with torch.no_grad():
                    f2,g2,z2=target(spb)
                    q_next=q_from_components(f2,g2,z2,action_grid).max(dim=1)[0]
                    q_t=rb+gamma*q_next
                loss=F.mse_loss(q_pred,q_t)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(),10.); opt.step()
            eps=max(epsilon_final, eps-decay_rate)
            if gstep%target_update==0: target.load_state_dict(policy.state_dict())
        torch.save({"model_state_dict":policy.state_dict(),"eps":eps},save_path)
        print(f"Ep{ep+1:3d}/{episodes} buf={len(buf)} ε={eps:.3f}")
    return policy

# -----------------------------------------------------------------------------
# Roll‑out helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def evaluate_policy(model:DQN, *, env_params:EnvParams, action_grid:np.ndarray,
                    num_steps:int=10_000, alpha0:float=0., position0:float=0.):
    """Run greedy policy for fixed parameters.  Returns arrays (rewards, positions, alphas)."""
    env=TradingEnv(env_params); s=env.reset(alpha0, position0)
    rewards,positions,alphas=[],[s.position],[s.alpha]
    with torch.no_grad():
        for _ in range(num_steps):
            state_vec=torch.tensor([[s.position,s.alpha,env.p.rho,env.p.c,env.p.t_l]],device=DEVICE)
            f,g,z=model(state_vec)
            q=q_from_components(f,g,z,action_grid)
            act=float(action_grid[int(q.argmax())])
            s,r=env.step(act)
            rewards.append(r); positions.append(s.position); alphas.append(s.alpha)
    return np.array(rewards), np.array(positions), np.array(alphas)

# -----------------------------------------------------------------------------
# Load helper ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def load_model(path:str|pathlib.Path, noisy=True):
    chk=torch.load(path,map_location=DEVICE)
    model=DQN(noisy=noisy).to(DEVICE); model.load_state_dict(chk["model_state_dict"]); model.eval(); return model

# -----------------------------------------------------------------------------
# Quick demo (split into notebook cells) ---------------------------------------
# -----------------------------------------------------------------------------
"""markdown
```python
params=EnvParams(c=8., t_l=75., lam=0.5, rho=0.94, sigma_eps=0.2)
policy=train_dqn(env_params=params, episodes=400, noisy=True)
```

```python
rewards,pos,alpha=evaluate_policy(policy, env_params=params, action_grid=np.linspace(-1,1,41), num_steps=20_000)
print(rewards.sum())
```
"""
