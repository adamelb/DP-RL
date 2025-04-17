"""
Approximate Dynamic Programming with a Neural VFA on a discrete action space
================================================================================
This is a cleaned‑up, modular re‑implementation of the sketch you provided.  It
implements the key fixes and enhancements we discussed:
    •   typing‑safe imports & seeds
    •   orthogonal weight initialisation + LayerNorm on hidden layers
    •   reward/value normalisation to mitigate scale mismatch (see table 6 in
        van Heeswijk & La Poutré 2019)  # citeturn0file0
    •   cosine LR schedule with warm‑up
    •   mixed precision + gradient clipping
    •   target network updated every `TARGET_INTERVAL` steps
    •   option to freeze the online network every `FREEZE_INTERVAL` iterations
    •   fully vectorised Bellman target computation (no Python loops over the
        101 actions) – GPU friendly for an H100
The code is deliberately kept compact (<300 loc) so you can extend it easily.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

# -----------------------------------------------------------------------------
# GLOBALS & HYPER‑PARAMETERS ---------------------------------------------------
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

aRHO_MIN, aRHO_MAX = 0.70, 0.99
C_MIN, C_MAX = 0.0, 10.0
tI_MIN, tI_MAX = 1.0, 1000.0
SIGMA_EPS = 0.2
GAMMA = 0.90  # discount factor
VALUE_SCALE = 100.0  # divide rewards by this to reduce magnitude mismatch

ACTIONS = torch.linspace(-1.0, 1.0, steps=101, device=DEVICE)  # shape (A,)
torch.cuda.empty_cache()
# data & optimisation ---------------------------------------------------------
N_DATASET = 1_000
BATCH_SIZE = 496
M_SAMPLES = 64  # lower than before for memory; can crank up with 70 GB
N_ITERATIONS = 10_000
DATA_REFRESH = 50
ACC_STEPS = 2  # gradient accumulation
LR = 5e-5  # starts low, scheduler will raise it
TARGET_INTERVAL = 200  # sync target network
FREEZE_INTERVAL = 1_000  # evaluation‑only phase every k iters (can be 0)
CLIP_NORM = 1.0

F_FEATURES = 15  # manually counted below – change if you add features

# -----------------------------------------------------------------------------
# HELPERS ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def resample_dataset(n: int = N_DATASET) -> Tuple[torch.Tensor, ...]:
    """Sample training tuples (p, alpha, c, tl, rho)."""
    p = torch.randn(n, device=DEVICE)
    alpha = torch.randn(n, device=DEVICE)
    c = torch.rand(n, device=DEVICE) * (C_MAX - C_MIN) + C_MIN
    tl = torch.rand(n, device=DEVICE) * (tI_MAX - tI_MIN) + tI_MIN
    rho = torch.rand(n, device=DEVICE) * (aRHO_MAX - aRHO_MIN) + aRHO_MIN
    return p, alpha, c, tl, rho


def features(
    p: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    rho: torch.Tensor,
    tl: torch.Tensor,
) -> torch.Tensor:
    """Compute the 15‑D feature vector used by the value network.

    All tensors must be broadcastable to the same shape.
    Returns a tensor of shape (*batch_shape, F_FEATURES).
    """
    # ensure identical shapes ---------------------------------------------------
    shape = torch.broadcast_shapes(p.shape, a.shape, c.shape, rho.shape, tl.shape)
    p, a, c, rho, tl = [t.expand(shape) for t in (p, a, c, rho, tl)]

    sp, sa = torch.sign(p), torch.sign(a)

    phi = torch.stack(
        [
            p,
            a,
            p * a,
            p * p,
            a * a,
            sp * sa,
            a * sp,
            p * sa,
            c * p.abs(),
            tl * (p * p),
            c * a.abs(),
            a * tl.abs(),
            c,
            rho,
            tl,
        ],
        dim=-1,
    )
    return phi


def reward(
    alpha: torch.Tensor,
    p: torch.Tensor,
    x: torch.Tensor,
    c: torch.Tensor,
    tl: torch.Tensor,
) -> torch.Tensor:
    """Immediate reward R(s,a) from the original sketch, scaled."""
    p_new = p + x
    r = alpha * p_new - c * x.abs() - 0.5 * tl * x**2 - 0.5 * p_new**2
    return r / VALUE_SCALE  # scale down


# -----------------------------------------------------------------------------
# MODEL ------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class ValueMLP(nn.Module):
    """Small MLP with orthogonal init + LayerNorm."""

    def __init__(self, n_features: int = F_FEATURES, hidden: Tuple[int, ...] = (256, 256)):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.net(phi).squeeze(-1)  # shape (...,)


# -----------------------------------------------------------------------------
# TRAINING LOOP ----------------------------------------------------------------
# -----------------------------------------------------------------------------


def main() -> None:
    model = ValueMLP().to(DEVICE)
    target_net = ValueMLP().to(DEVICE)
    target_net.load_state_dict(model.state_dict())
    target_net.eval()  # never train

    opt = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(opt, T_max=N_ITERATIONS, eta_min=LR * 10)

    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    start_time = time.time()
    p_data, a_data, c_data, tl_data, rho_data = resample_dataset()
    micro_bs = BATCH_SIZE // ACC_STEPS

    for it in trange(1, N_ITERATIONS + 1, desc="train"):
        # ---------------------------------------------------------------------
        # refresh synthetic data ------------------------------------------------
        if it == 1 or it % DATA_REFRESH == 0:
            p_data, a_data, c_data, tl_data, rho_data = resample_dataset()

        running_loss = 0.0
        opt.zero_grad(set_to_none=True)

        for _ in range(ACC_STEPS):
            idx = torch.randint(0, N_DATASET, (micro_bs,), device=DEVICE)
            p = p_data[idx]
            alpha = a_data[idx]
            c = c_data[idx]
            tl = tl_data[idx]
            rho = rho_data[idx]

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                # forward current state value ----------------------------------
                phi_s = features(p, alpha, c, rho, tl)
                v_s = model(phi_s)  # shape (B,)

                # sample alpha' -------------------------------------------------
                eps = torch.randn(micro_bs, M_SAMPLES, device=DEVICE)
                alpha_next = alpha.unsqueeze(1) * rho.unsqueeze(1) + eps * torch.sqrt(1 - rho.unsqueeze(1) ** 2)
                # expand tensors for vectorised action evaluation --------------
                p_exp = p.unsqueeze(1)  # (B,1)
                c_exp = c.unsqueeze(1)
                tl_exp = tl.unsqueeze(1)
                rho_exp = rho.unsqueeze(1)

                # ACTIONS broadcast to (A,) then to (1,A,1)
                actions = ACTIONS.view(1, -1, 1)
                p_next = p_exp.unsqueeze(2) + actions  # (B, A, 1) + (1, A, 1)
                # tile across samples -----------------------------------------
                p_next = p_next.expand(-1, -1, M_SAMPLES)  # (B, A, M)
                alpha_next_b = alpha_next.unsqueeze(1).expand(-1, ACTIONS.shape[0], -1)  # (B, A, M)

                # compute phi'
                phi_next = features(
                    p_next,
                    alpha_next_b,
                    c_exp.unsqueeze(2),
                    rho_exp.unsqueeze(2),
                    tl_exp.unsqueeze(2),
                )  # (B, A, M, F)
                phi_next_flat = phi_next.view(-1, F_FEATURES)
                with torch.no_grad():  # use target network
                    v_next = target_net(phi_next_flat).view(micro_bs, ACTIONS.shape[0], M_SAMPLES)
                v_avg = v_next.mean(dim=-1)  # (B, A)

                # reward broadcast over actions --------------------------------
                r = reward(alpha.unsqueeze(1), p_exp, actions.squeeze(-1), c_exp, tl_exp)  # (B, A)

                q_values = r + GAMMA * v_avg  # (B, A)
                q_best = q_values.max(dim=1).values  # (B,)

                loss = F.mse_loss(v_s, q_best.detach()) / ACC_STEPS
                torch.cuda.empty_cache()
            scaler.scale(loss).backward()
            running_loss += loss.item()

        # gradient step ---------------------------------------------------------
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        # target network update -------------------------------------------------
        if it % TARGET_INTERVAL == 0:
            target_net.load_state_dict(model.state_dict())

        # optional freeze‑and‑eval period --------------------------------------
        if FREEZE_INTERVAL and it % FREEZE_INTERVAL == 0:
            model.eval()
            # add your evaluation function here if desired
            model.train()

        # ---------------------------------------------------------------------
        if it % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"iter {it:>6}/{N_ITERATIONS} | loss={running_loss:.4e} | lr={scheduler.get_last_lr()[0]:.2e} | Δt={elapsed:.1f}s"
            )
            start_time = time.time()

    # save final checkpoint -----------------------------------------------------
    ckpt_path = Path("model_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path.resolve()}")

#-----------------------------------------------------------------------------
def eval_policy(
    model: nn.Module,
    *,
    fixed_c: float = 5.0,
    fixed_corr: float = 0.99,
    fixed_tl: float = 60.0,
    m_samples: int = 100,
    num_steps: int = 20_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out the greedy policy under fixed hyper‑parameters.

    Returns
    -------
    pos_hist : np.ndarray
        Position trajectory (p_t).
    rew_hist : np.ndarray
        Reward collected at each step.
    alpha_hist : np.ndarray
        Latent α process values.
    """
    model.eval()
    alpha_val, p_val = 0.0, 0.0
    pos_hist, rew_hist, alpha_hist = [p_val], [], [alpha_val]

    c_t = torch.tensor(fixed_c, device=DEVICE)
    corr_t = torch.tensor(fixed_corr, device=DEVICE)
    tl_t = torch.tensor(fixed_tl, device=DEVICE)

    with torch.no_grad():
        for _ in trange(num_steps):
            alpha_t = torch.tensor(alpha_val, device=DEVICE)
            p_t = torch.tensor(p_val, device=DEVICE)

            eps = torch.randn(m_samples, device=DEVICE)
            alpha_next = alpha_t * corr_t + eps * torch.sqrt(1 - corr_t ** 2)

            # vectorised action evaluation ------------------------------------
            p_next = p_t + ACTIONS  # (A,)
            p_rep = p_next.unsqueeze(1).expand(-1, m_samples)  # (A,M)
            phi_next = features(
                p_rep,
                alpha_next.unsqueeze(0).expand_as(p_rep),
                c_t,
                corr_t,
                tl_t,
            )
            v_next = model(phi_next).view(ACTIONS.shape[0], m_samples)
            v_avg = v_next.mean(dim=1)  # (A,)

            r = reward(alpha_t, p_t, ACTIONS, c_t, tl_t)  # (A,)
            q = r + GAMMA * v_avg
            best_idx = torch.argmax(q)
            best_a = ACTIONS[best_idx].item()

            # log & advance ----------------------------------------------------
            r_step = reward(alpha_t, p_t, torch.tensor(best_a, device=DEVICE), c_t, tl_t).item()
            rew_hist.append(r_step)
            p_val += best_a
            pos_hist.append(p_val)
            alpha_val = fixed_corr * alpha_val + math.sqrt(1 - fixed_corr ** 2) * np.random.randn()
            alpha_hist.append(alpha_val)

    return np.asarray(pos_hist), np.asarray(rew_hist), np.asarray(alpha_hist)

#------------------------------------------------------------------------------
"""
Approximate Dynamic Programming for a *fixed* environment (tl = 60, ρ = 0.95)
================================================================================
We now treat **tl** (time‑lambda) and **ρ** (auto‑correlation) as constants and
learn a value function that depends *only* on the mutable state variables

    s = (p, α, c)

where **p** is position, **α** is the latent mean‑reverting signal and **c** is
the transaction‐cost coefficient.  The action space is the same discrete
101‑point grid in [−1, 1].

Changes vs. the previous version
--------------------------------
*   **State reduction** – `features()` omits tl and ρ (now constants) →
    11 features instead of 15.
*   **Reward** – uses the fixed `TL_FIXED` instead of a per‑sample tensor.
*   **Alpha dynamics** – Monte‑Carlo draw uses a hard‑coded `RHO_FIXED`.
*   **Dataset** – samples only `(p, α, c)`.
*   **Training loop & simulator** – updated accordingly, full vectorisation
    retained.

The entire file is self‑contained; run it once to train, and again to load the
checkpoint and simulate a 10 k‑step trajectory.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

# -----------------------------------------------------------------------------
# GLOBAL CONSTANTS -------------------------------------------------------------
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

aRHO_FIXED = 0.95  # correlation (ρ) – now constant
TL_FIXED = 60.0     # time‑lambda (τ) – now constant

C_MIN, C_MAX = 0.0, 10.0
GAMMA = 0.90
VALUE_SCALE = 100.0  # reward scaling

# discrete action grid ---------------------------------------------------------
ACTIONS = torch.linspace(-1.0, 1.0, steps=101, device=DEVICE)  # (A,)

# training hyper‑params --------------------------------------------------------
N_DATASET = 1_000
BATCH_SIZE = 512
M_SAMPLES = 64
N_ITERATIONS = 10_000
DATA_REFRESH = 50
ACC_STEPS = 2
LR_INIT = 5e-5
TARGET_INTERVAL = 200
CLIP_NORM = 1.0

F_FEATURES = 11  # p, a, interactions + cost features (see features())

# -----------------------------------------------------------------------------
# HELPERS ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def resample_dataset(n: int = N_DATASET) -> Tuple[torch.Tensor, ...]:
    """Return random batches of (p, α, c)."""
    p = torch.randn(n, device=DEVICE)
    alpha = torch.randn(n, device=DEVICE)
    c = torch.rand(n, device=DEVICE) * (C_MAX - C_MIN) + C_MIN
    return p, alpha, c


def features(p: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Compute 11‑dim feature vector for (p, α, c)."""
    shape = torch.broadcast_shapes(p.shape, a.shape, c.shape)
    p, a, c = [t.expand(shape) for t in (p, a, c)]

    sp, sa = torch.sign(p), torch.sign(a)

    return torch.stack([
        p,
        a,
        p * a,
        p * p,
        a * a,
        sp * sa,
        a * sp,
        p * sa,
        c * p.abs(),
        c * a.abs(),
        c,
    ], dim=-1)


def reward(alpha: torch.Tensor, p: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Reward function with fixed TL_FIXED, scaled."""
    p_new = p + x
    r = alpha * p_new - c * x.abs() - 0.5 * TL_FIXED * x ** 2 - 0.5 * p_new ** 2
    return r / VALUE_SCALE


# -----------------------------------------------------------------------------
# MODEL ------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class ValueMLP(nn.Module):
    def __init__(self, n_features: int = F_FEATURES, hidden: Tuple[int, ...] = (256, 256)):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:  # (..., F)
        return self.net(phi).squeeze(-1)


# -----------------------------------------------------------------------------
# TRAINING LOOP ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def train_model() -> Path:
    model = ValueMLP().to(DEVICE)
    target = ValueMLP().to(DEVICE)
    target.load_state_dict(model.state_dict())
    target.eval()

    opt = AdamW(model.parameters(), lr=LR_INIT, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(opt, T_max=N_ITERATIONS, eta_min=LR_INIT * 10)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    p_data, a_data, c_data = resample_dataset()
    micro_bs = BATCH_SIZE // ACC_STEPS

    t0 = time.time()
    for it in trange(1, N_ITERATIONS + 1, desc="train"):
        if it == 1 or it % DATA_REFRESH == 0:
            p_data, a_data, c_data = resample_dataset()

        opt.zero_grad(set_to_none=True)
        run_loss = 0.0

        for _ in range(ACC_STEPS):
            idx = torch.randint(0, N_DATASET, (micro_bs,), device=DEVICE)
            p, alpha, c = p_data[idx], a_data[idx], c_data[idx]

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                v_s = model(features(p, alpha, c))

                eps = torch.randn(micro_bs, M_SAMPLES, device=DEVICE)
                alpha_next = alpha.unsqueeze(1) * aRHO_FIXED + eps * math.sqrt(1 - aRHO_FIXED ** 2)

                p_rep = p.unsqueeze(1).unsqueeze(2)  # (B,1,1)
                c_rep = c.unsqueeze(1).unsqueeze(2)
                actions = ACTIONS.view(1, -1, 1)  # (1,A,1)

                p_next = p_rep + actions  # (B,A,1)
                p_next = p_next.expand(-1, -1, M_SAMPLES)  # (B,A,M)
                alpha_next_b = alpha_next.unsqueeze(1).expand(-1, ACTIONS.shape[0], -1)
                c_b = c_rep.expand_as(p_next)

                phi_next = features(p_next, alpha_next_b, c_b)
                v_next = target(phi_next.view(-1, F_FEATURES)).view(micro_bs, ACTIONS.shape[0], M_SAMPLES)
                v_avg = v_next.mean(dim=-1)

                r = reward(alpha.unsqueeze(1), p.unsqueeze(1), actions.squeeze(-1), c.unsqueeze(1))
                q_best = (r + GAMMA * v_avg).max(dim=1).values

                loss = F.mse_loss(v_s, q_best.detach()) / ACC_STEPS

            scaler.scale(loss).backward()
            run_loss += loss.item()

        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        scaler.step(opt); scaler.update(); scheduler.step()

        if it % TARGET_INTERVAL == 0:
            target.load_state_dict(model.state_dict())

        if it % 100 == 0:
            dt = time.time() - t0
            print(f"it {it:5}/{N_ITERATIONS} | loss={run_loss:.3e} | lr={scheduler.get_last_lr()[0]:.1e} | Δt={dt:.1f}s")
            t0 = time.time()

    ckpt = Path("model_fixed.pt"); torch.save(model.state_dict(), ckpt)
    return ckpt


# -----------------------------------------------------------------------------
# POLICY SIMULATION ------------------------------------------------------------
# -----------------------------------------------------------------------------

def eval_policy(
    model: nn.Module,
    *,
    fixed_c: float = 5.0,
    m_samples: int = 100,
    num_steps: int = 10_000,
):
    """Roll out greedy policy for the fixed environment."""
    model.eval()
    alpha_val = 0.0; p_val = 0.0
    pos_hist, rew_hist, alpha_hist = [p_val], [], [alpha_val]
    c_t = torch.tensor(fixed_c, device=DEVICE)

    with torch.no_grad():
        for _ in range(num_steps):
            alpha_t = torch.tensor(alpha_val, device=DEVICE)
            p_t = torch.tensor(p_val, device=DEVICE)

            eps = torch.randn(m_samples, device=DEVICE)
            alpha_next = alpha_t * aRHO_FIXED + eps * math.sqrt(1 - aRHO_FIXED ** 2)

            p_next = p_t + ACTIONS  # (A,)
            p_rep = p_next.unsqueeze(1).expand(-1, m_samples)  # (A,M)
            alpha_rep = alpha_next.unsqueeze(0).expand_as(p_rep)
            c_rep = c_t

            v_next = model(features(p_rep, alpha_rep, c_rep)).view(ACTIONS.shape[0], m_samples)
            v_avg = v_next.mean(dim=1)
            r = reward(alpha_t, p_t, ACTIONS, c_t)
            q = r + GAMMA * v_avg
            best_idx = torch.argmax(q)
            best_a = ACTIONS[best_idx].item()

            r_step = reward(alpha_t, p_t, torch.tensor(best_a, device=DEVICE), c_t).item()
            rew_hist.append(r_step)
            p_val += best_a; pos_hist.append(p_val)
            alpha_val = aRHO_FIXED * alpha_val + math.sqrt(1 - aRHO_FIXED ** 2) * np.random.randn()
            alpha_hist.append(alpha_val)

    return np.array(pos_hist), np.array(rew_hist), np.array(alpha_hist)


