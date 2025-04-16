# portfolio_fqi.py – Notebook‑friendly building blocks
"""python
Linear‑feature *Fitted‑Q‑Iteration* (FQI) for a single‑asset trading / inventory‑management problem
=================================================================================================

This file **only contains definitions** (classes, helper functions, etc.) so that you can
`import` or copy‑and‑paste them into a Jupyter / Colab notebook cell and run them
*in any order you like*.  Nothing is executed at import time – there is **no `main()`**
and **no argument‑parser** clutter.

The big picture
--------------
* **Environment** – one risky asset, a private “alpha” signal that follows an AR(1), and a
  quadratic‑plus‑absolute trading cost.  The agent’s state is `(alpha, position)`.
* **Objective / reward per step** is

  ```math
  r_t(x_t) = \alpha_t (p_t + x_t)
             - c\,|x_t|
             - \tfrac12\,t_{\ell}\,x_t^2
             - \lambda\,(p_t + x_t)^2.
  ```

  Here ``x_t`` is the **control** (trade size, i.e. delta‑position).
* **Fitted‑Q‑Iteration** – we approximate the state‑action value ``Q(s, a)`` by a
  *linear* combination of **hand‑crafted features** (see Method #4 in the chat).
* **Features** include polynomials, sign‑kink indicators, interactions with cost
  parameters, and (optionally) a 5 × 5 Radial Basis Function grid on
  `(alpha, position)`.

The code is split into five logical sections:
1. **Imports & utilities**
2. **Environment definition**
3. **Feature engineering helpers**
4. **Linear FQI agent**
5. **Dataset generation helpers** (random parameter sampling + roll‑outs)

Copy the whole file into one notebook cell *or* pick only the bits you need.
"""

# -----------------------------------------------------------------------------
# 1. Imports & utilities
# -----------------------------------------------------------------------------

from __future__ import annotations  # nice for forward references in type hints

# 👉  Standard scientific‑Python stack only – no heavy RL frameworks needed.
import numpy as np
from numpy.linalg import pinv  # Moore–Penrose pseudo‑inverse for least squares
from scipy.spatial.distance import cdist  # quick pairwise distance helper

from dataclasses import dataclass  # lightweight container for parameters & state
from typing import Tuple, List, Callable

# -----------------------------------------------------------------------------
# 2. Environment definition
# -----------------------------------------------------------------------------

@dataclass
class EnvParams:
    """Parameters that *stay constant* during one episode but may vary across episodes."""

    c: float          # linear (absolute) trading‑cost coefficient
    t_l: float        # temporary impact coefficient – penalty ∝ x²
    lam: float        # risk‑aversion on inventory – penalty ∝ (p + x)²
    rho: float        # AR(1) autocorrelation of the alpha signal
    sigma_eps: float  # innovation st.dev. of the alpha signal


@dataclass
class State:
    """The Markov state *visible* to the agent – just alpha and current position."""

    alpha: float      # private alpha signal at time t
    position: float   # current inventory p_t


class TradingEnv:
    """A *stateless* helper that evolves the dynamics from (state, action) → (next_state, reward).

    We keep it stateless so that you can run many roll‑outs in vectorised code or
    parallel workers without worrying about hidden internal variables.
    """

    def __init__(self, params: EnvParams):
        self.p = params  # just store a reference

    # ------------------------------------------------------------------
    def step(
        self,
        state: State,
        action: float,
        rng: np.random.Generator,
    ) -> Tuple[State, float]:
        """Advance *one* time step.

        Parameters
        ----------
        state  : current `(alpha, position)`
        action : trade size `x_t` (can be negative)
        rng    : NumPy random generator for the innovation

        Returns
        -------
        next_state : State after executing `action`
        reward     : scalar reward collected *immediately* at this step
        """

        # ----------------------- reward calculation --------------------
        alpha, p_old = state.alpha, state.position
        p_new = p_old + action  # inventory update

        reward = (
            alpha * p_new                    # PnL from the alpha signal
            - self.p.c   * np.abs(action)    # linear trading cost
            - 0.5        * self.p.t_l * action ** 2  # quadratic temporary cost
            - self.p.lam * p_new ** 2        # inventory risk penalty
        )

        # ----------------------- state transition ----------------------
        eps = rng.normal(0.0, self.p.sigma_eps)
        alpha_new = self.p.rho * alpha + eps  # AR(1) alpha update

        next_state = State(alpha_new, p_new)
        return next_state, reward

# -----------------------------------------------------------------------------
# 3. Feature engineering helpers
# -----------------------------------------------------------------------------

# Quick inline sign function (vectorised for free thanks to NumPy)
_sign = np.sign  # alias for brevity – no need to reinvent the wheel


@dataclass
class RBFGrid:
    """Callable radial‑basis‑function grid φ(a, p) → R^K."""

    centres: np.ndarray  # shape (K, 2) for (alpha, position) centres
    gamma: float         # 1 / (2σ²) – controls bump width

    def __call__(self, a: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Evaluate all RBF bumps at the *batched* inputs.

        Parameters
        ----------
        a, p : *1‑D arrays* of same length = batch size.

        Returns
        -------
        φ : ndarray shape (batch, K)
        """
        # Stack into shape (batch, 2) so that SciPy’s `cdist` can do its magic
        batch = np.column_stack([a, p])

        # Squared Euclidean distance to each centre → shape (batch, K)
        d2 = cdist(batch, self.centres, metric="sqeuclidean")

        # Apply the RBF kernel  exp(-γ ||x - c||²)
        return np.exp(-self.gamma * d2)


# ------------------------------------------------------------------
# Convenience wrapper to *construct* an RBFGrid with evenly‑spaced centres
# ------------------------------------------------------------------

def make_rbf_grid(
    n_per_dim: int = 5,
    a_bounds: Tuple[float, float] = (-2.0, 2.0),
    p_bounds: Tuple[float, float] = (-2.0, 2.0),
    gamma: float = 2.5,
) -> RBFGrid:
    """Generate a uniformly‑spaced n×n grid of RBF centres on (alpha, position)."""
    a_cent = np.linspace(*a_bounds, n_per_dim)
    p_cent = np.linspace(*p_bounds, n_per_dim)
    aa, pp = np.meshgrid(a_cent, p_cent, indexing="ij")  # shape (n, n)
    centres = np.column_stack([aa.ravel(), pp.ravel()])    # flatten → (K, 2)
    return RBFGrid(centres, gamma)


# ------------------------------------------------------------------
# The **feature‑constructor factory**
# ------------------------------------------------------------------

def build_feature_fn(
    *,
    use_rbf: bool = True,
    rbf_gamma: float = 2.5,
    n_rbf: int = 5,
) -> Callable[[State, float, EnvParams], np.ndarray]:
    """Return **phi(s, a)** – a function that maps *one* (state, action) to a feature‑vector.

    The closure captures whether we add RBFs and how many.
    """

    rbf = make_rbf_grid(n_rbf, gamma=rbf_gamma) if use_rbf else None

    # Inner function that will actually be used by the agent
    def phi(state: State, action: float, params: EnvParams) -> np.ndarray:
        """Compute the full feature vector for *one* sample.

        Layout (feel free to tweak / add):
        1. Constant 1                       – bias term
        2. Alpha, position, alpha*position – pure polynomials
        3. Sign‑based kink indicators      – sgn(a), sgn(p), a*sgn(p), p*sgn(a)
        4. Interactions with *cost params* – c*|p|, t_l*p², λ*p², c*|alpha|
        5. (Optional) RBF bumps            – 25 of them if 5×5 grid
        6. Action and interactions         – a, a*alpha, a*pos, …, a²
        """

        a = state.alpha
        p = state.position

        # ------- (1)–(4) handcrafted features -------------------------
        feats: List[float] = [
            1.0,                      # (1) bias
            a, p, a * p,             # (2) polynomials
            _sign(a), _sign(p),      # (3) sign indicators
            a * _sign(p),
            p * _sign(a),
            params.c   * np.abs(p),  # (4) interactions with cost params
            params.t_l * p ** 2,
            params.lam * p ** 2,
            params.c   * np.abs(a),
        ]

        # ------- (5) optional RBF bumps -------------------------------
        if rbf is not None:
            feats.extend( rbf(np.array([a]), np.array([p]))[0] )  # take row 0

        # ------- (6) action‑related features --------------------------
        feats.extend([
            action,
            action * a,
            action * p,
            action * _sign(a),
            action * _sign(p),
            action ** 2,
        ])

        return np.asarray(feats, dtype=np.float64)

    return phi

# -----------------------------------------------------------------------------
# 4. Linear Fitted‑Q‑Iteration agent
# -----------------------------------------------------------------------------

class LinearFQI:
    """A *minimal* implementation of Fitted‑Q‑Iteration with a **linear** model.

    *No deep learning*, just ordinary least squares → extremely fast to train.
    """

    def __init__(
        self,
        feature_fn: Callable[[State, float, EnvParams], np.ndarray],
        action_grid: np.ndarray,
        *,
        gamma: float = 0.99,
    ) -> None:
        self.phi = feature_fn          # feature constructor
        self.action_grid = action_grid # discrete set of candidate trades
        self.gamma = gamma            
        self.A = None                 # weight vector – initialised lazily

    # ------------------------------------------------------------------
    def q(self, state: State, action: float, params: EnvParams) -> float:
        """Predict Q(s,a) for *one* sample (convenience wrapper)."""
        if self.A is None:
            return 0.0  # cold start: zero init → conservative
        return self.phi(state, action, params).dot(self.A)

    # ------------------------------------------------------------------
    def greedy_action(self, state: State, params: EnvParams) -> float:
        """Return **argmax_a Q(s,a)** over the predefined action grid."""
        qs = [self.q(state, a, params) for a in self.action_grid]
        return self.action_grid[int(np.argmax(qs))]

    # ------------------------------------------------------------------
    def iterate(
        self,
        batch_states: np.ndarray,      # shape (N, 2)  – columns = (alpha, position)
        batch_actions: np.ndarray,     # shape (N,)    – executed actions
        batch_next_states: np.ndarray, # shape (N, 2)
        batch_rewards: np.ndarray,     # shape (N,)
        batch_params: List[EnvParams], # length N
    ) -> None:
        """**One** Bellman regression step – core of FQI.

        1. Design matrix `Φ = [φ(s_i, a_i)]`  (shape N×d)
        2. Target `y_i = r_i + γ max_{a'} Q(s'_i, a')`
        3. Solve least‑squares:  w = argmin ||Φ w − y||²  → closed‑form (pseudo‑inv)
        """

        N = len(batch_states)

        # ---- 1. Build the design matrix Φ ---------------------------------
        Phi = np.stack([
            self.phi(State(*batch_states[i]), batch_actions[i], batch_params[i])
            for i in range(N)
        ])  # shape (N, d)

        # ---- 2. Build the target vector y ---------------------------------
        if self.A is None:
            # First iteration – use immediate reward only
            y = batch_rewards.copy()
        else:
            max_q_next = np.empty(N)
            for i in range(N):
                s_next = State(*batch_next_states[i])
                # vectorised/fast version would precompute Φ for the whole grid;
                # here we stay simple and loop – still super fast at N~1e5.
                qs = [
                    self.q(s_next, a, batch_params[i]) for a in self.action_grid
                ]
                max_q_next[i] = np.max(qs)
            y = batch_rewards + self.gamma * max_q_next

        # ---- 3. Ordinary least squares ------------------------------------
        # (ΦᵀΦ)^⁻¹ Φᵀ y  – using Moore–Penrose pseudo‑inverse for safety.
        self.A = pinv(Phi).dot(y)

# -----------------------------------------------------------------------------
# 5. Dataset generation helpers (randomised roll‑out collector)
# -----------------------------------------------------------------------------

################################################################################
# ⚠️  Nothing below is *essential* for FQI itself – you may prefer your own
#     market simulator or a historical replay.  We provide these helpers so that
#     the notebook runs out‑of‑the‑box.
################################################################################

def sample_env_params(rng: np.random.Generator) -> EnvParams:
    """Draw a *random* set of environment parameters from wide, plausible ranges."""
    return EnvParams(
        c         = rng.uniform(0.0, 0.5),
        t_l       = rng.uniform(0.01, 1.0),
        lam       = rng.uniform(0.01, 1.0),
        rho       = rng.uniform(0.4, 0.95),
        sigma_eps = rng.uniform(0.05, 0.3),
    )


def collect_dataset(
    *,
    n_steps: int,
    horizon: int,
    action_grid: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[EnvParams]]:
    """Run many short episodes with *random* parameters to build an FQI training set.

    Returns
    -------
    s      : ndarray (N, 2)  – alpha and position at time t
    a      : ndarray (N,)    – executed action x_t
    s_next : ndarray (N, 2)  – next‑state (t+1)
    r      : ndarray (N,)    – immediate reward
    params : list[EnvParams] length N – the environment parameters *for that step*
    """

    batch_s, batch_a, batch_sp, batch_r, batch_params = [], [], [], [], []

    steps_per_episode = horizon
    n_episodes = n_steps // steps_per_episode

    for _ in range(n_episodes):
        params = sample_env_params(rng)
        env = TradingEnv(params)

        # Start each episode with alpha ~ N(0,1) and zero inventory
        state = State(alpha=rng.normal(0.0, 1.0), position=0.0)

        for _ in range(steps_per_episode):
            # Exploration policy – purely *random* draw from action grid
            action = rng.choice(action_grid)

            next_state, reward = env.step(state, action, rng)

            # Record transition
            batch_s.append([state.alpha, state.position])
            batch_a.append(action)
            batch_sp.append([next_state.alpha, next_state.position])
            batch_r.append(reward)
            batch_params.append(params)

            # Move on
            state = next_state

    # Stack lists into NumPy arrays for efficient vectorised maths later
    return (
        np.asarray(batch_s,  dtype=np.float64),
        np.asarray(batch_a,  dtype=np.float64),
        np.asarray(batch_sp, dtype=np.float64),
        np.asarray(batch_r,  dtype=np.float64),
        batch_params,  # list of dataclass objects – keep as Python list
    )

# -----------------------------------------------------------------------------
# Example *notebook* usage (copy into separate cells, not executed here)
# -----------------------------------------------------------------------------

"""markdown
```python
# --- 1. Put *all the code above* into one cell and run it --------------------
```

```python
# --- 2. Build feature function and agent ------------------------------------
phi = build_feature_fn(use_rbf=True, n_rbf=5, rbf_gamma=2.5)

action_grid = np.linspace(-1.0, 1.0, 21)  # 21 actions from -1 to +1
agent = LinearFQI(phi, action_grid, gamma=0.99)
```

```python
# --- 3. Collect a dataset ----------------------------------------------------
rng = np.random.default_rng(0)
S, A, Sp, R, P = collect_dataset(
    n_steps  = 50_000,
    horizon  = 50,
    action_grid = action_grid,
    rng = rng,
)
```

```python
# --- 4. Train for, say, 20 iterations ---------------------------------------
for it in range(20):
    agent.iterate(S, A, Sp, R, P)
    print(f"Iteration {it+1} complete")
```

```python
# --- 5. Query the greedy policy on a random state ---------------------------
params = sample_env_params(rng)
state  = State(alpha=rng.normal(), position=rng.uniform(-1, 1))
print("Greedy trade:", agent.greedy_action(state, params))
```
"""
