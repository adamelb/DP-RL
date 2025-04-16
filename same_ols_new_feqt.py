# notebook_fqi_montecarlo.py
"""python
End‑to‑end Monte‑Carlo fitted‑value iteration *à la* your prototype
===================================================================

This notebook‑style module keeps **exactly the step‑by‑step structure** you
posted, but the feature set is upgraded to the richer basis we discussed:

* constant 1
* p, α, p α, p², α²
* sgn‑based kinks            – sgn(p), sgn(α), α sgn(p), p sgn(α)
* cost‑interactions          – c|p|, t_λ p², c|α|
* raw parameters as features – c, ρ (corr), t_λ (so the network can “see” them)

That makes **F = 17 features in total**; you can add/remove without touching the
rest of the code – everything is vectorised.

The learning loop is the same Monte‑Carlo regression‑based scheme you had:

1. Sample `n_samples` triples `(p, α, next α)` ***and*** the environment
   parameters `(c, t_λ, ρ)`.
2. Enumerate **all** actions from a discrete grid, Monte‑Carlo average the
   next‑state value with `m_samples` draws of `next α`.
3. Build targets `Q_best = max_a [ r + γ E[V(s')] ]`.
4. **OLS** fit `V(s) = φ(s)·θ` with the new feature matrix.
5. Repeat for `num_iterations`.

Finally a *fixed‑parameter* evaluation roll‑out reproduces the chain you
sketched.

All tensors follow your original shapes:

```text
  state arrays           : (n_samples, 1, 1)
  tiled action dimension : (    1    , n_actions, 1)
  Monte‑Carlo dimension  : (    1    , n_actions, m_samples)
```

Nothing is executed on import – dump the whole block into a Jupyter cell and
call the bottom‑section **examples** in separate cells.
"""

# -----------------------------------------------------------------------------
# 0. Imports & random‑helper
# -----------------------------------------------------------------------------

import numpy as np
from numpy.linalg import pinv  # for ordinary least squares

rng_global = np.random.default_rng(0)  # change seed if needed

# -----------------------------------------------------------------------------
# 1.  *****  Monte‑Carlo SAMPLER  *********************************************
# -----------------------------------------------------------------------------

def sampler(
    *,
    n_samples: int,
    m_samples: int,
    mult: float = 1.0,
    dist: str = "normal",
):
    """Draw *n_samples* independent states and parameters.

    Shapes follow your prototype exactly so you can plug the arrays straight
    into the rest of the pipeline.

    Returns
    -------
    p             : (n_samples, 1, 1)
    α             : (n_samples, 1, 1)
    next_α        : (n_samples, 1, m_samples)  – m Monte‑Carlo draws *each*
    c_vals        : (n_samples, 1, 1)
    tλ_vals       : (n_samples, 1, 1)
    corr_vals     : (n_samples, 1, 1)
    """

    if dist == "normal":
        p = rng_global.standard_normal((n_samples, 1, 1)) * mult
        alpha = rng_global.standard_normal((n_samples, 1, 1)) * mult
    elif dist == "uniform":
        p = rng_global.uniform(-mult, mult, size=(n_samples, 1, 1))
        alpha = rng_global.uniform(-mult, mult, size=(n_samples, 1, 1))
    else:
        raise ValueError("dist must be 'normal' or 'uniform'")

    # AR(1) autocorr ρ ∈ (-1,1)  – choose wide-ish positive range
    uni = rng_global.uniform(5.0, 100.0, size=(n_samples, 1, 1))
    corr_vals = -1 + 1 / uni  # pushes values into (-1,1)

    # trading‑cost parameters
    c_vals = rng_global.uniform(0.0, 20.0, size=(n_samples, 1, 1))
    tλ_vals = rng_global.uniform(1.0, 1000.0, size=(n_samples, 1, 1))

    # AR(1) innovation draw for *m_samples* scenarios per state
    eps = rng_global.standard_normal((n_samples, 1, m_samples))
    next_alpha = alpha * corr_vals + np.sqrt(1.0 - corr_vals**2) * eps

    # Optional deterministic starting point (0,0) so first sample is neat
    p[0, 0, 0] = 0.0
    alpha[0, 0, 0] = 0.0

    return p, alpha, next_alpha, c_vals, tλ_vals, corr_vals

# -----------------------------------------------------------------------------
# 2. Feature builders – training, evaluation, next‑state variants
# -----------------------------------------------------------------------------

# Helper lambdas for sign so we keep everything vectorised
_sgn = np.sign

# ---- full feature set length -------------------------------------------------
_F = 17  # 1 + 6 basic + 4 sign/kink + 3 interactions + 3 raw params


def _assemble_features(p, alpha, c, corr, tl):
    """Return φ with **last axis = F** – assume arrays already broadcastable."""
    # Basic polynomials
    f_const = np.ones_like(p)
    f_p     = p
    f_a     = alpha
    f_pa    = p * alpha
    f_p2    = p ** 2
    f_a2    = alpha ** 2

    # Sign/kink terms
    f_sp   = _sgn(p)
    f_sa   = _sgn(alpha)
    f_a_sp = alpha * _sgn(p)
    f_p_sa = p * _sgn(alpha)

    # Cost interactions
    f_c_abs_p = c  * np.abs(p)
    f_tl_p2   = tl * p ** 2
    f_c_abs_a = c  * np.abs(alpha)

    # Raw parameters as separate coordinates (broadcast to state shape)
    f_c    = c
    f_corr = corr
    f_tl   = tl

    return np.stack([
        f_const, f_p, f_a, f_pa, f_p2, f_a2,
        f_sp, f_sa, f_a_sp, f_p_sa,
        f_c_abs_p, f_tl_p2, f_c_abs_a,
        f_c, f_corr, f_tl,
        np.zeros_like(p),  # spare slot if you want to add extra feature later
    ], axis=-1)


def get_features(p, alpha, c_vals, corr_vals, tl_vals):
    """φ(s) for the **current** state (shape …×F)."""
    return _assemble_features(p, alpha, c_vals, corr_vals, tl_vals)


def get_features_next(p_next, alpha_next, c_vals, corr_vals, tl_vals):
    """φ(s′) for each Monte‑Carlo next‑state sample (shape …×F)."""
    return _assemble_features(p_next, alpha_next, c_vals, corr_vals, tl_vals)


def get_features_eval(p_arr, alpha_arr, fixed_c, fixed_corr, fixed_tl):
    """Convenience wrapper for the *evaluation* roll‑out (scalar params)."""
    c   = fixed_c   * np.ones_like(p_arr)
    tl  = fixed_tl  * np.ones_like(p_arr)
    rho = fixed_corr * np.ones_like(p_arr)
    return _assemble_features(p_arr, alpha_arr, c, rho, tl)

# -----------------------------------------------------------------------------
# 3. Simple helpers – reward & V‑evaluation
# -----------------------------------------------------------------------------

def evaluate(theta, feat):
    """Vectorised 〈φ,θ〉 with broadcasting over any leading dims."""
    return np.tensordot(feat, theta, axes=([-1], [0]))  # output shape = feat.shape[:-1]


def get_reward(alpha, p, actions, c_vals, tl_vals):
    """Immediate reward R(s, x).  Shapes are broadcast‑compatible by design."""
    p_new = p + actions
    return (
        alpha * p_new
        - c_vals * np.abs(actions)
        - 0.5 * tl_vals * actions ** 2
    )

# -----------------------------------------------------------------------------
# 4. *******  Fitted‑Value outer loop  *****************************************
# -----------------------------------------------------------------------------

def fitted_value_iteration(
    *,
    n_samples      = 10_000,
    m_samples      = 100,
    num_iterations = 300,
    gamma          = 0.99,
    action_grid    = np.arange(-100, 100).reshape(1, -1, 1) * 1e-2,
):
    """Main training routine – returns learned θ (shape (F,))."""

    # Initialise parameter vector θ with zeros
    theta = np.zeros(_F)

    for it in range(num_iterations):
        # ---- 1. Monte‑Carlo sampling -----------------------------------
        p, alpha, next_alpha, c_vals, tl_vals, corr_vals = sampler(
            n_samples=n_samples, m_samples=m_samples
        )

        # ---- 2. Enumerate candidate next positions for *each* action ----
        # Shapes:
        #   p.shape            = (N,1,1)
        #   actions.shape      = (1,A,1)
        # ⇒ candidate_p        = (N,A,1)
        candidate_p = p + action_grid

        # Tile to add the Monte‑Carlo axis → shape (N, A, M)
        p_next_tiled     = np.tile(candidate_p, (1, 1, m_samples))
        alpha_next_tiled = np.tile(next_alpha,  (1, action_grid.shape[1], 1))
        c_tiled   = np.tile(c_vals,  (1, action_grid.shape[1], m_samples))
        tl_tiled  = np.tile(tl_vals, (1, action_grid.shape[1], m_samples))
        corr_tiled = np.tile(corr_vals, (1, action_grid.shape[1], m_samples))

        # ---- 3. φ(s′) and Monte‑Carlo expectation ----------------------
        feat_next = get_features_next(p_next_tiled, alpha_next_tiled,
                                      c_tiled, corr_tiled, tl_tiled)
        V_candidates = evaluate(theta, feat_next)              # (N,A,M)
        V_next_avg   = V_candidates.mean(axis=2, keepdims=True)  # (N,A,1)

        # ---- 4. Immediate reward + discounted value --------------------
        R_immediate = get_reward(alpha, p, action_grid, c_vals, tl_vals)  # (N,A,1)
        Qvals = R_immediate + gamma * V_next_avg                         # (N,A,1)

        # ---- 5. Choose best action and build regression targets -------
        Q_best = Qvals.max(axis=1)          # (N,1)

        # ---- 6. Current‑state features Φ ------------------------------
        feat_current = get_features(p, alpha, c_vals, corr_vals, tl_vals)  # (N,1,1,F)
        Φ = feat_current[:, 0, 0, :]  # flatten leading dims → (N,F)

        # ---- 7. Ordinary least squares θ ← argmin ||Φ θ − Q_best||² ----
        theta = pinv(Φ).dot(Q_best.squeeze())

        if (it + 1) % 50 == 0:
            print(f"Iteration {it+1:3d}/{num_iterations} done – ‖θ‖={np.linalg.norm(theta):.3f}")

    return theta

# -----------------------------------------------------------------------------
# 5. Fixed‑parameter evaluation roll‑out (your ‘chain_sample’ loop)
# -----------------------------------------------------------------------------

def eval_fixed_policy(
    theta,
    *,
    fixed_c     = 8.0,
    fixed_corr  = 0.94,
    fixed_tl    = 75.0,
    gamma       = 0.99,
    action_grid = np.arange(-100, 100).reshape(1, -1, 1) * 1e-2,
    num_steps   = 100_000,
):
    """Generate one trajectory following the greedy policy w.r.t learned θ."""
    alpha_val = 0.0  # start at zero signal
    p = 0.0         # start flat
    pos_list, reward_list = [p], []

    for _ in range(num_steps):
        # Prepare scalar → shape (1,1,1) so helpers work unchanged
        alpha_cur = np.array([[alpha_val]])
        p_cur     = np.array([[p]])

        # ---- enumerate candidate trades --------------------------------
        candidate_p = p_cur + action_grid            # (1,A,1)
        # single draw of next‑alpha for each *candidate* action so that we
        # use the same sampling strategy as fitted iteration
        next_alpha = alpha_cur * fixed_corr + np.sqrt(1 - fixed_corr**2) * rng_global.standard_normal((1, 1, 1))
        alpha_next_tiled = np.tile(next_alpha, (1, action_grid.shape[1], 1))
        p_next_tiled     = np.tile(candidate_p, (1, 1, 1))

        feat_next = get_features_eval(p_next_tiled, alpha_next_tiled,
                                      fixed_c, fixed_corr, fixed_tl)
        V_next = evaluate(theta, feat_next)           # (1,A,1)
        R_immed = get_reward(alpha_cur, p_cur, action_grid, fixed_c, fixed_tl)
        Q = R_immed + gamma * V_next

        # Greedy action index and value
        idx = int(np.argmax(Q[0, :, 0]))
        optimal_action = action_grid[0, idx, 0]

        # Apply action & record reward
        r_step = get_reward(alpha_cur, p_cur, np.array([[optimal_action]]), fixed_c, fixed_tl).item()
        reward_list.append(r_step)

        p += optimal_action  # inventory update
        pos_list.append(p)

        # Realise next alpha
        alpha_val = fixed_corr * alpha_val + np.sqrt(1 - fixed_corr**2) * rng_global.standard_normal()

    return np.array(pos_list), np.array(reward_list)

# -----------------------------------------------------------------------------
# 6. Example *notebook* usage (split into separate cells!)
# -----------------------------------------------------------------------------

"""markdown
```python
# --- Train θ --------------------------------------------------------------
θ = fitted_value_iteration(n_samples=10_000,
                           m_samples=100,
                           num_iterations=300,
                           gamma=0.99)
```

```python
# --- Evaluate greedy policy under fixed parameters -----------------------
pos, rew = eval_fixed_policy(θ,
                               fixed_c=8.0,
                               fixed_corr=0.94,
                               fixed_tl=75.0,
                               num_steps=10_000)  # shorten for a quick test
print("Total PnL:", rew.sum())
```
"""
