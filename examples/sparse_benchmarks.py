"""
sparse_benchmarks.py
====================
Unified MPC benchmark: solve + differentiate with configurable mode.

Experiments
-----------
  diff_mode  :  "fwd"  — forward-mode (JVP)
                "rev"  — reverse-mode (VJP)
  batched    :  False  — one tangent/cotangent at a time (loop)
                True   — vmap over all tangents/cotangents at once
  diff_scope :  "small" — few differentiation directions
                  fwd: tangents w.r.t. initial condition x0 only (nx dirs)
                  rev: cotangent on first state component only (1 dir)
                "large" — many differentiation directions
                  fwd: tangents w.r.t. all QP parameters (P,q,A,b,G,h)
                  rev: cotangent on full solution vector (nz dirs)

Profiling
---------
  Set PROFILE = True to capture Perfetto traces of:
    1. A single forward solve (no differentiation)
    2. A single differentiation call (fwd or rev, batched or sequential)
  Traces are saved to PROFILE_DIR. Open at https://ui.perfetto.dev/

Set the knobs in the "USER CONFIG" block below, then run:

    python sparse_benchmarks.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, jvp, vjp, vmap
from jax.experimental.sparse import BCOO
from time import perf_counter

jax.config.update("jax_enable_x64", True)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                        USER CONFIG                               ║
# ╚══════════════════════════════════════════════════════════════════╝

DIFF_MODE   = "rev"       # "fwd" | "rev"
BATCHED     = True         # True  → vmap over all tangent dirs at once
DIFF_SCOPE  = "large"      # "small" | "large"  (see docstring above)

HORIZON     = 350          # MPC prediction horizon
N_SAMPLES   = 50           # number of random initial conditions to benchmark

PROFILE     = True        # True → capture Perfetto traces
PROFILE_DIR = "./.perfetto_trace"   # directory for trace output

# ╔══════════════════════════════════════════════════════════════════╗
# ║                     MPC PROBLEM SETUP                            ║
# ╚══════════════════════════════════════════════════════════════════╝

A_dyn = jnp.array([[1, 1], [0, 1]])
B_dyn = jnp.array([[0], [1]])

xmax, xmin = 5.0, -5.0
umax, umin = 0.5, -0.5
cost_state, cost_input = 1.0, 0.1

nx, nu = B_dyn.shape
N = HORIZON
nz = (N + 1) * nx + N * nu

# ── Cost (diagonal → sparse) ────────────────────────────────────────
P_dense = jnp.diag(jnp.hstack((
    jnp.ones((N + 1) * nx) * cost_state,
    jnp.ones(N * nu) * cost_input,
)))
P = BCOO.fromdense(P_dense)
q = jnp.zeros(nz)

# ── Inequality constraints (box → sparse) ───────────────────────────
G_dense = jnp.vstack((jnp.eye(nz), -jnp.eye(nz)))
G = BCOO.fromdense(G_dense)

h = jnp.hstack((
    jnp.ones((N + 1) * nx) * xmax,
    jnp.ones(N * nu) * umax,
    -jnp.ones((N + 1) * nx) * xmin,
    -jnp.ones(N * nu) * umin,
))

# ── Equality constraints (dynamics → sparse) ────────────────────────
S = jnp.diag(jnp.ones(N), -1)
Ax = jnp.kron(jnp.eye(N + 1), jnp.eye(nx)) + jnp.kron(S, -A_dyn)
Su = jnp.vstack((jnp.zeros((1, N)), jnp.eye(N)))
Au = jnp.kron(Su, -B_dyn)
Aeq_dense = jnp.hstack((Ax, Au))
Aeq = BCOO.fromdense(Aeq_dense)

neq = Aeq_dense.shape[0]
nineq = G_dense.shape[0]

_b_template = jnp.zeros(neq)

def beq(x_init):
    return _b_template.at[:nx].set(x_init)

# ── Sparsity report ─────────────────────────────────────────────────
print(f"  Sparsity: P has {P.nse}/{P_dense.size} nonzeros "
      f"({P.nse / P_dense.size * 100:.1f}%)")
print(f"  Sparsity: A has {Aeq.nse}/{Aeq_dense.size} nonzeros "
      f"({Aeq.nse / Aeq_dense.size * 100:.1f}%)")
print(f"  Sparsity: G has {G.nse}/{G_dense.size} nonzeros "
      f"({G.nse / G_dense.size * 100:.1f}%)")

# ╔══════════════════════════════════════════════════════════════════╗
# ║                      SOLVER SETUP                                ║
# ╚══════════════════════════════════════════════════════════════════╝

from jaxsparrow import setup_sparse_solver

sparsity_patterns = {"P": P, "A": Aeq, "G": G}

# ── Determine which parameters are dynamic based on diff_scope ──────

if DIFF_SCOPE == "small":
    fixed_elements = {"P": P, "q": q, "A": Aeq, "G": G, "h": h}
    dynamic_keys = ("b",)
    scope_desc = "x0 only → b dynamic"
elif DIFF_SCOPE == "large":
    fixed_elements = {}
    dynamic_keys = ("P", "q", "A", "b", "G", "h")
    scope_desc = "all QP parameters dynamic"
else:
    raise ValueError(
        f"Unknown DIFF_SCOPE: {DIFF_SCOPE!r}. Use 'small' or 'large'."
    )

solver = setup_sparse_solver(
    n_var=nz,
    n_ineq=nineq,
    n_eq=neq,
    sparsity_patterns=sparsity_patterns,
    fixed_elements=fixed_elements if fixed_elements else None,
    options={"diff_mode": "rev" if DIFF_MODE == "rev" else "fwd"},
)

# ╔══════════════════════════════════════════════════════════════════╗
# ║            DIFFERENTIABLE SOLVE WRAPPERS                         ║
# ╚══════════════════════════════════════════════════════════════════╝

if DIFF_SCOPE == "small":
    @jit
    def solve_mpc(x_init):
        return solver(b=beq(x_init))

    @jit
    def get_x(x_init):
        return solve_mpc(x_init)["x"]

    n_tangent_input = nx

elif DIFF_SCOPE == "large":
    @jit
    def solve_mpc(x_init):
        return solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)

    @jit
    def get_x(x_init):
        return solve_mpc(x_init)["x"]

    n_tangent_input = nx

# ╔══════════════════════════════════════════════════════════════════╗
# ║              BUILD TANGENT / COTANGENT DIRECTIONS                ║
# ╚══════════════════════════════════════════════════════════════════╝

rng = np.random.default_rng(0)

if DIFF_MODE == "fwd":
    if DIFF_SCOPE == "small":
        n_diff = nx
        tangent_dirs = jnp.eye(nx)
    else:
        n_diff = nx
        tangent_dirs = jnp.eye(nx)

elif DIFF_MODE == "rev":
    if DIFF_SCOPE == "small":
        n_diff = nx
        cotangent_dirs = jnp.eye(nz, nx)
        cotangent_dirs = cotangent_dirs.T
    else:
        n_diff = nz
        cotangent_dirs = jnp.eye(nz)

else:
    raise ValueError(f"Unknown diff_mode: {DIFF_MODE!r}. Use 'fwd' or 'rev'.")

# ╔══════════════════════════════════════════════════════════════════╗
# ║               DIFFERENTIATION FUNCTIONS                          ║
# ╚══════════════════════════════════════════════════════════════════╝

if DIFF_MODE == "fwd":
    @jit
    def jvp_single(x0, dx0):
        return jvp(get_x, (x0,), (dx0,))

    if BATCHED:
        jvp_batched = jit(vmap(jvp_single, in_axes=(None, 0)))

        def diff_fn(x0):
            primals, tangents = jvp_batched(x0, tangent_dirs)
            return primals[0], tangents

elif DIFF_MODE == "rev":
    @jit
    def vjp_setup(x0):
        return vjp(get_x, x0)

    @jit
    def vjp_single(vjp_func, v):
        return vjp_func(v)[0]

    if BATCHED:
        @jit
        def diff_fn(x0):
            sol, vjp_func = vjp(get_x, x0)
            jac = vmap(vjp_func)(cotangent_dirs)
            return sol, jac[0]

# ╔══════════════════════════════════════════════════════════════════╗
# ║                    WARMUP (JIT COMPILE)                          ║
# ╚══════════════════════════════════════════════════════════════════╝

print(f"\n  Warming up (JIT compilation) ...")
x0_warmup = jnp.array([-1.0, -1.0])

t0 = perf_counter()
if BATCHED:
    _ = diff_fn(x0_warmup)
elif DIFF_MODE == "fwd":
    _ = jvp_single(x0_warmup, tangent_dirs[0])
    _ = jvp_single(x0_warmup, tangent_dirs[0])
elif DIFF_MODE == "rev":
    _sol, _vjp_fn = vjp_setup(x0_warmup)
    _sol, _vjp_fn = vjp_setup(x0_warmup)
    _ = vjp_single(_vjp_fn, cotangent_dirs[0])
    _ = vjp_single(_vjp_fn, cotangent_dirs[0])
jit_time = perf_counter() - t0
print(f"  JIT compile time: {jit_time:.3f} s")

# Also warmup the plain solve (no diff) for profiling
_ = solve_mpc(x0_warmup)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                   PROFILING (optional)                           ║
# ╚══════════════════════════════════════════════════════════════════╝

if PROFILE:
    from pathlib import Path as _P
    _profile_root = _P(PROFILE_DIR).resolve()
    _profile_root.mkdir(parents=True, exist_ok=True)

    x0_prof = jnp.array([-1.0, -1.0])

    # ── Profile a single solve (no differentiation) ──────────────────
    solve_dir = str(_profile_root / "solve")
    print(f"\n  Profiling single solve → {solve_dir}/")
    with jax.profiler.trace(solve_dir, create_perfetto_link=True):
        result = solve_mpc(x0_prof)
        result["x"].block_until_ready()

    # ── Profile a single differentiation call ────────────────────────
    diff_label = f"diff_{DIFF_MODE}"
    if BATCHED:
        diff_label += "_batched"
    diff_dir = str(_profile_root / diff_label)
    print(f"  Profiling single {DIFF_MODE} diff → {diff_dir}/")

    with jax.profiler.trace(diff_dir, create_perfetto_link=True):
        if BATCHED:
            sol, jac = diff_fn(x0_prof)
            sol.block_until_ready()
        elif DIFF_MODE == "fwd":
            sol, t = jvp_single(x0_prof, tangent_dirs[0])
            sol.block_until_ready()
        elif DIFF_MODE == "rev":
            sol, vjp_func = vjp_setup(x0_prof)
            sol.block_until_ready()
            g = vjp_single(vjp_func, cotangent_dirs[0])
            g.block_until_ready()


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        BENCHMARK                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

rng_bench = np.random.default_rng(42)
x0_samples = rng_bench.uniform(-1.0, 1.0, size=(N_SAMPLES, nx))

solver.timings.reset()

wall_times = []

if BATCHED:
    for i in range(N_SAMPLES):
        x0_i = jnp.array(x0_samples[i])
        start = perf_counter()
        sol, jac = diff_fn(x0_i)
        sol.block_until_ready()
        wall_times.append(perf_counter() - start)

elif DIFF_MODE == "fwd":
    for i in range(N_SAMPLES):
        x0_i = jnp.array(x0_samples[i])
        for j in range(n_diff):
            start = perf_counter()
            sol, t = jvp_single(x0_i, tangent_dirs[j])
            sol.block_until_ready()
            wall_times.append(perf_counter() - start)

elif DIFF_MODE == "rev":
    for i in range(N_SAMPLES):
        x0_i = jnp.array(x0_samples[i])
        sol, vjp_func = vjp_setup(x0_i)
        sol.block_until_ready()
        for j in range(n_diff):
            start = perf_counter()
            g = vjp_single(vjp_func, cotangent_dirs[j])
            g.block_until_ready()
            wall_times.append(perf_counter() - start)

wall_times = np.array(wall_times)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                         RESULTS                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

mode_label = "JVP (forward)" if DIFF_MODE == "fwd" else "VJP (reverse)"
batch_label = "batched (vmap)" if BATCHED else "sequential (loop)"
n_calls = len(wall_times)

print(f"\n{'=' * 65}")
print(f"  MPC Differentiation Benchmark (SPARSE)")
print(f"{'=' * 65}")
print(f"  diff_mode             : {DIFF_MODE}  ({mode_label})")
print(f"  diff_scope            : {DIFF_SCOPE}  ({scope_desc})")
print(f"  batched               : {BATCHED}  ({batch_label})")
print(f"  n_diff                : {n_diff}")
print(f"  horizon               : {HORIZON}")
print(f"  n_var / n_eq / n_ineq : {nz} / {neq} / {nineq}")
print(f"  N_SAMPLES             : {N_SAMPLES}")
print(f"  total calls timed     : {n_calls}")
if PROFILE:
    print(f"  profiling             : ON → {PROFILE_DIR}/")
print(f"{'=' * 65}")
print(f"\n  JIT compile time  : {jit_time:.4f} s")
print(f"  Wall-clock total  : {wall_times.sum():.4f} s")
print(f"  Wall-clock / call : {wall_times.mean() * 1e3:.3f} ms  (mean)")
print(f"                      {wall_times.std() * 1e3:.3f} ms  (std)")
print(f"                      {wall_times.min() * 1e3:.3f} ms  (min)")
print(f"                      {wall_times.max() * 1e3:.3f} ms  (max)")
print(f"                      {np.median(wall_times) * 1e3:.3f} ms  (median)")
print()
print(solver.timings.summary())