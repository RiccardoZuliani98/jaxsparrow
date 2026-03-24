from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp
import jax
from jax import jit, value_and_grad
from jaxsparrow import qp_value

jax.config.update("jax_enable_x64", True)

horizon = 50

A = jnp.array([[1,1],[0,1]])
B = jnp.array([[0],[1]])

xmax = 5
xmin = -5
umax = 0.5
umin = -0.5

cost_state = 1
cost_input = 0.1

nx,nu = B.shape
N = horizon

nz = (N+1)*nx + N*nu

# cost
P = jnp.diag(
    jnp.hstack((
        jnp.ones((N+1)*nx) * cost_state,
        jnp.ones(N*nu) * cost_input
    ))
)

q = jnp.zeros(nz)

# inequality constraints
G = jnp.vstack((jnp.eye(nz), -jnp.eye(nz)))

h = jnp.hstack((
    jnp.ones((N+1)*nx)*xmax,
    jnp.ones(N*nu)*umax,
    -jnp.ones((N+1)*nx)*xmin,
    -jnp.ones(N*nu)*umin
))

# subdiagonal shift matrix
S = jnp.diag(jnp.ones(N), -1)

# state part
Ax = jnp.kron(jnp.eye(N+1), jnp.eye(nx)) + jnp.kron(S, -A)

# input part
Su = jnp.vstack((jnp.zeros((1, N)), jnp.eye(N)))
Au = jnp.kron(Su, -B)

Aeq = jnp.hstack((Ax, Au))

# parameterized RHS
def beq(x_init):
    return jnp.hstack((
        x_init,
        jnp.zeros(N*nx)
    ))

neq = Aeq.shape[0]
nineq = G.shape[0]

from jaxsparrow import setup_dense_solver

solver = setup_dense_solver(n_var=nz, n_ineq=nineq, n_eq=neq)

@jit
def solve_mpc(x_init):
    return solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)

@jit
def value(x_init, sol):
    return qp_value(P=P, q=q, G=G, h=h, A=Aeq, b=beq(x_init), sol=sol)

@jit
def value_and_dv(x_init, sol):
    return value_and_grad(value, argnums=0)(x_init, sol)


# ============================================================================
# Finite difference verification
# ============================================================================

def finite_diff_check(x0, epsilons=None, seed=42):
    """
    For several random perturbation directions and several step sizes,
    compare:
        (V(x0 + eps*d) - V(x0)) / eps     [forward difference]
        (V(x0 + eps*d) - V(x0 - eps*d)) / (2*eps)  [central difference]
    against:
        grad V(x0) . d                     [envelope theorem]

    We expect the forward difference error to be O(eps) and the central
    difference error to be O(eps^2).

    Returns a list of dicts with per-direction best central-difference errors.
    """
    if epsilons is None:
        epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

    key = jax.random.PRNGKey(seed)

    # solve and get envelope gradient at x0
    sol0 = solve_mpc(x0)
    v0, grad0 = value_and_dv(x0, sol0)

    print(f"x0          = {x0}")
    print(f"V(x0)       = {v0:.10f}")
    print(f"grad V(x0)  = {grad0}")
    print()

    n_directions = 5
    keys = jax.random.split(key, n_directions)

    direction_results = []

    for i_dir in range(n_directions):
        # random unit direction
        d = jax.random.normal(keys[i_dir], shape=x0.shape)
        d = d / jnp.linalg.norm(d)

        # directional derivative from envelope theorem
        dv_analytical = jnp.dot(grad0, d)

        print(f"--- direction {i_dir}: d = {d} ---")
        print(f"  analytical  dV/d(eps) = {dv_analytical:.10f}")
        print(f"  {'eps':>10s}  {'fwd diff':>14s}  {'fwd err':>12s}  "
              f"{'ctr diff':>14s}  {'ctr err':>12s}  {'fwd ratio':>10s}  {'ctr ratio':>10s}")

        prev_fwd_err = None
        prev_ctr_err = None
        best_ctr_err = float("inf")

        for eps in epsilons:
            # forward: solve at x0 + eps*d
            sol_fwd = solve_mpc(x0 + eps * d)
            v_fwd = value(x0 + eps * d, sol_fwd)

            # backward: solve at x0 - eps*d
            sol_bwd = solve_mpc(x0 - eps * d)
            v_bwd = value(x0 - eps * d, sol_bwd)

            dv_fwd = (v_fwd - v0) / eps
            dv_ctr = (v_fwd - v_bwd) / (2 * eps)

            fwd_err = abs(float(dv_fwd - dv_analytical))
            ctr_err = abs(float(dv_ctr - dv_analytical))

            best_ctr_err = min(best_ctr_err, ctr_err)

            # convergence rate: ratio of successive errors
            fwd_ratio_str = ""
            ctr_ratio_str = ""
            if prev_fwd_err is not None and fwd_err > 0:
                fwd_ratio_str = f"{prev_fwd_err / fwd_err:.2f}"
            if prev_ctr_err is not None and ctr_err > 0:
                ctr_ratio_str = f"{prev_ctr_err / ctr_err:.2f}"

            print(f"  {eps:10.1e}  {float(dv_fwd):14.10f}  {fwd_err:12.2e}  "
                  f"{float(dv_ctr):14.10f}  {ctr_err:12.2e}  "
                  f"{fwd_ratio_str:>10s}  {ctr_ratio_str:>10s}")

            prev_fwd_err = fwd_err
            prev_ctr_err = ctr_err

        direction_results.append({
            "x0": x0,
            "direction": i_dir,
            "analytical": float(dv_analytical),
            "best_ctr_err": best_ctr_err,
        })
        print()

    return direction_results


# ============================================================================
# Run checks from several initial conditions
# ============================================================================

all_results = []

print("=" * 100)
print("FINITE DIFFERENCE CHECK: envelope theorem gradient of MPC value function")
print("=" * 100)
print()

# Test 1: interior point (no active state/input bounds)
print("TEST 1: mild initial condition (likely few active constraints)")
print("-" * 100)
all_results.extend(finite_diff_check(jnp.array([-1.0, 0.5])))

# Test 2: larger initial condition (likely activates constraints)
print("TEST 2: larger initial condition (likely activates bounds)")
print("-" * 100)
all_results.extend(finite_diff_check(jnp.array([-3.0, -1.0])))

# Test 3: near a bound
print("TEST 3: near state bound")
print("-" * 100)
all_results.extend(finite_diff_check(jnp.array([4.5, 0.0])))

# Test 4: random initial conditions
print("TEST 4: random initial conditions")
print("-" * 100)
key = jax.random.PRNGKey(123)
for j in range(3):
    key, subkey = jax.random.split(key)
    x0_rand = jax.random.uniform(subkey, shape=(nx,), minval=-1.0, maxval=1.0)
    print(f"  ** random x0 #{j} **")
    all_results.extend(finite_diff_check(x0_rand, seed=j))


# ============================================================================
# Summary
# ============================================================================

print("=" * 100)
print("SUMMARY")
print("=" * 100)
print()

n_total = len(all_results)
tol = 1e-5
n_passed = sum(1 for r in all_results if r["best_ctr_err"] < tol)
worst_err = max(r["best_ctr_err"] for r in all_results)
median_err = float(jnp.median(jnp.array([r["best_ctr_err"] for r in all_results])))

print(f"  Total directional derivatives tested : {n_total}")
print(f"  Tolerance (best central diff error)  : {tol:.0e}")
print(f"  Passed                               : {n_passed}/{n_total}")
print(f"  Median best central-difference error  : {median_err:.2e}")
print(f"  Worst  best central-difference error  : {worst_err:.2e}")
print()

# per-test breakdown
print(f"  {'x0':>30s}  {'dir':>4s}  {'|dV/de|':>12s}  {'best ctr err':>14s}  {'status':>8s}")
print(f"  {'-'*30}  {'-'*4}  {'-'*12}  {'-'*14}  {'-'*8}")
for r in all_results:
    status = "PASS" if r["best_ctr_err"] < tol else "FAIL"
    x0_str = str(r["x0"].tolist())
    print(f"  {x0_str:>30s}  {r['direction']:>4d}  {abs(r['analytical']):12.6f}  "
          f"{r['best_ctr_err']:14.2e}  {status:>8s}")

print()
if n_passed == n_total:
    print("  *** ALL CHECKS PASSED ***")
else:
    print(f"  *** {n_total - n_passed} CHECK(S) FAILED ***")