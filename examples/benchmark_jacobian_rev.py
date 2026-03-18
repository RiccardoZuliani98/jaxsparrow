"""
Benchmark: MPC solve + Jacobian via vmap(vjp) with random initial conditions.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp
import jax
import numpy as np
from jax import jit, vjp, vmap
from time import perf_counter

jax.config.update("jax_enable_x64", True)

# ─── Problem setup ───────────────────────────────────────────────────

horizon = 50
N_SAMPLES = 50

A = jnp.array([[1, 1], [0, 1]])
B = jnp.array([[0], [1]])

xmax, xmin = 5, -5
umax, umin = 0.5, -0.5
cost_state, cost_input = 1.0, 0.1

nx, nu = B.shape
N = horizon
nz = (N + 1) * nx + N * nu

P = jnp.diag(jnp.hstack((
    jnp.ones((N + 1) * nx) * cost_state,
    jnp.ones(N * nu) * cost_input,
)))
q = jnp.zeros(nz)

G = jnp.vstack((jnp.eye(nz), -jnp.eye(nz)))
h = jnp.hstack((
    jnp.ones((N + 1) * nx) * xmax,
    jnp.ones(N * nu) * umax,
    -jnp.ones((N + 1) * nx) * xmin,
    -jnp.ones(N * nu) * umin,
))

S = jnp.diag(jnp.ones(N), -1)
Ax = jnp.kron(jnp.eye(N + 1), jnp.eye(nx)) + jnp.kron(S, -A)
Su = jnp.vstack((jnp.zeros((1, N)), jnp.eye(N)))
Au = jnp.kron(Su, -B)
Aeq = jnp.hstack((Ax, Au))

neq = Aeq.shape[0]
nineq = G.shape[0]

def beq(x_init):
    return jnp.hstack((x_init, jnp.zeros(N * nx)))

# ─── Solver setup ────────────────────────────────────────────────────

from src.solver_dense.solver_dense import setup_dense_solver

# Use reverse-mode differentiator
solver = setup_dense_solver(
    n_var=nz, 
    n_ineq=nineq, 
    n_eq=neq,
    options={"differentiator_type": "kkt_rev"}  # Explicitly use reverse mode
)

def solve_mpc(x_init):
    return solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)

# Jacobian via vmap(vjp) with identity cotangent vectors
def get_jacobian_vjp(x0):
    """Compute Jacobian using reverse-mode autodiff (VJP)."""
    # Forward pass: solve MPC and prepare for VJP
    sol = solve_mpc(x0)
    
    # Define function that returns just x (primal solution)
    # Adjust this based on which output you need Jacobian for
    def get_x(x_init):
        return solve_mpc(x_init)["x"]
    
    # Set up VJP function
    vjp_func = vjp(get_x, x0)[1]
    
    # Compute Jacobian row by row using basis vectors
    # For nx inputs and nz outputs, we need nz VJP calls (one per output dimension)
    n_out = nz  # length of x vector
    jac_rows = []
    
    for i in range(n_out):
        # Create basis cotangent vector for i-th output
        e_i = jnp.zeros(n_out)
        e_i = e_i.at[i].set(1.0)
        
        # Compute gradient (row of Jacobian) via VJP
        grad_i = vjp_func(e_i)[0]
        jac_rows.append(grad_i)
    
    # Stack rows to form Jacobian (n_out, nx)
    return sol, jnp.stack(jac_rows)

def get_jacobian_vjp_batched(x0_batch):
    """Batched version of Jacobian computation using vmap."""
    
    def compute_jac_single(x0):
        # Forward pass
        sol = solve_mpc(x0)
        
        # VJP function
        vjp_func = vjp(lambda x: solve_mpc(x)["x"], x0)[1]
        
        # Compute full Jacobian by VJP with identity matrix
        # More efficient: pass identity matrix as cotangent to get full Jacobian at once
        I = jnp.eye(nz)  # Identity matrix for outputs
        jac = vmap(vjp_func)(I)  # Shape: (nz, nx)
        
        return sol, jac
    
    # vmap over batch dimension
    return vmap(compute_jac_single)(x0_batch)

# Alternative: Most efficient - compute Jacobian in one shot using vjp with identity
def get_jacobian_vjp_efficient(x0):
    """Compute Jacobian efficiently using VJP with identity matrix."""
    def get_x(x_init):
        return solve_mpc(x_init)["x"]
    
    # VJP function
    vjp_func = vjp(get_x, x0)[1]
    
    # Compute full Jacobian by applying VJP to each basis vector
    # This is still nz VJP calls, but now vectorized
    I = jnp.eye(nz)
    jac = vmap(vjp_func)(I)  # Shape: (nz, nx)
    
    return get_x(x0), jac

# Use the batched version
jacobian_fn = jit(vmap(get_jacobian_vjp_efficient))

# ─── Warmup (JIT compilation) ────────────────────────────────────────

x0_warmup = jnp.array([-1.0, -1.0])
x0_batch_warmup = jnp.stack([x0_warmup] * 2)  # Small batch for warmup
_ = jacobian_fn(x0_batch_warmup)

# ─── Benchmark ───────────────────────────────────────────────────────

rng = np.random.default_rng(42)
x0_samples = rng.uniform(-1.0, 1.0, size=(N_SAMPLES, nx))

# solver.timings.reset()

wall_start = perf_counter()

# Process in batches for better efficiency
batch_size = 10
n_batches = N_SAMPLES // batch_size

for i in range(n_batches):
    batch = x0_samples[i*batch_size:(i+1)*batch_size]
    x0_batch = jnp.array(batch)
    sol_batch, jac_batch = jacobian_fn(x0_batch)

# Handle remaining samples
if n_batches * batch_size < N_SAMPLES:
    remaining = x0_samples[n_batches*batch_size:]
    x0_rem = jnp.array(remaining)
    sol_rem, jac_rem = jacobian_fn(x0_rem)

wall_elapsed = perf_counter() - wall_start

# ─── Results ─────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"  MPC Jacobian benchmark (VJP / Reverse Mode)")
print(f"  horizon = {horizon}, n_var = {nz}, n_eq = {neq}, n_ineq = {nineq}")
print(f"  {N_SAMPLES} random initial conditions, Jacobian via vmap(vjp)")
print(f"  Output dimension: {nz}, Input dimension: {nx}")
print(f"{'=' * 60}")
print(f"\n  Wall-clock total : {wall_elapsed:.4f} s")
print(f"  Wall-clock / call: {wall_elapsed / N_SAMPLES * 1e3:.3f} ms\n")
print(solver.timings.summary())

# Optional: Verify correctness against finite differences
print(f"\n{'=' * 60}")
print(f"  Verification (first sample)")
print(f"{'=' * 60}")

def finite_diff_jacobian(x0, eps=1e-6):
    """Compute Jacobian via finite differences for verification."""
    x0 = jnp.array(x0)
    jac = np.zeros((nz, nx))
    x0_np = np.array(x0)
    
    for i in range(nx):
        x_plus = x0_np.copy()
        x_plus[i] += eps
        x_minus = x0_np.copy()
        x_minus[i] -= eps
        
        f_plus = np.array(solve_mpc(jnp.array(x_plus))["x"])
        f_minus = np.array(solve_mpc(jnp.array(x_minus))["x"])
        
        jac[:, i] = (f_plus - f_minus) / (2 * eps)
    
    return jac

# Test with first sample
x0_test = jnp.array(x0_samples[0])
_, jac_vjp = get_jacobian_vjp_efficient(x0_test)
jac_fd = finite_diff_jacobian(x0_test)

diff = jnp.abs(jac_vjp - jac_fd).max()
print(f"  Max difference vs finite differences: {diff:.2e}")
print(f"  Mean difference: {jnp.abs(jac_vjp - jac_fd).mean():.2e}")