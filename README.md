# JaxSPARROW

**Jax - Sensitivity for PARametric Optimization (Wow!)**

JaxSPARROW is a JAX library for solving parametric quadratic programs (QPs) with exact, efficient derivatives. It wraps existing QP solvers (via [qpsolvers](https://github.com/qpsolvers/qpsolvers)) and makes them fully differentiable through JAX's `jvp`, `vjp`, and `vmap` transformations — enabling gradient-based optimization of systems that include QPs as a subroutine, such as model predictive control (MPC).

Both dense and sparse problem formulations are supported. The sparse path keeps matrices in CSC format throughout, including inside the KKT differentiation step, so it scales to large, structured problems without any dense round-trips.

---

## Installation

```bash
pip install jaxsparrow
```

Or from source:

```bash
git clone https://github.com/<your-org>/jaxsparrow.git
cd jaxsparrow
pip install -e .
```

Requires Python 3.10+ and JAX 0.4+.

---

## Quick start

```python
import jax
import jax.numpy as jnp
from jaxsparrow import setup_dense_solver

# min 0.5 x^T P x + q^T x   s.t.  Gx <= h
P = jnp.array([[2.0, 0.5], [0.5, 1.0]])
q = jnp.array([1.0, 1.0])
G = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
h = jnp.array([0.0, 0.0])

solver = setup_dense_solver(n_var=2, n_ineq=2)

def solve(q_val):
    return solver(P=P, q=q_val, G=G, h=h)

sol = solve(q)
print(sol["x"])  # optimal decision variables

# forward-mode sensitivity
_, dsol = jax.jvp(solve, (q,), (jnp.ones(2),))
print(dsol["x"])  # dx/dq @ ones

# reverse-mode gradient
def objective(q_val):
    return jnp.sum(solve(q_val)["x"] ** 2)

grad = jax.grad(objective)(q)
print(grad)
```

---

## Sparse mode

For large structured problems (e.g. MPC), the sparse path avoids dense matrix operations entirely:

```python
from jax.experimental.sparse import BCOO
from scipy.sparse import csc_matrix
from jaxsparrow import setup_sparse_solver

# Define sparsity patterns (only structure matters, values are ignored)
P_pattern = BCOO.fromdense(P_dense)
A_pattern = BCOO.fromdense(A_dense)
G_pattern = BCOO.fromdense(G_dense)

solver = setup_sparse_solver(
    n_var=n, n_eq=m_eq, n_ineq=m_ineq,
    sparsity_patterns={"P": P_pattern, "A": A_pattern, "G": G_pattern},
    # Fix constant ingredients to avoid repeated conversions:
    fixed_elements={
        "P": csc_matrix(P_dense),
        "G": csc_matrix(G_dense),
        "h": h_array,
    },
)

# At solve time, pass BCOO for dynamic matrices, jax.Array for vectors
result = solver(A=A_bcoo, b=b_vec)
```

---

## Fixed elements

Any QP ingredient that stays constant across solves can be provided at setup time via `fixed_elements`. This has two benefits: the ingredient is excluded from JAX's traced path (no unnecessary differentiation), and it avoids repeated JAX-to-NumPy conversion on every call.

```python
solver = setup_dense_solver(
    n_var=n, n_ineq=m,
    fixed_elements={"P": P_np, "q": q_np, "G": G_np, "h": h_np},
)

# Only b is dynamic — only b flows through the traced path
def solve(b_val):
    return solver(b=b_val)
```

---

## Differentiation modes

JaxSPARROW supports two differentiation strategies, selected at setup:

```python
# Forward mode (JVP) — efficient when n_inputs < n_outputs
solver_fwd = setup_dense_solver(..., options={"differentiator_type": "kkt_fwd"})

# Reverse mode (VJP) — efficient when n_outputs < n_inputs
solver_rev = setup_dense_solver(..., options={"differentiator_type": "kkt_rev"})
```

Both modes work with `jax.jvp`, `jax.vjp`, `jax.grad`, `jax.jacfwd`, `jax.jacrev`, and `jax.vmap`. The choice affects performance, not correctness.

For a scalar loss over a closed-loop MPC simulation, reverse mode computes the full gradient in a single backward pass, while forward mode requires one pass per input dimension.

---

## Batched differentiation

`jax.vmap` works out of the box. When vmapped, the KKT system is factorized once and solved for all tangent/cotangent directions simultaneously:

```python
def jvp_one(dq):
    _, tangents = jax.jvp(solve, (q,), (dq,))
    return tangents["x"]

# Compute dx/dq for 10 directions at once
directions = jax.random.normal(key, (10, n_var))
batched_dx = jax.vmap(jvp_one)(directions)
```

---

## Diagnostics

The solver exposes timing and finite-difference validation:

```python
# Timing
print(solver.timings.summary())
solver.timings.reset()

# Finite-difference checks (enabled at setup)
solver = setup_dense_solver(
    ...,
    options={"fd_check": True, "fd_eps": 1e-6},
)
```

---

## API reference

### `setup_dense_solver`

```python
setup_dense_solver(
    n_var: int,
    n_ineq: int = 0,
    n_eq: int = 0,
    fixed_elements: dict | None = None,
    options: dict | None = None,
) -> solver
```

Build a differentiable dense QP solver. Returns a callable `solver(*, warmstart=None, **kwargs) -> QPOutput` where `QPOutput` is a dict with keys `"x"`, `"lam"`, `"mu"`.

### `setup_sparse_solver`

```python
setup_sparse_solver(
    n_var: int,
    n_ineq: int = 0,
    n_eq: int = 0,
    sparsity_patterns: dict[str, BCOO] | None = None,
    fixed_elements: dict | None = None,
    options: dict | None = None,
) -> solver
```

Build a differentiable sparse QP solver. Matrices (P, A, G) are passed as `BCOO` at call time. Sparsity patterns must be provided for every dynamic matrix key.

### Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `differentiator_type` | `str` | `"kkt_fwd"` | `"kkt_fwd"` (JVP) or `"kkt_rev"` (VJP) |
| `solver_type` | `str` | `"qp_solvers"` | Backend solver family |
| `dtype` | `dtype` | `float64` | Numpy dtype for all computations |
| `debug` | `bool` | `False` | Enable runtime validation checks |
| `fd_check` | `bool` | `False` | Run finite-difference checks alongside analytical derivatives |
| `fd_eps` | `float` | `1e-6` | Perturbation size for FD checks |

Solver-specific options (e.g. `solver_name`) are passed via the nested `"solver"` dict:

```python
options={"solver": {"solver_name": "osqp"}}
```

---

## How it works

JaxSPARROW differentiates through QP solves using the KKT optimality conditions. At a high level:

1. **Forward solve**: the QP is solved by an external solver (OSQP, CLARABEL, etc.) via NumPy/SciPy, called from JAX through `pure_callback`.

2. **Differentiation**: given the optimal solution (x*, λ*, μ*), the KKT conditions define an implicit function. Differentiating this system yields a linear equation whose solution gives the sensitivity of the optimal variables with respect to any problem parameter.

3. **JAX integration**: `custom_jvp` or `custom_vjp` registers the differentiation rule so that standard JAX transformations (`jvp`, `vjp`, `grad`, `vmap`) work transparently.

4. **Batching**: `vmap_method="expand_dims"` in `pure_callback` enables batched differentiation. The KKT matrix is factorized once; each batch element is a separate RHS column.

---

## Project structure

```
jaxsparrow/
├── __init__.py                  # Public API: setup_dense_solver, setup_sparse_solver
├── _solver_common.py            # Shared: custom_jvp/vjp wiring, pure_callback, batching
├── _options_common.py           # Default options and TypedDicts
├── _types_common.py             # QPOutput, QPDiffOut type aliases
├── _solver_dense/
│   ├── _setup.py                # setup_dense_solver entry point
│   ├── _solvers.py              # Dense numpy QP solver closure
│   ├── _differentiators.py      # Dense KKT forward/reverse differentiators
│   └── _converters.py           # JAX array ↔ numpy ndarray
├── _solver_sparse/
│   ├── _setup.py                # setup_sparse_solver entry point
│   ├── _solvers.py              # Sparse numpy QP solver closure (CSC throughout)
│   ├── _differentiators.py      # Sparse KKT forward/reverse differentiators
│   ├── _converters.py           # BCOO ↔ CSC, sparsity info extraction
│   └── _types.py                # Sparse-specific type aliases
└── _utils/
    ├── _parsing.py              # Option merging with defaults
    ├── _timing.py               # TimingRecorder
    └── _fd_recorder.py          # Finite-difference validation
```

The dense and sparse paths share all JAX-level wiring (`_solver_common.py`) and differ only in how data is converted and how the KKT system is assembled and solved. The common module never inspects the numpy representation — it delegates everything through converter callables supplied by the dense or sparse setup function.

---

## License

MIT