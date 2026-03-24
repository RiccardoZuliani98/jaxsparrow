# JaxSPARROW

**Jax - Sensitivity for PARametric Optimization (Wow!)**

I wanted a differentiable QP solver that is fast and works in Jax.
I did not find one (yes QPax works fine, but there are many other solvers running in numpy that I sometimes wants to use bacause more options > fewer options).
So I made this library.

It's similar to [dQP](https://github.com/cwmagoon/dQP) but it's written in Jax instead of pytorch.

This library provides two functions that allow the definition of differentiable QP solvers, one is dense and one is sparse.
The QP problem is solved using the wrapper offered by [qpsolvers](https://github.com/qpsolvers/qpsolvers), but more backends will be available soon.

This library implements efficient conversions between numpy and jax basically.
Also you can take derivatives through the QP solver (both forward and reverse).
They are quite efficient (also using numpy linear algebra routines, batching whenever multiple linear solves are required with the same LHS).

Feel free to contribute or reach out if you find issues ([rzuliani@ethz.ch](mailto:rzuliani@ethz.ch))!

---

## Installation

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
# Reverse mode (VJP) — required for jax.grad and jax.jacobian (jacrev)
solver_rev = setup_dense_solver(..., options={"diff_mode": "rev"})

# Forward mode (JVP) — only accessible via jax.jvp, jacfwd is not supported
solver_fwd = setup_dense_solver(..., options={"diff_mode": "fwd"})
```

- **Reverse mode** (`"rev"`): Use `jax.grad`, `jax.jacobian` (which internally uses `jacrev`), and `jax.vjp`. This is the recommended choice for most optimisation and gradient‑based workflows.
- **Forward mode** (`"fwd"`): Only available through `jax.jvp`. `jax.jacfwd` is **not supported** because of how the custom jvp is implemented. To compute forward‑mode Jacobian‑vector products, simply pass a batched tangent direction to `jax.jvp` – the solver will efficiently solve all directions at once.

Both modes work with `jax.vmap`. The choice affects performance, not correctness.

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
| `diff_mode` | `str` | `"fwd"` | `"fwd"` (JVP) or `"rev"` (VJP) |
| `solver` | `dict` | `{}` | Options for the QP solver backend (e.g. `{"solver_name": "piqp"}`) |
| `differentiator` | `dict` | `{}` | Options for the differentiation backend (e.g. `{"linear_solver": "splu"}`) |
| `dtype` | `dtype` | `float64` | NumPy dtype for all computations |
| `debug` | `bool` | `False` | Enable runtime validation checks |
| `fd_check` | `bool` | `False` | Run finite-difference checks alongside analytical derivatives |
| `fd_eps` | `float` | `1e-6` | Perturbation size for FD checks |

Solver-specific options (e.g. `solver_name`) are passed via the nested `"solver"` dict:

```python
options={"solver": {"solver_name": "osqp"}}
```

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
│   ├── _dense_diff_backend.py   # Dense KKT backend implementation
│   ├── _options.py              # Dense-specific options
│   ├── _types.py                # Dense type aliases
│   └── _converters.py           # JAX array ↔ numpy ndarray
├── _solver_sparse/
│   ├── _setup.py                # setup_sparse_solver entry point
│   ├── _solvers.py              # Sparse numpy QP solver closure (CSC throughout)
│   ├── _differentiators.py      # Sparse KKT forward/reverse differentiators
│   ├── _sparse_diff_backend.py  # Sparse KKT backend implementation
│   ├── _options.py              # Sparse-specific options
│   ├── _types.py                # Sparse type aliases
│   └── _converters.py           # BCOO ↔ CSC, sparsity info extraction
└── _utils/
    ├── _diff_backends.py        # Differentiator backend protocol
    ├── _solver_backends.py      # QP solver backend protocol
    ├── _fd_recorder.py          # Finite‑difference validation
    ├── _linear_solvers.py       # Dense/sparse linear solvers
    ├── _parsing_utils.py        # Option merging
    ├── _printing_utils.py       # Formatting helpers
    ├── _sparse_utils.py         # BCOO → CSC conversion
    └── _timing_utils.py         # TimingRecorder
```

The dense and sparse paths share all JAX-level wiring (`_solver_common.py`) and differ only in how data is converted and how the KKT system is assembled and solved. The common module never inspects the numpy representation — it delegates everything through converter callables supplied by the dense or sparse setup function.

---

## License

MIT