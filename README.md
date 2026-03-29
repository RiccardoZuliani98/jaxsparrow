# JaxSPARROW

**Jax - Sensitivity for PARametric Optimization (Wow!)**

I wanted a differentiable QP solver that is fast and works in Jax. The issue is that most solvers don't run natively in Jax, so I made this library. It's similar to [dQP](https://github.com/cwmagoon/dQP) but it's written in Jax instead of pytorch.

This library provides two functions that allow the definition of differentiable QP solvers, one is dense and one is sparse.
The QP problem is solved using the wrapper offered by [qpsolvers](https://github.com/qpsolvers/qpsolvers), but more backends will be available soon.
Everything is done on CPU because that's where the numpy solvers operate. If you are looking for GPU options, consider [QPax](https://github.com/kevin-tracy/qpax).

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

# min 0.5 x^T P x + q^T x   s.t.  Gx <= h, Ax = b
P = jnp.array([[2.0, 0.5], [0.5, 1.0]])
q = jnp.array([1.0, 1.0])
G = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
h = jnp.array([0.0, 0.0])

solver = setup_dense_solver(n_var=2, n_ineq=2)

def solve(q_val):
    return solver(P=P, q=q_val, G=G, h=h)

sol = solve(q)
print(sol["x"])  # optimal decision variables
```
Take a look at the `examples` folder for several optimal control examples.

---

## Sparse mode

Sparse mode is generally faster for moderately-sized problems (you should choose a solver that supports sparse mode!):

```python
from jax.experimental.sparse import BCOO
from jaxsparrow import setup_sparse_solver

# Define sparsity patterns (only structure matters, values are ignored)
P_sparse = BCOO.fromdense(P_dense)
A_sparse = BCOO.fromdense(A_dense)
G_sparse = BCOO.fromdense(G_dense)

solver = setup_sparse_solver(
    n_var=n, n_eq=m_eq, n_ineq=m_ineq,
    # Pass sparsity of non-constant elements
    sparsity_pattern={
        "A": A_sparse
    },
    # Pass constant elements here
    fixed_elements={
        "P": P_dense,
        "G": G_dense,
        "h": h_array,
    },
)

# At solve time, pass BCOO for dynamic matrices, jax.Array for vectors
result = solver(A=A_bcoo, b=b_vec)
```

---

## Fixed elements

Any QP ingredient that stays constant across solves can be provided at setup time via `fixed_elements`. This has two benefits: the ingredient is excluded from Jax's traced path (no unnecessary differentiation), and it avoids repeated Jax-to-Numpy conversion on every call.

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

`jax.vmap` batches over tangent/cotangent directions. The KKT system is factorized once and solved for all directions simultaneously:

```python
# Forward: full Jacobian via vmapped JVP
I_nx = jnp.eye(nx)

def jvp_single(x0, dx0):
    return jax.jvp(solve_mpc, (x0,), (dx0,))

jac_fn = jax.jit(jax.vmap(jvp_single, in_axes=(None, 0)))
sol, dsol = jac_fn(x0, I_nx)

# Reverse: full Jacobian via vmapped VJP
def get_x(x_init):
    return solver(P=P, q=q, A=Aeq, b=beq(x_init), G=G, h=h)["x"]

sol, vjp_func = jax.vjp(get_x, x0)
jac = jax.vmap(vjp_func)(jnp.eye(nz))
```

**Note:** batching works for derivatives only, not for primal solves.

---

## Envelope theorem

`qp_value` computes the optimal QP objective value as a function of the problem parameters. Combined with `jax.grad`, this gives the sensitivity of the optimal value via the envelope theorem — no need to differentiate through the solver:

```python
from jaxsparrow import qp_value

sol = solve_mpc(x0)
v, dv = jax.value_and_grad(
    lambda x: qp_value(P=P, q=q, G=G, h=h, A=Aeq, b=beq(x), sol=sol)
)(x0)
```

---

## Finite-difference validation

Enable automatic FD checks to validate every derivative call:

```python
solver = setup_dense_solver(
    ...,
    options={"fd_check": True, "fd_eps": 1e-6},
)

# After running some solves with derivatives:
print(solver.fd_check.summary())
```

---

## Diagnostics

```python
# Timing breakdown
print(solver.timings.summary())
solver.timings.reset()
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

Top-level options are passed via the `options` dict:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `diff_mode` | `str` | `"fwd"` | `"fwd"` (JVP) or `"rev"` (VJP) |
| `solver` | `dict` | `{}` | Nested dict passed to the QP solver backend (see below) |
| `differentiator` | `dict` | `{}` | Nested dict passed to the differentiation backend (see below) |
| `dtype` | `jnp.dtype` | `jnp.float64` | JAX dtype for all computations |
| `debug` | `bool` | `True` | Enable runtime validation checks |
| `fd_check` | `bool` | `False` | Run finite-difference checks alongside analytical derivatives |
| `fd_eps` | `float` | `1e-6` | Perturbation size for FD checks |
| `verbose` | `bool` | `True` | Print solver and differentiator progress |

**Solver options** (`options["solver"]`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `solver_name` | `str` | `"piqp"` | QP solver name passed to `qpsolvers` |
| `dtype` | `np.dtype` | `np.float64` | NumPy dtype for solver arrays |
| `backend` | `str` | `"qpsolvers"` | Solver backend protocol |

**Differentiator options** (`options["differentiator"]`):

| Key | Type | Default (dense) | Default (sparse) | Description |
|-----|------|-----------------|-------------------|-------------|
| `backend` | `str` | `"dense_kkt"` | `"sparse_kkt"` | Differentiator backend |
| `linear_solver` | `str` | `"solve"` | `"splu"` | Linear solver for KKT systems |
| `cst_tol` | `float` | `1e-8` | `1e-8` | Tolerance for active constraint detection |
| `dtype` | `np.dtype` | `np.float64` | `np.float64` | NumPy dtype for differentiator |

Available `linear_solver` values: `"solve"`, `"lu"`, `"lstsq"` (dense), `"splu"`, `"spilu"`, `"spsolve"`, `"sp_lstsq"` (sparse).

---

## Project structure

```
jaxsparrow/
├── __init__.py                  # Public API: setup_dense_solver, setup_sparse_solver, qp_value
├── _solver_common.py            # Shared: custom_jvp/vjp wiring, pure_callback, batching, warmstart
├── _options_common.py           # ConstructorOptions, defaults
├── _types_common.py             # SolverOutput, SolverOutputNP, diff output types, Solver protocol
├── _envelope.py                 # qp_value (envelope theorem)
├── _solver_dense/
│   ├── _setup.py                # setup_dense_solver entry point
│   ├── _solvers.py              # Dense NumPy QP solver closure
│   ├── _differentiators.py      # Dense KKT forward/reverse differentiator factories
│   ├── _dense_diff_backend.py   # Dense KKT backend implementation
│   ├── _options.py              # Dense-specific options and defaults
│   ├── _types.py                # DenseIngredients, DenseIngredientsNP, tangent types
│   └── _converters.py           # JAX array ↔ NumPy ndarray
├── _solver_sparse/
│   ├── _setup.py                # setup_sparse_solver entry point
│   ├── _solvers.py              # Sparse NumPy QP solver closure (CSC throughout)
│   ├── _differentiators.py      # Sparse KKT forward/reverse differentiator factories
│   ├── _sparse_diff_backend.py  # Sparse KKT backend implementation
│   ├── _options.py              # Sparse-specific options and defaults
│   ├── _types.py                # SparseIngredients, SparseIngredientsNP, tangent types
│   └── _converters.py           # BCOO ↔ CSC, sparsity info extraction
└── _utils/
    ├── _diff_backends.py        # DifferentiatorBackend protocol and registry
    ├── _solver_backends.py      # SolverBackend protocol and registry
    ├── _fd_recorder.py          # Finite-difference validation recorder
    ├── _linear_solvers.py       # Dense/sparse linear solver dispatch
    ├── _parsing_utils.py        # Option merging (parse_options)
    ├── _printing_utils.py       # Formatting helpers
    ├── _sparse_utils.py         # BCOO → CSC conversion utilities
    └── _timing_utils.py         # TimingRecorder
```

The dense and sparse paths share all JAX-level wiring (`_solver_common.py`) and differ only in how data is converted and how the KKT system is assembled and solved. The common module never inspects the NumPy representation — it delegates everything through converter callables supplied by the dense or sparse setup function.

---

## License

Apache 2.0