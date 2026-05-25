"""
test_mpc_sparse_regularized.py
==============================
Single-shot MPC solve + JVP correctness check — sparse mode 
using Differentiable-by-design Tikhonov regularization.
"""

from time import perf_counter

import numpy as np
import scipy.sparse as sp

import jax
import jax.numpy as jnp
from jax import jit, jvp
from jax.experimental.sparse import BCOO

from jaxsparrow import setup_sparse_solver

jax.config.update("jax_enable_x64", True)

# ── 1. Regularized QP Solver Class ──────────────────────────────────

class RegularizedQPSolver:
    """
    Solves a regularized QP iteratively using Iterative Regularization.
    Optionally accepts fixed_elements (like P and q) to optimize JIT compilation.
    """
    
    def __init__(self, n_x, n_in, n_eq, num_steps=1, sparsity_patterns=None, fixed_elements=None, rho=1e-4, options=None):
        
        self.n_x = n_x
        self.n_in = n_in
        self.n_eq = n_eq
        self.n_z = n_x + n_in + n_eq
        self.num_steps = num_steps
        
        # Handle rho as a sequence or a scalar
        if isinstance(rho, (list, tuple, np.ndarray)):
            self.rho = jnp.array(rho)
            if len(self.rho) != self.num_steps:
                raise ValueError(f"Length of rho list ({len(self.rho)}) must match num_steps ({self.num_steps}).")
        else:
            self.rho = jnp.full(self.num_steps, rho)
            
        # Use the first rho value to instantiate any fixed augmented matrices
        rho_init = self.rho[0]
        
        self.fixed_elements = fixed_elements or {}
        sparsity_patterns = sparsity_patterns or {}
        
        # Helper to ensure fixed matrices are scipy sparse CSC format
        def to_csc(mat):
            if not sp.issparse(mat):
                return sp.csc_matrix(mat)
            return mat.tocsc()
            
        I_nx = sp.eye(n_x, format='csc')
        I_nin = sp.eye(n_in, format='csc')
        I_neq = sp.eye(n_eq, format='csc')
        
        aug_fixed_elements = {}
        aug_patterns = {}
        
        # Precompute index helpers for dynamic matrices
        diag_nx = jnp.arange(n_x)
        diag_nin = jnp.arange(n_in)
        diag_neq = jnp.arange(n_eq)
        I_nx_idx = jnp.stack([diag_nx, diag_nx], axis=-1)
        I_nin_idx = jnp.stack([n_x + diag_nin, n_x + diag_nin], axis=-1)
        I_neq_idx = jnp.stack([n_x + n_in + diag_neq, n_x + n_in + diag_neq], axis=-1)
        
        # --- Handle P ---
        if "P" in self.fixed_elements:
            P_fixed = to_csc(self.fixed_elements["P"])
            aug_fixed_elements["P"] = sp.block_diag(
                [P_fixed + rho_init * I_nx, rho_init * I_nin, rho_init * I_neq], format='csc'
            )
        else:
            P_pat = sparsity_patterns["P"]
            self.P_tilde_indices = jnp.vstack([P_pat.indices, I_nx_idx, I_nin_idx, I_neq_idx])
            self.P_tilde_shape = (self.n_z, self.n_z)
            aug_patterns["P"] = BCOO((jnp.ones(len(self.P_tilde_indices)), self.P_tilde_indices), shape=self.P_tilde_shape)

        # --- Handle G ---
        if "G" in self.fixed_elements:
            G_fixed = to_csc(self.fixed_elements["G"])
            aug_fixed_elements["G"] = sp.bmat([[G_fixed, rho_init * I_nin, None]], format='csc')
        else:
            G_pat = sparsity_patterns["G"]
            G_rho_idx = jnp.stack([diag_nin, n_x + diag_nin], axis=-1)
            self.G_tilde_indices = jnp.vstack([G_pat.indices, G_rho_idx])
            self.G_tilde_shape = (n_in, self.n_z)
            aug_patterns["G"] = BCOO((jnp.ones(len(self.G_tilde_indices)), self.G_tilde_indices), shape=self.G_tilde_shape)

        # --- Handle A ---
        if "A" in self.fixed_elements:
            A_fixed = to_csc(self.fixed_elements["A"])
            aug_fixed_elements["A"] = sp.bmat([[A_fixed, None, rho_init * I_neq]], format='csc')
        else:
            A_pat = sparsity_patterns["A"]
            A_rho_idx = jnp.stack([diag_neq, n_x + n_in + diag_neq], axis=-1)
            self.A_tilde_indices = jnp.vstack([A_pat.indices, A_rho_idx])
            self.A_tilde_shape = (n_eq, self.n_z)
            aug_patterns["A"] = BCOO((jnp.ones(len(self.A_tilde_indices)), self.A_tilde_indices), shape=self.A_tilde_shape)

        # --- Handle Vectors ---
        self.q_fixed = jnp.array(self.fixed_elements["q"]) if "q" in self.fixed_elements else None
        self.h_fixed = jnp.array(self.fixed_elements["h"]) if "h" in self.fixed_elements else None
        self.b_fixed = jnp.array(self.fixed_elements["b"]) if "b" in self.fixed_elements else None
        
        # Initialize Backend Solver
        self.solver = setup_sparse_solver(
            n_var=self.n_z, n_ineq=n_in, n_eq=n_eq,
            sparsity_patterns=aug_patterns if aug_patterns else None,
            fixed_elements=aug_fixed_elements if aug_fixed_elements else None,
            options=options or {"solver": {"backend": "piqp"}}
        )
        
        # num_steps is no longer required as a dynamic argument
        self.solve = jax.jit(self._solve_impl)

    def _solve_impl(self, bar_x, bar_lam, bar_mu, P=None, q=None, A=None, b=None, G=None, h=None):
        
        # 1. Resolve Vectors
        q_val = self.q_fixed if q is None else q
        h_val = self.h_fixed if h is None else h
        b_val = self.b_fixed if b is None else b
        
        solver_kwargs = {}
        
        # Pre-allocate zero arrays for the appended solver queries
        zero_lam = jnp.zeros_like(bar_lam)
        zero_mu = jnp.zeros_like(bar_mu)
        
        # 2. Iterative Solves (Proximal Point Steps)
        # JAX gracefully unrolls standard python loops when bounds are static
        for i in range(self.num_steps):
            rho_i = self.rho[i]
            
            # Build Dynamic Augmented Matrices utilizing current step's rho
            if "P" not in self.fixed_elements:
                P_tilde_data = jnp.concatenate([P.data, jnp.full(self.n_x, rho_i), jnp.full(self.n_in, rho_i), jnp.full(self.n_eq, rho_i)])
                solver_kwargs["P"] = BCOO((P_tilde_data, self.P_tilde_indices), shape=self.P_tilde_shape)
                
            if "G" not in self.fixed_elements:
                G_tilde_data = jnp.concatenate([G.data, jnp.full(self.n_in, rho_i)])
                solver_kwargs["G"] = BCOO((G_tilde_data, self.G_tilde_indices), shape=self.G_tilde_shape)
                
            if "A" not in self.fixed_elements:
                A_tilde_data = jnp.concatenate([A.data, jnp.full(self.n_eq, rho_i)])
                solver_kwargs["A"] = BCOO((A_tilde_data, self.A_tilde_indices), shape=self.A_tilde_shape)
                
            solver_kwargs["q"] = jnp.concatenate([q_val - rho_i * bar_x, zero_lam, zero_mu])
            solver_kwargs["h"] = h_val + rho_i * bar_lam
            solver_kwargs["b"] = b_val + rho_i * bar_mu
            
            sol = self.solver(**solver_kwargs)
            z = sol["x"]
            
            # If not the final step, stop gradient and update reference iterables
            if i < self.num_steps - 1:
                bar_x = jax.lax.stop_gradient(z[:self.n_x])
                bar_lam = jax.lax.stop_gradient(z[self.n_x : self.n_x + self.n_in])
                bar_mu = jax.lax.stop_gradient(z[self.n_x + self.n_in :])
            
        # 3. Format final solution output mapping
        sol["x"] = z[:self.n_x]
        sol["lam"] = -z[self.n_x : self.n_x + self.n_in]
        sol["mu"] = -z[self.n_x + self.n_in :]
        
        # Cleanup previously generated default keys if they exist
        if "z" in sol: sol.pop("z")
        if "y" in sol: sol.pop("y")
        
        return sol