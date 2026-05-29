import numpy as np
import scipy.sparse as sp

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from jaxsparrow import setup_sparse_solver
from jaxsparrow._types_common import SolverOutput

class RegularizedQPSolver:
    
    def __init__(
        self, 
        n_x, 
        n_in, 
        n_eq, 
        num_steps=1, 
        sparsity_patterns=None, 
        fixed_elements=None, 
        rho=1e-4, 
        options=None
    ) -> None:
        
        # original state dimension
        self.n_x = n_x

        # number of inequality constraints
        self.n_in = n_in

        # number of equality constraints
        self.n_eq = n_eq

        # total number of variables in the regularized problem
        self.n_z = n_x + n_in + n_eq

        # number of times the regularized problem will be solved
        self.num_steps = num_steps
        
        # if rho is passed as a sequence, then it must match the 
        # number of steps
        #TODO: is there a more robust way to handle this?
        # maybe more strict type checking?
        if isinstance(rho, (list, tuple, np.ndarray)):
            self.rho = jnp.array(rho)
            if len(self.rho) != self.num_steps:
                raise ValueError(
                    f"Length of rho list ({len(self.rho)}) must match num_steps ({self.num_steps})."
                )
        # otherwise, it must be a scalar and every problem will have the same value
        else:
            self.rho = jnp.full(self.num_steps, rho)
            
        # Use the first rho value to instantiate any fixed augmented matrices
        rho_init = float(self.rho[0])
        
        self.fixed_elements = fixed_elements or {}
        sparsity_patterns = sparsity_patterns or {}
        
        # Helper to ensure fixed matrices are scipy sparse CSC format
        def to_csc(mat):
            if not sp.issparse(mat):
                return sp.csc_matrix(mat)
            return mat.tocsc()
        
        # sparse identity matrices for all variables
        I_nx = sp.eye(n_x, format='csc')
        I_nin = sp.eye(n_in, format='csc')
        I_neq = sp.eye(n_eq, format='csc')
        
        # preallocate dictionaries for augmented fixed ingredients
        aug_fixed_elements = {}
        # and for augmented sparsity patterns
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
            aug_fixed_elements["G"] = sp.bmat([[G_fixed, -rho_init * I_nin, sp.csc_matrix((n_in, n_eq))]], format='csc')
        else:
            G_pat = sparsity_patterns["G"]
            G_rho_idx = jnp.stack([diag_nin, n_x + diag_nin], axis=-1)
            self.G_tilde_indices = jnp.vstack([G_pat.indices, G_rho_idx])
            self.G_tilde_shape = (n_in, self.n_z)
            aug_patterns["G"] = BCOO((jnp.ones(len(self.G_tilde_indices)), self.G_tilde_indices), shape=self.G_tilde_shape)

        # --- Handle A ---
        if "A" in self.fixed_elements:
            A_fixed = to_csc(self.fixed_elements["A"])
            aug_fixed_elements["A"] = sp.bmat([[A_fixed, sp.csc_matrix((n_eq, n_in)), -rho_init * I_neq]], format='csc')
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
            options = options or {"diff_mode":"rev"},
        )
        
        self.solve = jax.jit(self._solve_impl)
    
    # Helper to strip AD tracers from both standard arrays and PyTrees (like BCOO)
    @staticmethod
    def detach(tree):
        if tree is None: return None
        # return jax.tree_util.tree_map(jax.lax.stop_gradient, tree)
        return tree

    def _solve_impl(
        self, 
        bar_x, 
        bar_lam, 
        bar_mu, 
        P=None, 
        q=None, 
        A=None, 
        b=None, 
        G=None, 
        h=None
    ) -> SolverOutput:
            
        # 1. Resolve Vectors (Differentiable versions)
        q_val = self.q_fixed if q is None else q
        h_val = self.h_fixed if h is None else h
        b_val = self.b_fixed if b is None else b

        # Detach everything for the burn-in phase to prevent AD tracking
        P_sg, G_sg, A_sg = self.detach(P), self.detach(G), self.detach(A)
        q_sg, h_sg, b_sg = self.detach(q_val), self.detach(h_val), self.detach(b_val)
        
        bar_x = self.detach(bar_x)
        bar_lam = self.detach(bar_lam)
        bar_mu = self.detach(bar_mu)
        
        # 2. Burn-in Phase (Steps 0 to num_steps - 2)
        for i in range(self.num_steps - 1):

            rho_i = self.rho[i]
            solver_kwargs = {}

            if "P" not in self.fixed_elements:
                P_data = jnp.concatenate([P_sg.data, jnp.full(self.n_z, rho_i)])
                solver_kwargs["P"] = BCOO((P_data, self.P_tilde_indices), shape=self.P_tilde_shape)
                
            if "G" not in self.fixed_elements:
                G_data = jnp.concatenate([G_sg.data, jnp.full(self.n_in, -rho_i)])
                solver_kwargs["G"] = BCOO((G_data, self.G_tilde_indices), shape=self.G_tilde_shape)
                
            if "A" not in self.fixed_elements:
                A_data = jnp.concatenate([A_sg.data, jnp.full(self.n_eq, -rho_i)])
                solver_kwargs["A"] = BCOO((A_data, self.A_tilde_indices), shape=self.A_tilde_shape)
                
            solver_kwargs["q"] = jnp.concatenate([q_sg - rho_i * bar_x, jnp.zeros(self.n_eq + self.n_in)])
            solver_kwargs["h"] = h_sg - rho_i * bar_lam
            solver_kwargs["b"] = b_sg - rho_i * bar_mu
            
            z = self.solver(**solver_kwargs)["x"]
            
            # Update guesses
            bar_x = z[:self.n_x]
            bar_lam = z[self.n_x : self.n_x + self.n_in]
            bar_mu = z[self.n_x + self.n_in :]

        # Detach the intermediate results one last time to ensure no leaks
        bar_x = self.detach(bar_x)
        bar_lam = self.detach(bar_lam)
        bar_mu = self.detach(bar_mu)

        # 3. Final Step (Differentiable)
        # Now we inject the original, gradient-carrying matrices and vectors.
        rho_final = self.rho[-1]
        solver_kwargs_final = {}
        
        if "P" not in self.fixed_elements:
            # P.data carries the AD tracer here
            P_data = jnp.concatenate([P.data, jnp.full(self.n_z, rho_final)])
            solver_kwargs_final["P"] = BCOO((P_data, self.P_tilde_indices), shape=self.P_tilde_shape)
            
        if "G" not in self.fixed_elements:
            G_data = jnp.concatenate([G.data, jnp.full(self.n_in, -rho_final)])
            solver_kwargs_final["G"] = BCOO((G_data, self.G_tilde_indices), shape=self.G_tilde_shape)
            
        if "A" not in self.fixed_elements:
            A_data = jnp.concatenate([A.data, jnp.full(self.n_eq, -rho_final)])
            solver_kwargs_final["A"] = BCOO((A_data, self.A_tilde_indices), shape=self.A_tilde_shape)
            
        # q_val, h_val, and b_val carry the AD tracers here
        solver_kwargs_final["q"] = jnp.concatenate([q_val - rho_final * bar_x, jnp.zeros(self.n_eq + self.n_in)])
        solver_kwargs_final["h"] = h_val - rho_final * bar_lam
        solver_kwargs_final["b"] = b_val - rho_final * bar_mu
        
        z_final = self.solver(**solver_kwargs_final)["x"]
        
        # 4. Format final solution output mapping
        x_final = z_final[:self.n_x]
        lam_final = z_final[self.n_x : self.n_x + self.n_in]
        mu_final = z_final[self.n_x + self.n_in :]

        return {"x":x_final,"lam":lam_final,"mu":mu_final}