
def _convert_qp_ingredients_to_numpy(P, q, A, b, G, h):

    # Convert vectors
    start = perf_counter()
    q_np = np.asarray(q, dtype=_dtype).squeeze()
    b_np = np.asarray(b, dtype=_dtype).squeeze()
    h_np = np.asarray(h, dtype=_dtype).squeeze()
    t_convert = {"convert_vectors": perf_counter() - start}

    # Convert matrices
    start = perf_counter()
    P_np = np.asarray(P, dtype=_dtype).squeeze()
    A_np = np.asarray(A, dtype=_dtype).squeeze()
    G_np = np.asarray(G, dtype=_dtype).squeeze()
    t_convert["convert_matrices"] = perf_counter() - start

    output_dict = {
        "P": P_np, 
        "q": q_np, 
        "A": A_np, 
        "b": b_np, 
        "G": G_np, 
        "h": h_np
    }

    return output_dict, t_convert

def _convert_solution_to_jax(x, lam, mu, active, batch_size=0, n_var=0, n_eq=0, n_ineq=0):
    start = perf_counter()
    if batch_size > 0:
        sol =  {
            "x": jnp.broadcast_to(jnp.array(x,dtype=_dtype),(batch_size, n_var)),
            "lam":jnp.broadcast_to(jnp.array(lam,dtype=_dtype),(batch_size,n_ineq)),
            "mu":jnp.broadcast_to(jnp.array(mu,dtype=_dtype),(batch_size,n_eq)),
            "active":jnp.broadcast_to(jnp.array(active,dtype=jnp.bool_),(batch_size,n_ineq))
        }
    else:
        sol =  {
            "x": jnp.array(x,dtype=_dtype),
            "lam":jnp.array(lam,dtype=_dtype),
            "mu":jnp.array(mu,dtype=_dtype),
            "active":jnp.array(active,dtype=jnp.bool_)
        }

    t_convert = perf_counter() - start
    return sol, t_convert