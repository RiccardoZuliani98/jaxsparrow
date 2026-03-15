This is how a user should call the function

```python
>>> solver = sOPT(dimensions,solver_options,diff_options)
```
Suppose the conic program looks like
$$
\begin{aligned}
\operatorname*{minimize}_x & \quad \frac{1}{2} x^\top P x + q^\top x \\
\text{subject to} & \quad Ax = b,\\
& \quad Gx-h \in \mathcal{K},
\end{aligned}
$$
where $\mathcal{K}$ is a convex cone (could be non-negative orthant, leading to a QP).
Then solve using
```python
>>> primal, dual, value = solver(P,q,A,b,G,h,warmstart)
```
All these values should be differentiable, primal-dual through some IFT, value through the envelope theorem.

First question: how can we implement a function "solver" with custom forward and backward?

I think we need to use "pure_callback" ([Pure callback in Jax](https://docs.jax.dev/en/latest/_autosummary/jax.pure_callback.html#jax.pure_callback)) since this is the only one that allows differentiation ([External-callbacks in Jax](https://docs.jax.dev/en/latest/external-callbacks.html)).

This can also work with vmap and jit!

---

Problems we need to address:

1. when differentiating, we may want to use numpy or scipy, but this should be done avoiding converting back and forth twice;
2. if multiple RHS are passed when differentiating, this method runs the differentiator multiple times, wasting a lot of time. Can we somehow implement a custom vmap for the jvp (or vjp) so that multiple tangents are implemented as different columns in tangents?
3. the custom jvp has to be implemented in pure jax or else we can't jit the process and there's no point even doing this because it's too slow.
4. when running jit we can't allow the linear system to have a variable dimension, because jax has to preallocate. This means we either need to keep the inactive multipliers, or write a second pure callback for the linear solve. 

