This is how a user should call the function

```python
solver = sOPT(dimensions,solver_options,diff_options)
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
primal, dual, value = solver(P,q,A,b,G,h,warmstart)
```