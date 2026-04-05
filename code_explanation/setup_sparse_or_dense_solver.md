**Inputs**

* [[constructor_options]]
* fixed_elements
* sparsity_patterns in sparse mode

First it assembles the fixed element (either sparse or dense), then recreates the bwd shape using only the nonzero entries this time.

Then calls an outside constructor to obtain [[solver_numpy]], [[diff_numpy]], and [[converter]].

Finally it calls [[build_solver]].