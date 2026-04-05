#todo: there seems to be a double conversion in diff_reverse for primals.
#todo: here is where I need to get rid of numpy => jax conversion
#todo: the shape of the backward should only take into account the nonzero entries, not all the entries. We could pass this as an argument since it anyways is computed by the functions calling build_solver.

**INPUTS**

* [[converter]]
* [[constructor_options]]
* bwd_shapes
* [[solver_numpy]]
* [[diff_numpy]]
* fixed_key_set


**WHAT IT DOES**

Then it defines the solver, forward and backward differentiator.

The primals are passed as Arrays or BCOO and always converted to numpy through the converter, then passed to *solver_numpy*.

The forward differentiator first understands if batching was applied and gets the batch dimension, then runs *diff_forward_numpy*.

Similarly, the reverse runs conversion and *diff_reverse_numpy*.

It defines the shapes of the inputs / outputs. Then it defines the custom function with the pure callbacks.