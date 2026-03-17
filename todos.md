* the dense solver seems reasonable, I need to check the warmstarting.
* I need to check all tests and verify if it's fast.
* Next I need to verify if vmap is leveraged when computing closed-loop derivatives of multiple parameters.
* Diff algorithms and qp solvers should be provided as options, this way we can use the same file for multiple qp solvers / differentiation algorithms.

Then we should move on to:
* sparse solver (get code from bpmpc_jax) using scipy. Major difference: you have to provide the shapes of the matrices.
* vjp mode directly instead of jvp, this will likely require a brand new file.