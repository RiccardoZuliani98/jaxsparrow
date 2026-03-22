from jaxtyping import Float
from numpy import ndarray
from jax import Array
from typing import TypedDict, NamedTuple, Protocol

class SolverOutput(TypedDict):
    x:      Float[Array, "n_var"]       |  Float[Array, "n_batch n_var"]
    lam:    Float[Array, "n_ineq"]      |  Float[Array, "n_batch n_ineq"]
    mu:     Float[Array, "n_eq"]        |  Float[Array, "n_batch n_eq"]

class SolverOutputNP(NamedTuple):
    x:      Float[ndarray, "n_var"]     |  Float[ndarray, "n_batch n_var"]
    lam:    Float[ndarray, "n_ineq"]    |  Float[ndarray, "n_batch n_ineq"]
    mu:     Float[ndarray, "n_eq"]      |  Float[ndarray, "n_batch n_eq"]

class SolverDiffOut(TypedDict):
    x   : Float[Array, "n_var"]         |  Float[Array, "n_batch n_var"] 
    lam : Float[Array, "n_ineq"]        |  Float[Array, "n_batch n_ineq"]
    mu  : Float[Array, "n_eq"]          |  Float[Array, "n_batch n_eq"]

class SolverDiffOutNP(NamedTuple):
    x_np   : Float[ndarray, "n_var"]    |  Float[ndarray, "n_batch n_var"] 
    lam_np : Float[ndarray, "n_ineq"]   |  Float[ndarray, "n_batch n_ineq"]
    mu_np  : Float[ndarray, "n_eq"]     |  Float[ndarray, "n_batch n_eq"]

class Solver(Protocol):

    def __call__(
        self, **kwargs: ndarray,
    ) -> tuple[SolverOutputNP, dict[str, float]]: ...