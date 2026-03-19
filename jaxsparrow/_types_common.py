from jaxtyping import Float, Bool
from numpy import ndarray
from jax import Array
from typing import TypedDict, NamedTuple

class QPOutput(TypedDict):
    x:      Float[Array, "n_var"]       |  Float[Array, "n_batch n_var"]
    lam:    Float[Array, "n_ineq"]      |  Float[Array, "n_batch n_ineq"]
    mu:     Float[Array, "n_eq"]        |  Float[Array, "n_batch n_eq"]

class QPOutputNP(NamedTuple):
    x:      Float[ndarray, "n_var"]     |  Float[ndarray, "n_batch n_var"]
    lam:    Float[ndarray, "n_ineq"]    |  Float[ndarray, "n_batch n_ineq"]
    mu:     Float[ndarray, "n_eq"]      |  Float[ndarray, "n_batch n_eq"]
    active: Bool[ndarray, "n_ineq"]     |  Bool[ndarray,  "n_batch n_ineq"]

class QPDiffOut(TypedDict):
    x   : Float[Array, "n_var"]         |  Float[Array, "n_batch n_var"] 
    lam : Float[Array, "n_ineq"]        |  Float[Array, "n_batch n_ineq"]
    mu  : Float[Array, "n_eq"]          |  Float[Array, "n_batch n_eq"]

class QPDiffOutNP(NamedTuple):
    x_np   : Float[ndarray, "n_var"]    |  Float[ndarray, "n_batch n_var"] 
    lam_np : Float[ndarray, "n_ineq"]   |  Float[ndarray, "n_batch n_ineq"]
    mu_np  : Float[ndarray, "n_eq"]     |  Float[ndarray, "n_batch n_eq"]