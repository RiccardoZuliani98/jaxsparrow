#TODO: docstring

from typing import TypedDict, NamedTuple
from jax import Array
from numpy import ndarray
from jaxtyping import Float, Bool

class DenseQPIngredients(TypedDict, total=False):
    P: Float[Array, "n_var n_var"]
    q: Float[Array, "n_var"]
    A: Float[Array, "n_eq n_var"]
    b: Float[Array, "n_eq"]
    G: Float[Array, "n_ineq n_var"]
    h: Float[Array, "n_ineq"]

class DenseQPIngredientsFull(TypedDict):
    P: Float[Array, "n_var n_var"]
    q: Float[Array, "n_var"]
    A: Float[Array, "n_eq n_var"]
    b: Float[Array, "n_eq"]
    G: Float[Array, "n_ineq n_var"]
    h: Float[Array, "n_ineq"]

class DenseQPIngredientsNP(TypedDict, total=False):
    P: Float[ndarray, "n_var n_var"]
    q: Float[ndarray, "n_var"]
    A: Float[ndarray, "n_eq n_var"]
    b: Float[ndarray, "n_eq"]
    G: Float[ndarray, "n_ineq n_var"]
    h: Float[ndarray, "n_ineq"]

class DenseQPIngredientsNPFull(TypedDict):
    P: Float[ndarray, "n_var n_var"]
    q: Float[ndarray, "n_var"]
    A: Float[ndarray, "n_eq n_var"]
    b: Float[ndarray, "n_eq"]
    G: Float[ndarray, "n_ineq n_var"]
    h: Float[ndarray, "n_ineq"]

class DenseQPIngredientsTangentsNP(TypedDict, total=False):
    P: Float[ndarray, "n_var n_var"]    |  Float[ndarray, "n_batch n_var n_var"]
    q: Float[ndarray, "n_var"]          |  Float[ndarray, "n_batch n_var"]
    A: Float[ndarray, "n_eq n_var"]     |  Float[ndarray, "n_batch n_eq n_var"]
    b: Float[ndarray, "n_eq"]           |  Float[ndarray, "n_batch n_eq"]
    G: Float[ndarray, "n_ineq n_var"]   |  Float[ndarray, "n_batch n_ineq n_var"]
    h: Float[ndarray, "n_ineq"]         |  Float[ndarray, "n_batch n_ineq"]

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