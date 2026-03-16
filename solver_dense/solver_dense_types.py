"""
Every solver will have its own types, that is, a structure with which you can
pass data to the solver / differentiator.
It makes sense to keep this separate from the rest.
Note that these are solver-specific but they are also used in the differentiator.

"""

#TODO: docstring

from typing import TypedDict
from jax import Array
from numpy import ndarray

# Dense QP ingredients
#
#  A  : (n_eq, n_var)     equality lhs
#  b  : (n_eq,)           equality rhs
#  G  : (n_in, n_var)     inequality lhs
#  h  : (n_in,)           inequality rhs
#  P  : (n_var, n_var)    Hessian
#  q  : (n_var,)          linear term
#  c  : () or (1,)        constant scalar
#
class DenseProblemIngredients(TypedDict, total=False):
    A: Array
    b: Array
    G: Array
    h: Array
    P: Array
    q: Array

class DenseProblemIngredientsNP(TypedDict, total=False):
    A: ndarray
    b: ndarray
    G: ndarray
    h: ndarray
    P: ndarray
    q: ndarray