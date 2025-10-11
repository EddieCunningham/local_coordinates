from typing import Any, Callable, Tuple, Annotated, Optional, List
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from linsdex import AbstractBatchableObject
from local_coordinates.basis import BasisVectors, get_basis_transform
from plum import dispatch
from local_coordinates.tensor import Tensor, change_coordinates
from local_coordinates.jet import Jet, _expand_jet
from local_coordinates.jet import jet_decorator

class Connection(AbstractBatchableObject):
  """
  A Connection is a map from one tangent space to another.
  """
  basis: BasisVectors
  christoffel_symbols: Annotated[Jet, "N D D"] # The components of the Christoffel symbols written in the chosen basis

  @property
  def batch_size(self):
    # Delegate batching to underlying basis
    return self.basis.batch_size

  def covariant_derivative(self, X: Annotated[Tensor, "0 1"], Y: Annotated[Tensor, "0 1"]) -> Tensor:
    assert X.tensor_type.is_vector(), f"X must be a vector, got {X.tensor_type}"

    # Make sure X and Y are written in the same basis
    X: Tensor = change_coordinates(X, self.basis)
    Y: Tensor = change_coordinates(Y, self.basis)

    # Compute the covariant derivative
    @jet_decorator
    def components(gamma_val, x_val, y_val, y_grad_val) -> Array:
      term1 = jnp.einsum("i,ki->k", x_val, y_grad_val)
      term2 = jnp.einsum("kij,i,j->k", gamma_val, x_val, y_val)
      return term1 + term2

    gamma_value_jet, _, _ = _expand_jet(self.christoffel_symbols)
    x_value_jet, _, _ = _expand_jet(X.components)
    y_value_jet, y_gradient_jet, _ = _expand_jet(Y.components)
    new_components = components(gamma_value_jet, x_value_jet, y_value_jet, y_gradient_jet)
    return Tensor(X.tensor_type, self.basis, new_components)

@dispatch
def change_coordinates(connection: Connection, new_basis: BasisVectors) -> Connection:
  """
  Transform a connection from one basis to another. The Christoffel symbols
  are not tensors, so the transformation rule is a bit different.
  """
  T_jet: Jet = get_basis_transform(connection.basis, new_basis)

  @jet_decorator
  def get_components(christoffel_symbols_val, T_val, T_grad) -> Jet:
    term1 = jnp.einsum("ai,cja,kc->kij", T_val, T_grad, T_val)
    term2 = jnp.einsum("cab,ai,bj,kc->kij", christoffel_symbols_val, T_val, T_val, T_val)
    return term1 + term2

  cs_value_jet, _, _ = _expand_jet(connection.christoffel_symbols)
  T_value_jet, T_gradient_jet, _ = _expand_jet(T_jet)
  new_christoffel_symbols = get_components(cs_value_jet, T_value_jet, T_gradient_jet)
  return Connection(basis=new_basis, christoffel_symbols=new_christoffel_symbols)
