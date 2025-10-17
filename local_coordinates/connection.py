from typing import Any, Callable, Tuple, Annotated, Optional, List
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from linsdex import AbstractBatchableObject
from local_coordinates.basis import BasisVectors, DualBasis, get_basis_transform, get_standard_basis
from local_coordinates.frame import get_lie_bracket_components
from plum import dispatch
from local_coordinates.tensor import Tensor, change_coordinates
from local_coordinates.jet import Jet
from local_coordinates.jet import jet_decorator
from local_coordinates.metric import RiemannianMetric

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
    """
    TODO: Handle covariant derivatives of general tensors.
    """

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

    gamma_value_jet = self.christoffel_symbols.get_value_jet()
    x_value_jet = X.components.get_value_jet()
    y_value_jet = Y.components.get_value_jet()
    y_gradient_jet = Y.components.get_gradient_jet()
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

  cs_value_jet = connection.christoffel_symbols.get_value_jet()
  T_value_jet = T_jet.get_value_jet()
  T_gradient_jet = T_jet.get_gradient_jet()
  new_christoffel_symbols = get_components(cs_value_jet, T_value_jet, T_gradient_jet)
  return Connection(basis=new_basis, christoffel_symbols=new_christoffel_symbols)

def get_levi_civita_connection(metric: RiemannianMetric) -> Connection:
  # Get the lie bracket components of the basis
  dual_basis: DualBasis = metric.basis
  basis: BasisVectors = dual_basis.to_primal()
  c_kij: Jet = get_lie_bracket_components(basis)

  # Get the metric components
  g_ijk: Jet = metric.components.get_gradient_jet()

  @jet_decorator
  def get_christoffel_symbols(c_kij_val, g_ij_val, g_ijk_grad) -> Array:
    """
    Gamma_{kij} = 1/2 (g_{jk,i} + g_{ik,j} - g_{ij,k} + c_{kij} - c_{jik} - c_{ijk})
    """
    ginv = jnp.linalg.inv(g_ij_val)              # g^{kl}

    c_kij_lower = jnp.einsum("mij,mk->kij", c_kij_val, g_ij_val)

    t1 = jnp.einsum("ijk->jki", g_ijk_grad)
    t2 = jnp.einsum("ijk->ikj", g_ijk_grad)
    t3 = jnp.einsum("ijk->ijk", g_ijk_grad)
    t4 = jnp.einsum("kij->kij", c_kij_lower)
    t5 = jnp.einsum("kij->jik", c_kij_lower)
    t6 = jnp.einsum("kij->ijk", c_kij_lower)

    gamma_lower = 0.5 * (t1 + t2 - t3 + t4 - t5 - t6)

    gamma_upper = jnp.einsum("km,mij->kij", ginv, gamma_lower)

    return gamma_upper

  c_kij_val = c_kij.get_value_jet()
  g_ij_val = metric.components.get_value_jet()
  g_ijk_grad = g_ijk.get_value_jet()
  christoffel_symbols: Jet = get_christoffel_symbols(c_kij_val, g_ij_val, g_ijk_grad)
  import pdb; pdb.set_trace()
  return Connection(basis=basis, christoffel_symbols=christoffel_symbols)




