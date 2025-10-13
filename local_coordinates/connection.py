from typing import Any, Callable, Tuple, Annotated, Optional, List
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from linsdex import AbstractBatchableObject
from local_coordinates.basis import BasisVectors, get_basis_transform, get_standard_basis, get_lie_bracket_components
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
    assert X.tensor_type.is_vector(), f"X must be a vector, got {X.tensor_type}"

    # Convert everything to local coordinates
    standard_basis = get_standard_basis(self.basis.p)

    # Make sure X and Y are written in the same basis
    X: Tensor = change_coordinates(X, standard_basis)
    Y: Tensor = change_coordinates(Y, standard_basis)
    conn: Connection = change_coordinates(self, standard_basis)

    # Compute the covariant derivative
    @jet_decorator
    def components(gamma_val, x_val, y_val, y_grad_val) -> Array:
      term1 = jnp.einsum("i,ki->k", x_val, y_grad_val)
      term2 = jnp.einsum("kij,i,j->k", gamma_val, x_val, y_val)
      return term1 + term2

    gamma_value_jet = conn.christoffel_symbols.get_value_jet()
    x_value_jet = X.components.get_value_jet()
    y_value_jet = Y.components.get_value_jet()
    y_gradient_jet = Y.components.get_gradient_jet()
    new_components = components(gamma_value_jet, x_value_jet, y_value_jet, y_gradient_jet)
    return Tensor(X.tensor_type, standard_basis, new_components)

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
  basis: BasisVectors = metric.basis
  c_kij: Jet = get_lie_bracket_components(basis)

  # Get the metric components
  g_ijk: Jet = metric.components.get_gradient_jet()

  @jet_decorator
  def get_christoffel_symbols(c_kij_val, g_ij_val, g_ijk_grad) -> Array:
    """
    Gamma_{kij} = 1/2 (g_{jk,i} + g_{ij,j} - g_{ij,k} + c_{kij} - c_{jik} - c_{ijk})
    """
    ginv = jnp.linalg.inv(g_ij_val)              # g^{kl}

    # Metric derivative terms: T1+T2−T3 with A[i,j,k] = g_{ij,k}
    T1 = jnp.einsum("kl,jli->kij", ginv, g_ijk_grad)   # g^{kl} ∂_i g_{jl}
    T2 = jnp.einsum("kl,ilj->kij", ginv, g_ijk_grad)   # g^{kl} ∂_j g_{il}
    T3 = jnp.einsum("kl,ijl->kij", ginv, g_ijk_grad)   # g^{kl} ∂_l g_{ij}

    # Bracket terms: +C^k_{ij} − g^{kl} g_{jm} C^m_{il} − g^{kl} g_{im} C^m_{jl}
    B1 = c_kij_val
    B2 = jnp.einsum("kl, jm, mil -> kij", ginv, g_ij_val, c_kij_val)             # − g^{kl} g_{jm} c_kij_val^m_{il}
    B3 = jnp.einsum("kl,im,mjl->kij", ginv, g_ij_val, c_kij_val)  # − g^{kl} g_{im} C^m_{jl}

    gamma_upper = 0.5 * (T1 + T2 - T3 + B1 - B2 - B3)
    return gamma_upper

  c_kij_val = c_kij.get_value_jet()
  g_ij_val = metric.components.get_value_jet()
  g_ijk_grad = g_ijk.get_value_jet()
  christoffel_symbols: Jet = get_christoffel_symbols(c_kij_val, g_ij_val, g_ijk_grad)
  return Connection(basis=basis, christoffel_symbols=christoffel_symbols)




