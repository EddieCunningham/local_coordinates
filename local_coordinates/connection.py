from typing import Any, Callable, Tuple, Annotated, Optional, List
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from linsdex import AbstractBatchableObject
from local_coordinates.basis import BasisVectors, get_basis_transform, get_standard_basis
from plum import dispatch
from local_coordinates.tensor import Tensor, change_basis
from local_coordinates.jet import Jet
from local_coordinates.jet import jet_decorator
from local_coordinates.metric import RiemannianMetric
from local_coordinates.jet import get_identity_jet
from local_coordinates.frame import Frame, get_lie_bracket_between_frame_pairs
from local_coordinates.tangent import TangentVector
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

  def covariant_derivative(self, X: TangentVector, Y: TangentVector) -> TangentVector:
    if not isinstance(X, TangentVector) or not isinstance(Y, TangentVector):
      raise ValueError("Only supporting covariant derivative of tangent vectors")

    # Make sure X and Y are written in the same basis
    X: TangentVector = change_basis(X, self.basis)
    Y: TangentVector = change_basis(Y, self.basis)

    # Compute the covariant derivative
    @jet_decorator
    def components(E_val, gamma_val, x_val, y_val, y_grad_val) -> Array:
      # (∇_X Y)^k = X^i E_i(Y^k) + Γ^k_{ij} X^i Y^j,
      # where E_i(Y^k) = E_i^a ∂_a Y^k.
      term1 = jnp.einsum("i,ai,ka->k", x_val, E_val, y_grad_val)
      term2 = jnp.einsum("kij,i,j->k", gamma_val, x_val, y_val)
      return term1 + term2

    E_value_jet = self.basis.components.get_value_jet()
    gamma_value_jet = self.christoffel_symbols.get_value_jet()
    x_value_jet = X.components.get_value_jet()
    y_value_jet = Y.components.get_value_jet()
    y_gradient_jet = Y.components.get_gradient_jet()
    new_components = components(E_value_jet, gamma_value_jet, x_value_jet, y_value_jet, y_gradient_jet)
    return TangentVector(self.basis.p, new_components, self.basis)

@dispatch
def change_basis(connection: Connection, new_basis: BasisVectors) -> Connection:
  """
  Transform a connection from one basis to another. The Christoffel symbols
  are not tensors, so the transformation rule is a bit different.
  """
  T_jet: Jet = get_basis_transform(connection.basis, new_basis)

  @jet_decorator
  def get_components(christoffel_symbols_val, E_tilde_val, T_val, T_grad) -> Jet:
    T_val_inv = jnp.linalg.inv(T_val)
    term1 = jnp.einsum("lj,ai,km,mal->kij", T_val, T_val_inv, T_val_inv, christoffel_symbols_val)
    term2 = jnp.einsum("ai,mja,km->kij", E_tilde_val, T_grad, T_val_inv)
    return term1 + term2

  cs_value_jet = connection.christoffel_symbols.get_value_jet()
  T_value_jet = T_jet.get_value_jet()
  T_gradient_jet = T_jet.get_gradient_jet()
  E_tilde_val_jet = new_basis.components.get_value_jet()
  new_christoffel_symbols = get_components(cs_value_jet, E_tilde_val_jet, T_value_jet, T_gradient_jet)
  return Connection(basis=new_basis, christoffel_symbols=new_christoffel_symbols)

def get_levi_civita_connection(metric: RiemannianMetric) -> Connection:
  # Accept both primal and dual bases on the metric; convert to primal if needed
  basis: BasisVectors = metric.basis

  N = basis.p.shape[0]
  frame = Frame(p=basis.p, basis=basis, components=get_identity_jet(N))
  lie_bracket_pairs: Annotated[TangentVector, "D D"] = get_lie_bracket_between_frame_pairs(frame)

  ax = (0, None)
  vmapped_change_basis = eqx.filter_vmap(eqx.filter_vmap(change_basis, in_axes=ax), in_axes=ax)
  lie_bracket_pairs = vmapped_change_basis(lie_bracket_pairs, basis)

  # Get the metric components
  g_ijk: Jet = metric.components.get_gradient_jet()

  @jet_decorator
  def get_christoffel_symbols(c_kij_val, g_ij_val, g_ijk_grad) -> Array:
    """
    Gamma_{kij} = 1/2 (g_{jk,i} + g_{ik,j} - g_{ij,k} + c_{kij} - c_{jik} - c_{ijk})
    """
    ginv = jnp.linalg.inv(g_ij_val)              # g^{kl}

    c_kij_lower = jnp.einsum("mij,mk->kij", c_kij_val, g_ij_val)

    # Arrange all terms with axes (k,i,j):
    # t1[k,i,j] = g_{jk,i}
    t1 = jnp.einsum("jki->kij", g_ijk_grad)
    # t2[k,i,j] = g_{ik,j}
    t2 = jnp.einsum("ikj->kij", g_ijk_grad)
    # t3[k,i,j] = g_{ij,k}
    t3 = jnp.einsum("ijk->kij", g_ijk_grad)
    t4 = jnp.einsum("kij->kij", c_kij_lower)
    t5 = jnp.einsum("kij->jik", c_kij_lower)
    t6 = jnp.einsum("kij->ijk", c_kij_lower)

    gamma_lower = 0.5 * (t1 + t2 - t3 + t4 - t5 - t6)
    gamma_upper = jnp.einsum("km,mij->kij", ginv, gamma_lower)
    return gamma_upper

  c_kij_val = lie_bracket_pairs.components.get_value_jet()
  g_ij_val = metric.components.get_value_jet()
  g_ijk_grad = g_ijk.get_value_jet()
  christoffel_symbols: Jet = get_christoffel_symbols(c_kij_val, g_ij_val, g_ijk_grad)
  return Connection(basis=basis, christoffel_symbols=christoffel_symbols)




