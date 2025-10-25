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
from local_coordinates.frame import Frame, get_lie_bracket_between_frame_pairs, basis_to_frame
from local_coordinates.tangent import TangentVector

class Connection(AbstractBatchableObject):
  """
  A Connection is a map from one tangent space to another.
  The components of the Christoffel symbols, Gamma^k_{ij}, are to be
  indexed as Gamma^k_{ij} = christoffel_symbols[i, j, k].
  """
  basis: BasisVectors
  christoffel_symbols: Annotated[Jet, "D D N"] # The components of the Christoffel symbols written in the chosen basis

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
      term2 = jnp.einsum("ijk,i,j->k", gamma_val, x_val, y_val)
      return term1 + term2

    E_value_jet = self.basis.components.get_value_jet()
    gamma_value_jet = self.christoffel_symbols.get_value_jet()
    x_value_jet = X.components.get_value_jet()
    y_value_jet = Y.components.get_value_jet()
    y_gradient_jet = Y.components.get_gradient_jet()
    new_components: Jet = components(E_value_jet, gamma_value_jet, x_value_jet, y_value_jet, y_gradient_jet)
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
    term1 = jnp.einsum("lj,ai,km,alm->ijk", T_val, T_val_inv, T_val_inv, christoffel_symbols_val)
    term2 = jnp.einsum("ai,mja,km->ijk", E_tilde_val, T_grad, T_val_inv)
    return term1 + term2

  cs_value_jet = connection.christoffel_symbols.get_value_jet()
  T_value_jet = T_jet.get_value_jet()
  T_gradient_jet = T_jet.get_gradient_jet()
  E_tilde_val_jet = new_basis.components.get_value_jet()
  new_christoffel_symbols = get_components(cs_value_jet, E_tilde_val_jet, T_value_jet, T_gradient_jet)
  return Connection(basis=new_basis, christoffel_symbols=new_christoffel_symbols)

def get_levi_civita_connection(metric: RiemannianMetric) -> Connection:
  """Get the Levi-Civita connection from a Riemannian metric in terms of the basis of the metric.
  """
  basis: BasisVectors = metric.basis

  frame = basis_to_frame(basis)
  lie_bracket_pairs: Annotated[TangentVector, "D D"] = get_lie_bracket_between_frame_pairs(frame)

  @jet_decorator
  def get_christoffel_symbols(E_val, g_val, g_grad, c_val) -> Array:
    r"""
    \Gamma_{ij}^m = \frac{1}{2}\left(E_i(g_{jk})g^{km} + E_j(g_{ik})g^{km} - E_k(g_{ij})g^{km} + c_{ij}^m - c_{ik}^l g_{lj}g^{km} - c_{jk}^l g_{li}g^{km}\right)
    """
    g_val_inv = jnp.linalg.inv(g_val)

    # E_i(g_{jk})g^{km}
    term1 = jnp.einsum("ai,jka,km->ijm", E_val, g_grad, g_val_inv)

    # E_j(g_{ik})g^{km}
    term2 = jnp.einsum("aj,ika,km->ijm", E_val, g_grad, g_val_inv)

    # -E_k(g_{ij})g^{km}
    term3 = -jnp.einsum("ak,ija,km->ijm", E_val, g_grad, g_val_inv)

    # c_{ij}^m
    term4 = jnp.einsum("ijm->ijm", c_val)

    # -c_{ik}^l g_{lj}g^{km}
    term5 = -jnp.einsum("ikl,lj,km->ijm", c_val, g_val, g_val_inv)

    # -c_{jk}^l g_{li}g^{km}
    term6 = -jnp.einsum("jkl,li,km->ijm", c_val, g_val, g_val_inv)

    return 0.5 * (term1 + term2 + term3 + term4 + term5 + term6)

  E_val: Jet = basis.components.get_value_jet()
  g_val: Jet = metric.components.get_value_jet()
  g_grad: Jet = metric.components.get_gradient_jet()
  c_val: Jet = lie_bracket_pairs.components.get_value_jet()

  christoffel_symbols: Jet = get_christoffel_symbols(E_val, g_val, g_grad, c_val)
  return Connection(basis=basis, christoffel_symbols=christoffel_symbols)
