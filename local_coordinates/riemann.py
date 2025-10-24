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
from local_coordinates.tensor import TensorType
from local_coordinates.connection import Connection, get_levi_civita_connection

class RiemannCurvatureTensor(Tensor):
  """
  The Riemann curvature tensor is a tensor that measures the curvature of a Riemannian manifold.
  """
  tensor_type: TensorType = eqx.field(static=True)
  basis: BasisVectors
  components: Annotated[Jet, "D D D D"] # The components of the tensor written in the chosen basis

  @property
  def batch_size(self):
    return self.basis.batch_size

  def __check_init__(self):
    super().__check_init__()
    if self.components.shape[-4] != self.components.shape[-3]:
      raise ValueError(f"Riemann curvature tensor must be a 4-index tensor")

    expected_batch_shape = self.basis.p.shape[:-1]
    actual_batch_shape = self.components.shape[:-4]
    if expected_batch_shape != actual_batch_shape:
        raise ValueError(
            f"Batch shape mismatch: basis implies {expected_batch_shape} but components have {actual_batch_shape}"
        )

def get_riemann_curvature_tensor(connection: Connection) -> RiemannCurvatureTensor:
  """
  Get the Riemann curvature tensor from a connection.
  R^i_{jkl} = partial_k Gamma^i_{jl} - partial_l Gamma^i_{jk} + Gamma^i_{mk} Gamma^m_{jl} - Gamma^i_{ml} Gamma^m_{jk}
  """
  # Ensure the calculation is done in the standard basis where the formula is simplest
  standard_basis = get_standard_basis(connection.basis.p)
  connection_std = change_basis(connection, standard_basis)

  basis: BasisVectors = connection_std.basis
  gamma = connection_std.christoffel_symbols

  @jet_decorator
  def get_components(gamma_val, gamma_grad) -> Array:
    # Term 1: partial_k Gamma^i_{jl} -> dGamma(i,j,l,k)
    term1 = jnp.einsum("ijlk->ijkl", gamma_grad)
    # Term 2: - partial_l Gamma^i_{jk} -> -dGamma(i,j,k,l)
    term2 = -jnp.einsum("ijkl->ijkl", gamma_grad)
    # Term 3: Gamma^i_{mk} Gamma^m_{jl}
    term3 = jnp.einsum("imk,mjl->ijkl", gamma_val, gamma_val)
    # Term 4: - Gamma^i_{ml} Gamma^m_{jk}
    term4 = -jnp.einsum("iml,mjk->ijkl", gamma_val, gamma_val)

    return term1 + term2 + term3 + term4

  gamma_value_jet = gamma.get_value_jet()
  gamma_gradient_jet = gamma.get_gradient_jet()

  # The resulting jet will have gradient=None and hessian=None
  # because the get_components function doesn't produce them.
  riemann_components = get_components(gamma_value_jet, gamma_gradient_jet)

  return RiemannCurvatureTensor(
    tensor_type=TensorType(k=3, l=1),
    basis=basis,
    components=riemann_components
  )
