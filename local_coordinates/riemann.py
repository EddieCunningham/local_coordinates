from typing import Any, Callable, Tuple, Annotated, Optional, List
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from local_coordinates.base import AbstractBatchableObject
from local_coordinates.basis import BasisVectors, get_basis_transform, get_standard_basis
from plum import dispatch
from local_coordinates.tensor import Tensor, change_basis
from local_coordinates.jet import Jet
from local_coordinates.jet import jet_decorator
from local_coordinates.metric import RiemannianMetric
from local_coordinates.jet import get_identity_jet
from local_coordinates.frame import Frame, get_lie_bracket_between_frame_pairs, basis_to_frame
from local_coordinates.tangent import TangentVector
from local_coordinates.tensor import TensorType
from local_coordinates.connection import Connection, get_levi_civita_connection

class RiemannCurvatureTensor(Tensor):
  """
  The Riemann curvature tensor is a tensor that measures the curvature of a Riemannian manifold.
  This is really the (3, 1) Riemann curvature endomorphism.  When defined using the Levi-Civita
  connection, the components of the tensor are given by:
  {R_{ijk}}^m = E_i(\Gamma^m_{jk}) - E_j(\Gamma^m_{ik}) + \Gamma^l_{jk}\Gamma^m_{il} - \Gamma^l_{ik}\Gamma^m_{jl} - c^l_{ij}\Gamma^m_{lk}
  """
  tensor_type: TensorType = eqx.field(static=True) # Always TensorType(k=3, l=1)
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

  def __call__(self, X: TangentVector, Y: TangentVector, Z: TangentVector) -> TangentVector:
    """
    Evaluate the Riemann curvature endomorphism at three tangent vectors.
    R(X, Y)Z = nabla_X nabla_Y Z - nabla_Y nabla_X Z - nabla_{[X,Y]} Z
    """
    X: TangentVector = change_basis(X, self.basis)
    Y: TangentVector = change_basis(Y, self.basis)
    Z: TangentVector = change_basis(Z, self.basis)
    @jet_decorator
    def components(X_val, Y_val, Z_val, R_val):
      return jnp.einsum("ijkl,i,j,k->l", R_val, X_val, Y_val, Z_val)
    result_components = components(X.components.get_value_jet(), Y.components.get_value_jet(), Z.components.get_value_jet(), self.components.get_value_jet())
    return TangentVector(p=self.basis.p, components=result_components, basis=self.basis)

def get_riemann_curvature_tensor(connection: Connection) -> RiemannCurvatureTensor:
  """
  Get the (3, 1) Riemann curvature tensor from a connection.
  {R_{ijk}}^m = E_i(\Gamma^m_{jk}) - E_j(\Gamma^m_{ik}) + \Gamma^l_{jk}\Gamma^m_{il} - \Gamma^l_{ik}\Gamma^m_{jl} - c^l_{ij}\Gamma^m_{lk}
  """
  basis: BasisVectors = connection.basis
  gamma = connection.christoffel_symbols

  frame = basis_to_frame(basis)
  lie_bracket_pairs: Annotated[TangentVector, "i j"] = get_lie_bracket_between_frame_pairs(frame)

  @jet_decorator
  def get_components(E_val, gamma_val, gamma_grad, c_val) -> Array:
    # Term 1: E_i(\Gamma^m_{jk})
    term1 = jnp.einsum("ai,jkma->ijkm", E_val, gamma_grad)

    # Term 2: -E_j(\Gamma^m_{ik})
    term2 = -jnp.einsum("aj,ikma->ijkm", E_val, gamma_grad)

    # Term 3: \Gamma^l_{jk}\Gamma^m_{il}
    term3 = jnp.einsum("jkl,ilm->ijkm", gamma_val, gamma_val)

    # Term 4: -\Gamma^l_{ik}\Gamma^m_{jl}
    term4 = -jnp.einsum("ikl,jlm->ijkm", gamma_val, gamma_val)

    # Term 5: -c^l_{ij}\Gamma^m_{lk}.
    # The (i,j,l) index passed in by lie_bracket_pairs corresponds to c^l_{ij}
    term5 = -jnp.einsum("ijl,lkm->ijkm", c_val, gamma_val)
    return term1 + term2 + term3 + term4 + term5

  E_val = basis.components.get_value_jet()
  gamma_val = gamma.get_value_jet()
  gamma_grad = gamma.get_gradient_jet()
  c_val = lie_bracket_pairs.components.get_value_jet()
  riemann_components: Jet = get_components(E_val, gamma_val, gamma_grad, c_val)

  return RiemannCurvatureTensor(
    tensor_type=TensorType(k=3, l=1),
    basis=basis,
    components=riemann_components
  )


class RicciTensor(Tensor):
  """
  (2, 0) Ricci curvature tensor.
  """
  tensor_type: TensorType = eqx.field(static=True) # Always TensorType(k=2, l=0)
  basis: BasisVectors
  components: Annotated[Jet, "D D"] # The components of the tensor written in the chosen basis

  @property
  def batch_size(self):
    return self.basis.batch_size

  def __check_init__(self):
    super().__check_init__()
    if self.components.shape[-2] != self.components.shape[-1]:
      raise ValueError(f"Ricci tensor must be a 2-index tensor")

    expected_batch_shape = self.basis.p.shape[:-1]
    actual_batch_shape = self.components.shape[:-2]
    if expected_batch_shape != actual_batch_shape:
        raise ValueError(
            f"Batch shape mismatch: basis implies {expected_batch_shape} but components have {actual_batch_shape}"
        )

def get_ricci_tensor(connection: Connection, R: Optional[RiemannCurvatureTensor] = None) -> RicciTensor:
  """
  Get the (2, 0) Ricci curvature tensor from a connection.
  """
  if R is None:
    R: RiemannCurvatureTensor = get_riemann_curvature_tensor(connection)

  @jet_decorator
  def get_ricci_components(R_val) -> Array:
    return jnp.einsum("iabi->ab", R_val)

  ricci_components: Jet = get_ricci_components(R.components.get_value_jet())

  return RicciTensor(
    tensor_type=TensorType(k=2, l=0),
    basis=connection.basis,
    components=ricci_components
  )
