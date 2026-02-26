from typing import Any, Callable, Tuple, Annotated, Optional, List, Union
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from linsdex import AbstractBatchableObject
from plum import dispatch
from local_coordinates.jet import Jet, jet_decorator, function_to_jet
from local_coordinates.basis import BasisVectors, get_dual_basis_transform, get_standard_basis
from local_coordinates.tensor import Tensor, TensorType
from local_coordinates.tangent import TangentVector, change_basis
from local_coordinates.jacobian import function_to_jacobian
from local_coordinates.jet import get_identity_jet

class RiemannianMetric(Tensor):
  """
  A Riemannian metric is a map from a tangent space to the real numbers.
  """
  tensor_type: TensorType = eqx.field(static=True)
  basis: BasisVectors
  components: Annotated[Jet, "D D"] # The components of the tensor written in the chosen basis

  def __init__(self, basis: BasisVectors, components: Annotated[Jet, "D D"], **kwargs):
    super().__init__(tensor_type=TensorType(k=2, l=0), basis=basis, components=components)

  @property
  def batch_size(self):
    return self.basis.batch_size

  def __check_init__(self):
    super().__check_init__()
    if self.components.shape[-2] != self.components.shape[-1]:
      raise ValueError(f"Metric must be a square matrix")

    expected_batch_shape = self.basis.p.shape[:-1]
    actual_batch_shape = self.components.shape[:-2]
    if expected_batch_shape != actual_batch_shape:
      raise ValueError(
        f"Batch shape mismatch: basis implies {expected_batch_shape} but components have {actual_batch_shape}"
      )

  def __call__(self, X: TangentVector, Y: TangentVector) -> Jet:
    """
    Evaluate the metric at two tangent vectors.
    """
    X: TangentVector = change_basis(X, self.basis)
    Y: TangentVector = change_basis(Y, self.basis)
    @jet_decorator
    def components(X_val, Y_val, g_val):
      return jnp.einsum("i,ij,j->...", X_val, g_val, Y_val)
    return components(X.components.get_value_jet(), Y.components.get_value_jet(), self.components.get_value_jet())

def raise_index(tensor: Tensor, metric: RiemannianMetric, index: int) -> Tensor:
  """
  Raise one covariant index of a tensor using the metric.

  Index semantics:
  - index is the overall 1-based index position in the tensor (covariant first, then contravariant).
  - For a tensor with type (k, l), valid range is 1 <= index <= k + l, and it must satisfy index <= k (i.e., refer to a covariant slot).
  """
  # if tensor.basis != metric.basis:
  #   raise ValueError("Tensor and metric must be expressed in the same basis to raise an index.")

  k = tensor.tensor_type.k
  l = tensor.tensor_type.l

  # Validate overall 1-based index semantics
  if not (1 <= index <= k + l):
    raise ValueError(f"Index to raise must be overall 1-based in [1, {k + l}], got {index}.")
  if index > k:
    raise ValueError(
      f"Index {index} refers to a contravariant position; only covariant indices can be raised."
    )

  # Convert to 0-based position within the covariant block
  covar_pos = index - 1

  # Build einsum indices
  alphabet = "ijklmnopqrstuvwxyzabcdefgh"
  if k + l + 2 > len(alphabet):
    raise ValueError(f"Tensor has too many dimensions ({k + l}) to be handled.")

  covar_labels = list(alphabet[:k])
  contra_labels = list(alphabet[k:k + l])

  # Label for the raised (new) contravariant index (choose an unused label)
  new_contra_label = alphabet[k + l]

  # The label of the covariant index we are raising (to be contracted)
  contracted_label = covar_labels[covar_pos]

  # Input tensor indices
  input_indices = ''.join(covar_labels + contra_labels)

  # Metric inverse indices: new free contravariant index and contracted label
  # g^{ab} with indices (new_contra_label, contracted_label)
  g_inv_indices = f"{new_contra_label}{contracted_label}"

  # Output indices: covariant (excluding the raised one), followed by existing
  # contravariant indices, followed by the new contravariant index
  output_covar = [lbl for i, lbl in enumerate(covar_labels) if i != covar_pos]
  output_contra = contra_labels + [new_contra_label]
  output_indices = ''.join(output_covar + output_contra)

  einsum_str = f"...{input_indices},...{g_inv_indices}->...{output_indices}"

  # Compute inverse metric as a Jet
  g_inv: Jet = jet_decorator(jnp.linalg.inv)(metric.components.get_value_jet())

  @jet_decorator
  def apply_raise(components, ginv):
    return jnp.einsum(einsum_str, components, ginv)

  new_components = apply_raise(tensor.components.get_value_jet(), g_inv.get_value_jet())

  return Tensor(
    tensor_type=TensorType(k=k - 1, l=l + 1),
    basis=tensor.basis,
    components=new_components,
  )

def lower_index(tensor: Tensor, metric: RiemannianMetric, index: int) -> Tensor:
  """
  Lower one contravariant index of a tensor using the metric.

  Index semantics:
  - index is the overall 1-based index position in the tensor (covariant first, then contravariant).
  - For a tensor with type (k, l), valid range is 1 <= index <= k + l, and it must satisfy index > k (i.e., refer to a contravariant slot).

  Here k is the number of covariant indices and l is the number of contravariant indices.
  """
  # if tensor.basis != metric.basis:
  #   raise ValueError("Tensor and metric must be expressed in the same basis to lower an index.")

  k = tensor.tensor_type.k
  l = tensor.tensor_type.l

  # Validate overall 1-based index semantics
  if not (1 <= index <= k + l):
    raise ValueError(f"Index to lower must be overall 1-based in [1, {k + l}], got {index}.")
  if index <= k:
    raise ValueError(
      f"Index {index} refers to a covariant position; only contravariant indices can be lowered."
    )

  # Convert to 0-based position within the contravariant block
  contra_pos = index - k - 1

  # Build einsum indices
  alphabet = "ijklmnopqrstuvwxyzabcdefgh"
  if k + l + 2 > len(alphabet):
    raise ValueError(f"Tensor has too many dimensions ({k + l}) to be handled.")

  covar_labels = list(alphabet[:k])
  contra_labels = list(alphabet[k:k + l])

  # Label for the lowered (new) covariant index (choose an unused label)
  new_covar_label = alphabet[k + l]

  # The label of the contravariant index we are lowering (to be contracted)
  contracted_label = contra_labels[contra_pos]

  # Input tensor indices
  input_indices = ''.join(covar_labels + contra_labels)

  # Metric indices: contracted label and new free covariant label
  # g_{ab} with indices (contracted_label, new_covar_label)
  g_indices = f"{contracted_label}{new_covar_label}"

  # Output indices: covariant (existing + new), followed by contravariant excluding lowered
  output_covar = covar_labels + [new_covar_label]
  output_contra = [lbl for i, lbl in enumerate(contra_labels) if i != contra_pos]
  output_indices = ''.join(output_covar + output_contra)

  einsum_str = f"...{input_indices},...{g_indices}->...{output_indices}"

  g = metric.components.get_value_jet()

  @jet_decorator
  def apply_lower(components, g_vals):
    return jnp.einsum(einsum_str, components, g_vals)

  new_components = apply_lower(tensor.components.get_value_jet(), g)

  return Tensor(
    tensor_type=TensorType(k=k + 1, l=l - 1),
    basis=tensor.basis,
    components=new_components,
  )

def get_euclidean_metric(p: Array) -> RiemannianMetric:
  """
  Get the Euclidean metric on R^p.shape[0].
  """
  return RiemannianMetric(
    basis=get_standard_basis(p),
    components=get_identity_jet(p.shape[0])
  )

def _compose_jet_with_jacobian(jet: Jet, jacobian) -> Jet:
  """Re-express a Jet's coordinate derivatives through a (possibly non-square) Jacobian.

  Given J[g]_y (a Jet with derivatives w.r.t. y-coordinates) and the Jacobian
  of a map f with y = f(x), return J[g circ f]_x (a Jet with derivatives
  w.r.t. x-coordinates) via the multivariate chain rule.

  The Jacobian need not be square, so this works for maps f: M -> N where
  dim(M) != dim(N).
  """
  g_val = jet.value
  g_grad = jet.gradient
  g_hess = jet.hessian

  J = jacobian.value
  H = jacobian.gradient

  if g_grad is None:
    return Jet(value=g_val, gradient=None, hessian=None)

  composed_grad = jnp.einsum("...c,ck->...k", g_grad, J)

  if g_hess is None:
    return Jet(value=g_val, gradient=composed_grad, hessian=None)

  composed_hess = jnp.einsum("...cd,ck,dl->...kl", g_hess, J, J)
  if H is not None:
    composed_hess = composed_hess + jnp.einsum("...c,ckl->...kl", g_grad, H)

  return Jet(value=g_val, gradient=composed_grad, hessian=composed_hess)


def pullback_metric(
  x: Array,
  f: Callable[[Array], Array],
  g: RiemannianMetric,
) -> RiemannianMetric:
  """
  Compute the pullback metric f^* g of a metric g on N under a map f: M -> N.

  The pullback metric g_f = f^* g is defined by
    (g_f)_ij(x) = (df^a/dx^i) g_ab(f(x)) (df^b/dx^j)

  The map f need not be dimension-preserving. When dim(M) != dim(N) the
  Jacobian df is rectangular and the resulting metric lives on M.

  Args:
    x: The point at which to evaluate the pullback metric.
    f: The map from M to N.
    g: The Riemannian metric g on N.

  Returns:
    The pullback metric g_f on M.
  """
  # Go to the standard basis
  g: RiemannianMetric = change_basis(g, get_standard_basis(g.basis.p))

  # Compute the Jacobian of f at x (in standard coordinates)
  G_jac = function_to_jacobian(f, x)
  G = Jet(value=G_jac.value, gradient=G_jac.gradient, hessian=G_jac.hessian)

  # Compose g's N-coordinate derivatives with the Jacobian of f so that both
  # G and g_composed carry M-coordinate derivatives (required by jet_decorator).
  g_composed = _compose_jet_with_jacobian(g.components, G_jac)

  @jet_decorator
  def compute_pullback(G_val, g_val):
    return jnp.einsum("ai,bj,ab->ij", G_val, G_val, g_val)
  gf_components: Jet = compute_pullback(G, g_composed)

  standard_basis = get_standard_basis(x)
  return RiemannianMetric(basis=standard_basis, components=gf_components)
