from typing import Any, Callable, Tuple, Annotated, Optional, List, Union
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from linsdex import AbstractBatchableObject
from plum import dispatch
from local_coordinates.jet import Jet, jet_decorator
from local_coordinates.basis import BasisVectors, get_dual_basis_transform
from local_coordinates.tensor import Tensor, TensorType

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