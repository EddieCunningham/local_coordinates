import string
from typing import Any, Callable, Tuple, Annotated, Optional, List, Union
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from linsdex import AbstractBatchableObject
from local_coordinates.basis import BasisVectors, get_basis_transform
from plum import dispatch
from local_coordinates.jet import Jet, jet_decorator
import warnings

class TensorType(eqx.Module):
  k: int # Covariant index
  l: int # Contravariant index

  def total_dims(self):
    return self.k + self.l

  def __add__(self, other: 'TensorType'):
    return TensorType(self.k + other.k, self.l + other.l)

  def is_covector(self):
    return self == TensorType(0, 1)

  def is_vector(self):
    return self == TensorType(1, 0)

  @property
  def k_names(self) -> List[str]:
    """Get the indices for the coordinates that we'll plug into
    einsum when evaluating the tensor.

    Returns:
      If this has 3 contravariant indices, then returns 'k0 k1 k2'
    """
    return [f'k{k}' for k in range(self.k)]

  @property
  def l_names(self) -> List[str]:
    """Get the indices for the coordinates that we'll plug into
    einsum when evaluating the tensor.

    Returns:
      If this has 3 covariant indices, then returns 'l0 l1 l2'
    """
    return [f'l{l}' for l in range(self.l)]

  def get_coordinate_indices(self) -> str:
    """Get the indices for the coordinates that we'll plug into
    einsum when evaluating the tensor.

    Returns:
      A (3, 5) tensor will have 'k0 k1 k2 l0 l1 l2 l3 l4'
    """
    if len(self.k_names) == 0:
      return ' '.join(self.l_names)
    if len(self.l_names) == 0:
      return ' '.join(self.k_names)
    return ' '.join(self.k_names) + ' ' + ' '.join(self.l_names)

################################################################################################################

class Tensor(AbstractBatchableObject):
  """
  A Tensor is a collection of arrays, each representing a different tensor.
  """
  tensor_type: TensorType = eqx.field(static=True)
  basis: BasisVectors
  components: Annotated[Jet, "... D"] # The components of the tensor written in the chosen basis

  def __check_init__(self):
    assert isinstance(self.components, Jet), "components must be a Jet"
    if self.components.ndim < self.tensor_type.total_dims():
      raise ValueError(f"Invalid number of dimensions: {self.components.ndim}")

  @property
  def p(self) -> Array:
    return self.basis.p

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    total_dim = self.tensor_type.total_dims()
    if self.components.ndim == total_dim:
      return None
    elif self.components.ndim == total_dim + 1:
      return self.components.shape[0]
    elif self.components.ndim > total_dim + 1:
      return self.components.shape[:-total_dim]
    else:
      raise ValueError(f"Invalid number of dimensions: {self.components.ndim}")

  def __add__(self, other: 'Tensor') -> 'Tensor':
    @jet_decorator
    def add_components(t_comps, other_comps):
      return t_comps + other_comps
    t_comps_val = self.components.get_value_jet()
    other_comps_val = other.components.get_value_jet()
    new_components = add_components(t_comps_val, other_comps_val)
    return Tensor(tensor_type=self.tensor_type, basis=self.basis, components=new_components)

def function_multiply_tensor(T: Tensor, f: Jet) -> Tensor:
  @jet_decorator
  def multiply_components(t_component_values, f_values):
      return t_component_values*f_values

  t_comps_vals = T.components.get_value_jet()
  f_vals = f.get_value_jet()
  new_components = multiply_components(t_comps_vals, f_vals)
  return Tensor(tensor_type=T.tensor_type, basis=T.basis, components=new_components)

@dispatch
def change_basis(tensor: Tensor, new_basis: BasisVectors) -> Tensor:
  """
  Transform a tensor from one basis to another.
  """
  T: Jet = get_basis_transform(tensor.basis, new_basis)
  Tinv: Jet = jet_decorator(jnp.linalg.inv)(T.get_value_jet())

  T_value_jet: Jet = T.get_value_jet()
  Tinv_value_jet: Jet = Tinv.get_value_jet()

  k = tensor.tensor_type.k
  l = tensor.tensor_type.l

  if tensor.components.ndim < k + l:
    raise ValueError("Tensor components have fewer dimensions than tensor type requires.")

  # Use ... for batch dimensions, and letters for tensor indices.
  # Using an explicit set of characters for indices to avoid collisions.
  alphabet = "ijklmnopqrstuvwxyzabcdefgh"

  # Check if we have enough unique characters for indices
  if k + l > len(alphabet):
    raise ValueError(f"Tensor has too many dimensions ({k+l}) to be handled.")

  input_indices = alphabet[:k+l]
  output_indices = alphabet[k+l:2*(k+l)]

  input_tensor_str = f"...{input_indices}"

  transforms = []
  transforms_str_parts = []

  # Covariant part (k) transforms with Tinv
  k_input_indices = input_indices[:k]
  k_output_indices = output_indices[:k]
  for i in range(k):
    transforms_str_parts.append(f"{k_output_indices[i]}{k_input_indices[i]}")
    transforms.append(Tinv_value_jet)

  # Contravariant part (l) transforms with T
  l_input_indices = input_indices[k:k+l]
  l_output_indices = output_indices[k:k+l]
  for i in range(l):
    transforms_str_parts.append(f"{l_output_indices[i]}{l_input_indices[i]}")
    transforms.append(T_value_jet)

  output_tensor_str = f"...{output_indices}"

  einsum_str = f"{input_tensor_str},{','.join(transforms_str_parts)}->{output_tensor_str}"

  @jet_decorator
  def transform_components(components, *transforms_vals) -> Array:
    return jnp.einsum(einsum_str, components, *transforms_vals)

  t_comps_val = tensor.components.get_value_jet()
  new_components: Jet = transform_components(t_comps_val, *transforms)

  tensor_class = type(tensor) # e.g. RiemannianMetric
  return tensor_class(
    tensor_type=tensor.tensor_type,
    basis=new_basis,
    components=new_components
  )