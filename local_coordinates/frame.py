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
from local_coordinates.basis import BasisVectors, get_basis_transform

class Frame(AbstractBatchableObject):
  """
  A set of basis vectors for a tangent space. The basis vectors are always written
  in the standard basis of Euclidean coordinates.
  """
  p: Float[Array, "N"]
  components: Annotated[Jet, "N D"] # Contains a matrix of Jets, each of which represents a single component
  basis: BasisVectors

  def __check_init__(self):
    assert isinstance(self.components, Jet), "components must be a Jet"
    if self.components.ndim != self.p.ndim + 1:
      raise ValueError(f"Invalid number of dimensions: {self.components.ndim}")

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.basis.batch_size

@dispatch
def change_coordinates(frame: Frame, new_basis: BasisVectors) -> Frame:
  """
  Transform a frame from one basis to another.
  """
  # Compute linear transform from the current basis to the new basis.
  T = get_basis_transform(frame.basis, new_basis)

  # Apply only the value-level linear map to the frame components while
  # propagating derivatives coming from the frame itself.
  @jet_decorator
  def transform_components(components):
    # Left-multiply by T (contravariant index transformation).
    return jnp.einsum("...ij,...jk->...ik", T.value, components)

  comps_val = frame.components.get_value_jet()
  new_components = transform_components(comps_val)

  return Frame(p=frame.p, components=new_components, basis=new_basis)