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
from local_coordinates.basis import BasisVectors, get_basis_transform, get_standard_basis

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

  # DualFrame functionality removed

@dispatch
def change_basis(frame: Frame, new_basis: BasisVectors) -> Frame:
  """
  Transform a frame from one basis to another.
  """
  # Compute linear transform from the current basis to the new basis.
  T: Jet = get_basis_transform(frame.basis, new_basis)

  # Apply full chain rule: differentiate through both T and the frame components.
  @jet_decorator
  def transform_components(T_in, components_in):
    # Left-multiply by T (contravariant index transformation).
    return jnp.einsum("...ij,...jk->...ik", T_in, components_in)

  new_components = transform_components(T, frame.components)
  return Frame(p=frame.p, components=new_components, basis=new_basis)

def get_lie_bracket_components(frame: Frame) -> Jet:
  """
  Get the components of the Lie bracket of the frame.  Returns
  c_{ij}^k where [E_i, E_j] = c_{ij}^k E_k.
  """
  standard_basis: BasisVectors = get_standard_basis(frame.p)

  # Go to the standard basis
  frame_standard: Frame = change_basis(frame, standard_basis)
  standard_components: Jet = frame_standard.components

  @jet_decorator
  def get_components(basis_vals: Array, basis_grads: Array) -> Array:
    term1 = jnp.einsum("ai,kja->kij", basis_vals, basis_grads)
    term2 = jnp.einsum("aj,kia->kij", basis_vals, basis_grads)
    return term1 - term2

  basis_vals: Jet = standard_components.get_value_jet()
  basis_grads: Jet = standard_components.get_gradient_jet()
  components_euclidean: Jet = get_components(basis_vals, basis_grads)

  # Convert to components in the same basis that we started with
  T_jet: Annotated[Jet, "N D D"] = get_basis_transform(standard_basis, frame.basis)

  @jet_decorator
  def transform_back(components_euclidean_vals: Array, T_val: Array) -> Array:
    return jnp.einsum("ka,aij->kij", T_val, components_euclidean_vals)

  out: Jet = transform_back(components_euclidean.get_value_jet(), T_jet.get_value_jet())
  return out

## DualFrame class and change_basis for DualFrame removed per deprecation