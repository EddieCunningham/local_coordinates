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
from local_coordinates.basis import BasisVectors, get_basis_transform, get_standard_basis, apply_contravariant_transform
from local_coordinates.tangent import TangentVector, lie_bracket
from functools import partial

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
    assert jnp.allclose(self.p, self.basis.p), "p and basis.p must be the same"

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.basis.batch_size

  def get_basis_vector(self, i: int) -> TangentVector:
    """
    Get the i-th basis vector.
    """
    return TangentVector(p=self.p, components=self.components[i], basis=self.basis)

  def to_standard_basis(self) -> 'Frame':
    """
    Transform the frame to the standard basis.
    """
    return change_basis(self, get_standard_basis(self.p))

@dispatch
def change_basis(frame: Frame, new_basis: BasisVectors) -> Frame:
  """
  Transform a frame from one basis to another.
  """
  # Compute linear transform from the current basis to the new basis.
  T: Jet = get_basis_transform(frame.basis, new_basis)

  # Apply the contravariant transform to each of the frame components
  new_components: Jet = eqx.filter_vmap(apply_contravariant_transform, in_axes=(None, 0))(T, frame.components)

  return Frame(p=frame.p, components=new_components, basis=new_basis)

@dispatch
def pushforward(frame: Frame, f: Callable) -> Frame:
  """
  Pushforward a frame through a smooth map.
  """
  raise NotImplementedError("It is not possible to pushforward frames whose components are Jets.")

  @eqx.filter_vmap
  def pushforward_basis_vector_components(vector_components: Jet) -> TangentVector:
    X = TangentVector(p=frame.p, components=vector_components, basis=frame.basis)
    return pushforward(X, f)

  new_basis_vectors: Annotated[TangentVector, "D"] = pushforward_basis_vector_components(frame.components)
  basis: BasisVectors = new_basis_vectors.basis[0]
  return Frame(p=basis.p, components=new_basis_vectors.components, basis=basis)

def get_lie_bracket_between_frame_pairs(frame: Frame) -> Annotated[TangentVector, "D D"]:
  """
  Returns a doubly batched TangentVector whose elements are a TangentVector
  representing the Lie bracket between the i-th and j-th basis vectors.
  """

  @partial(eqx.filter_vmap, in_axes=(0, None))
  @partial(eqx.filter_vmap, in_axes=(None, 0))
  def get_lie_bracket(Ei_components: Jet, Ej_components: Jet) -> Jet:
    Ei = TangentVector(p=frame.p, components=Ei_components, basis=frame.basis)
    Ej = TangentVector(p=frame.p, components=Ej_components, basis=frame.basis)
    return lie_bracket(Ei, Ej)

  out: Annotated[TangentVector, "D D"] = get_lie_bracket(frame.components, frame.components)
  return out

def frames_are_equivalent(a: Frame, b: Frame) -> bool:
  """
  Check if two frames are equivalent up to a change of basis.
  """
  a_standard: Frame = a.to_standard_basis()
  b_standard: Frame = b.to_standard_basis()
  return jnp.allclose(a_standard.components.value, b_standard.components.value) and jnp.allclose(a_standard.components.gradient, b_standard.components.gradient) and jnp.allclose(a_standard.components.hessian, b_standard.components.hessian)