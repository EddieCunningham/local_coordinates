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
from local_coordinates.basis import BasisVectors, get_basis_transform, get_standard_basis, apply_contravariant_transform, change_coordinates as change_coordinates_basis
from local_coordinates.jet import change_coordinates as change_coordinates_jet
from local_coordinates.jacobian import Jacobian
from local_coordinates.tangent import TangentVector, lie_bracket
from functools import partial
from local_coordinates.jet import get_identity_jet

class Frame(AbstractBatchableObject):
  """
  A set of basis vectors for a tangent space. The basis vectors are always written
  in the standard basis of Euclidean coordinates.

  Indexing Convention:
    Following Jet conventions, derivative indices are trailing. For a coordinate frame
    with Jacobian J^i_j = ∂x^i/∂z^j:

      - components.value[i, j] = J^i_j = i-th component of j-th basis vector
      - components.gradient[i, j, k] = ∂J^i_j/∂z^k = ∂²x^i/∂z^j∂z^k
      - components.hessian[i, j, k, l] = ∂³x^i/∂z^j∂z^k∂z^l

    Columns are basis vectors: components.value[:, j] = E_j (the j-th basis vector)
  """
  p: Float[Array, "N"]
  components: Annotated[Jet, "N D"] # Contains a matrix of Jets, columns are basis vectors
  basis: BasisVectors

  def __check_init__(self):
    assert isinstance(self.components, Jet), "components must be a Jet"
    if self.components.ndim != self.p.ndim + 1:
      raise ValueError(f"Invalid number of dimensions: {self.components.ndim}")
    # assert jnp.allclose(self.p, self.basis.p), "p and basis.p must be the same"

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.basis.batch_size

  def get_basis_vector(self, j: int) -> TangentVector:
    """
    Get the j-th basis vector.

    Since columns are basis vectors, we extract the j-th column:
      - value[:, j] = E_j (j-th basis vector)
      - gradient[:, j, :] = ∂E_j/∂x
      - hessian[:, j, :, :] = ∂²E_j/∂x²
    """
    value = self.components.value[:, j]
    gradient = self.components.gradient[:, j, :] if self.components.gradient is not None else None
    hessian = self.components.hessian[:, j, :, :] if self.components.hessian is not None else None
    dim = self.p.shape[0]
    components_jet = Jet(value=value, gradient=gradient, hessian=hessian, dim=dim)
    return TangentVector(p=self.p, components=components_jet, basis=self.basis)

  def to_standard_basis(self) -> 'Frame':
    """
    Transform the frame to the standard basis.
    """
    return change_basis(self, get_standard_basis(self.p))

def basis_to_frame(basis: BasisVectors) -> Frame:
  """
  Create a frame from a basis.
  """
  return Frame(p=basis.p, components=get_identity_jet(basis.p.shape[0]), basis=basis)

@dispatch
def change_basis(frame: Frame, new_basis: BasisVectors) -> Frame:
  """
  Transform a frame from one basis to another.

  Since columns are basis vectors, we vmap over axis 1 (columns) and stack on axis 1.
  """
  # Compute linear transform from the current basis to the new basis.
  T: Jet = get_basis_transform(frame.basis, new_basis)

  # Apply the contravariant transform to each column (basis vector)
  # in_axes=1 extracts columns, out_axes=1 stacks them back as columns
  new_components: Jet = eqx.filter_vmap(
    apply_contravariant_transform, in_axes=(None, 1), out_axes=1
  )(T, frame.components)

  return Frame(p=frame.p, components=new_components, basis=new_basis)

@dispatch
def change_coordinates(frame: Frame, x_to_z_jacobian: Jacobian) -> Frame:
  """
  Change coordinates for a Frame using a precomputed Jacobian.
  """
  new_basis = change_coordinates_basis(frame.basis, x_to_z_jacobian) # Covariant transform
  new_components = change_coordinates_jet(frame.components, x_to_z_jacobian) # Contravariant transform

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

  Since columns are basis vectors, we vmap over axis 1 (columns).
  """

  @partial(eqx.filter_vmap, in_axes=(1, None))  # outer: iterate i over columns
  @partial(eqx.filter_vmap, in_axes=(None, 1))  # inner: iterate j over columns
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
