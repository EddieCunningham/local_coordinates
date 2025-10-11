from typing import Any, Callable, Tuple, Annotated, Optional, List, Union
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from linsdex import AbstractBatchableObject
from plum import dispatch
from local_coordinates.jet import Jet, jet_decorator, _expand_jet

class BasisVectors(AbstractBatchableObject):
  """
  A set of basis vectors for a tangent space. The basis vectors are always written
  in the standard basis of Euclidean coordinates.
  """
  p: Float[Array, "N"]
  components: Annotated[Jet, "N D"] # Contains a matrix of Jets, each of which represents a single component

  def __check_init__(self):
    assert isinstance(self.components, Jet), "components must be a Jet"
    if self.components.ndim != self.p.ndim + 1:
      raise ValueError(f"Invalid number of dimensions: {self.components.ndim}")

  @property
  def basis_vectors(self) -> Array:
    return self.components.value

  @property
  def second_derivatives(self) -> Optional[Array]:
    return self.components.gradient

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.p.ndim > 2:
      return self.p.shape[:-1]
    elif self.p.ndim == 2:
      return self.p.shape[0]
    elif self.p.ndim == 1:
      return None
    else:
      raise ValueError(f"Invalid number of dimensions: {self.p.ndim}")

def get_basis_transform(from_basis: BasisVectors, to_basis: BasisVectors) -> Jet:
  """
  Get the transformation matrix from one set of basis vectors to another.
  """
  @jet_decorator
  def get_components(from_components, to_components) -> Array:
    return jnp.linalg.solve(to_components, from_components)

  from_components_val, _, _ = _expand_jet(from_basis.components)
  to_components_val, _, _ = _expand_jet(to_basis.components)
  new_components = get_components(
    from_components_val,
    to_components_val
  )
  return new_components

def make_coordinate_basis(basis: BasisVectors) -> BasisVectors:
  """
  Make a commuting frame (a coordinate basis) from a given basis.

  This function takes a set of basis vectors (a frame) and their derivatives,
  and returns a new BasisVectors object representing a commuting frame. It
  does this by enforcing the Frobenius integrability condition, [E_j, E_k] = 0,
  which is equivalent to symmetrizing the derivatives of the frame vectors
  when expressed in the frame's own basis.

  The frame vectors themselves are not changed, only their derivatives are
  projected onto the symmetric part.
  """
  p = basis.p
  frame = basis.components.value
  dframe_dx = basis.components.gradient

  if dframe_dx is None:
    raise ValueError(
      "Cannot make a coordinate basis without second derivatives "
      "(i.e., the hessian of the jet)."
    )

  # Create a new Jet with the original point and frame, but new derivatives.
  new_jet = Jet(value=frame, gradient=0.5*(dframe_dx + jnp.swapaxes(dframe_dx, -2, -1)), hessian=None)

  return BasisVectors(p=p, components=new_jet)

@dispatch.abstract
def change_basis(obj: Any, target_basis: BasisVectors) -> Any:
  """
  Change the basis of an object to a new basis.
  """
  pass


def get_mixing_function(basis: BasisVectors) -> Callable[[Float[Array, "D"]], Float[Array, "N"]]:
    """
    Creates a callable mixing function from a BasisVectors object.

    The returned function implements the second-order Taylor approximation of the
    coordinate system defined by the basis, transforming a displacement vector `dx` from
    the local tangent space to a point in the ambient space.
    """
    # The Jet's value is the base point `p`. The first derivative (gradient)
    # is given by the basis vectors themselves, and the second derivative (hessian)
    # is given by the gradient of the basis vector components.
    transform_jet = Jet(
        value=basis.p,
        gradient=basis.components.value,
        hessian=basis.components.gradient
    )

    def mixing_function(dx: Float[Array, "D"]) -> Float[Array, "N"]:
        return transform_jet(dx)

    return mixing_function