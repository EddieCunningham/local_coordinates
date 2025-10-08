from typing import Any, Callable, Tuple, Annotated, Optional, List, Union
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from linsdex import AbstractBatchableObject
from plum import dispatch
from local_coordinates.jet import Jet

class BasisVectors(AbstractBatchableObject):
  """
  A set of basis vectors for a tangent space. The basis vectors are always written
  in the standard basis of Euclidean coordinates.
  """
  _jet: Jet

  @property
  def p(self) -> Array:
    return self._jet.value

  @property
  def basis_vectors(self) -> Array:
    return self._jet.gradient

  @property
  def second_derivatives(self) -> Array:
    return self._jet.hessian

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

def get_basis_transform(from_basis: BasisVectors, to_basis: BasisVectors) -> Callable[[Array], Array]:
  """
  Get the transformation matrix from one set of basis vectors to another.
  """
  inverse_to_basis_vectors = jnp.linalg.inv(to_basis.basis_vectors)
  return inverse_to_basis_vectors @ from_basis.basis_vectors

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
  frame = basis.basis_vectors
  dframe_dx = basis.second_derivatives

  if dframe_dx is None:
    raise ValueError(
      "Cannot make a coordinate basis without second derivatives "
      "(i.e., the hessian of the jet)."
    )

  # Create a new Jet with the original point and frame, but new derivatives.
  new_jet = Jet(value=p, gradient=frame, hessian=0.5*(dframe_dx + jnp.swapaxes(dframe_dx, -2, -1)))

  return BasisVectors(_jet=new_jet)

@dispatch.abstract
def change_basis(obj: Any, target_basis: BasisVectors) -> Any:
  """
  Change the basis of an object to a new basis.
  """
  pass