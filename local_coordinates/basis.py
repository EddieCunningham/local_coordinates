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
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.p.ndim > 2:
      return self.p.shape[:-1]
    elif self.p.ndim == 2:
      return self.p.shape[0]
    elif self.p.ndim == 1:
      return None
    else:
      raise ValueError(f"Invalid number of dimensions: {self.p.ndim}")


class DualBasis(AbstractBatchableObject):
  """
  A basis of covectors (dual basis). Components are written in standard
  Euclidean coordinates, arranged so that θ @ E = I where E are the
  associated basis vectors in the same basis.
  """
  p: Float[Array, "N"]
  components: Annotated[Jet, "N D"]

  def __check_init__(self):
    assert isinstance(self.components, Jet), "components must be a Jet"
    if self.components.ndim != self.p.ndim + 1:
      raise ValueError(f"Invalid number of dimensions: {self.components.ndim}")

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

def get_lie_bracket_components(basis: BasisVectors) -> Jet:
  """
  Get the components of the Lie bracket of the basis vectors.  Returns
  c_{ij}^k where [E_i, E_j] = c_{ij}^k E_k.
  """
  @jet_decorator
  def get_components(basis_vals: Array, basis_grads: Array) -> Array:
    term1 = jnp.einsum("ai,kja->kij", basis_vals, basis_grads)
    term2 = jnp.einsum("aj,kia->kij", basis_vals, basis_grads)
    return term1 - term2

  basis_vals = basis.components.get_value_jet()
  basis_grads = basis.components.get_gradient_jet()
  components_euclidean: Jet = get_components(basis_vals, basis_grads)

  # Convert to components in the same basis that we started with
  standard_basis = get_standard_basis(basis.p)
  T_jet: Jet = get_basis_transform(standard_basis, basis)

  @jet_decorator
  def change_basis(components_euclidean_vals: Array, T_val: Array) -> Array:
    return jnp.einsum("ka,aij->kij", T_val, components_euclidean_vals)

  out: Jet = change_basis(components_euclidean.get_value_jet(), T_jet.get_value_jet())
  return out

@dispatch
def get_basis_transform(from_basis: BasisVectors, to_basis: BasisVectors) -> Jet:
  """
  Get the transformation matrix from one set of basis vectors to another.
  """
  @jet_decorator
  def get_components(from_components, to_components) -> Array:
    return jnp.linalg.solve(to_components, from_components)

  from_components_val = from_basis.components.get_value_jet()
  to_components_val = to_basis.components.get_value_jet()
  new_components = get_components(
    from_components_val,
    to_components_val
  )
  return new_components

@dispatch
def get_basis_transform(from_basis: DualBasis, to_basis: DualBasis) -> Jet:
  """
  Get the transformation matrix for dual bases (covectors).

  If T_vec = inv(E_to) @ E_from is the vector transform, then the dual
  transform satisfies T_dual = T_vec^{-1} = inv(E_from) @ E_to. In terms of
  dual components Θ = inv(E), this is Θ_from @ inv(Θ_to).
  """
  @jet_decorator
  def get_components(theta_from, theta_to) -> Array:
    return theta_from @ jnp.linalg.inv(theta_to)

  from_components_val = from_basis.components.get_value_jet()
  to_components_val = to_basis.components.get_value_jet()
  new_components = get_components(from_components_val, to_components_val)
  return new_components

def get_standard_basis(p: Float[Array, "N"]) -> BasisVectors:
  """
  Get the standard basis at a given point.
  """
  return BasisVectors(p=p, components=Jet(value=jnp.eye(p.shape[0]), gradient=jnp.zeros((p.shape[0], p.shape[0], p.shape[0])), hessian=jnp.zeros((p.shape[0], p.shape[0], p.shape[0], p.shape[0]))))

def get_standard_dual_basis(p: Float[Array, "N"]) -> DualBasis:
  """
  Get the standard dual basis at a given point.
  """
  return DualBasis(p=p, components=Jet(value=jnp.eye(p.shape[0]), gradient=jnp.zeros((p.shape[0], p.shape[0], p.shape[0])), hessian=jnp.zeros((p.shape[0], p.shape[0], p.shape[0], p.shape[0]))))

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
