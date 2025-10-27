from typing import Any, Callable, Tuple, Annotated, Optional, List, Union
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from linsdex import AbstractBatchableObject, auto_vmap
from functools import partial
from plum import dispatch
from local_coordinates.jet import Jet, jet_decorator
import warnings

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

@dispatch
def get_basis_transform(from_basis: BasisVectors, to_basis: BasisVectors) -> Jet:
  """
  Get the transformation matrix from one set of basis vectors to another.

  from_basis = (E_1, \dots, E_n) where E_j = E_j^i d/dx^i
  to_basis = (B_1, \dots, B_n) where B_j = B_j^i d/dx^i

  Suppose we have a vector V = V^j E_j = W^i B_i.  This function returns
  the linear map T_i^j that satisfies W^i = T_i^j V^j.

  V = V^j E_j = V^j E_j^i d/dx^i
  W = W^k B_k = W^k B_k^i d/dx^i

  This gives us W^k = (B^{-1})^k_i E^i_j V^j which implies

  --> T_j^k = (B^{-1})^k_i E^i_j
  """
  assert isinstance(from_basis, BasisVectors), f"from_basis must be a BasisVectors, got {type(from_basis)}"

  @jet_decorator
  def get_components(from_components, to_components) -> Array:
    return jnp.linalg.solve(to_components, from_components)

  from_components_val: Jet = from_basis.components.get_value_jet()
  to_components_val: Jet = to_basis.components.get_value_jet()
  new_components: Jet = get_components(
    from_components_val,
    to_components_val
  )
  return new_components

def apply_covariant_transform(T: Jet, old_basis_components: Jet) -> Jet:
  """
  Apply a covariant transform to a set of components.
  """
  @jet_decorator
  def apply_transform(T_val: Array, x_components: Array) -> Array:
    return jnp.vectorize(jnp.linalg.solve, signature="(n,n),(n)->(n)")(T_val.mT, x_components)

  new_basis_components: Jet = apply_transform(T.get_value_jet(), old_basis_components.get_value_jet())
  return new_basis_components

def apply_contravariant_transform(T: Jet, old_components: Jet) -> Jet:
  """
  Apply a contravariant transform to a set of components.
  """
  @jet_decorator
  def apply_transform(T_val: Array, x_components: Array) -> Array:
    return jnp.einsum("ij,...j->...i", T_val, x_components)

  new_components: Jet = apply_transform(T.get_value_jet(), old_components.get_value_jet())
  return new_components

@dispatch
def get_dual_basis_transform(from_basis: BasisVectors, to_basis: BasisVectors) -> Jet:
  """
  Get the transformation matrix acting on dual components induced by vector bases.

  If E_from, E_to are vector-basis component matrices, the vector transform is
    T_vec = inv(E_to) @ E_from.
  The induced dual transform is
    T_dual = (T_vec)^{-1} = inv(E_from) @ E_to.
  """
  @jet_decorator
  def get_components(theta_from, theta_to) -> Array:
    return jnp.linalg.solve(theta_from, theta_to)

  from_components_val = from_basis.components.get_value_jet()
  to_components_val = to_basis.components.get_value_jet()
  new_components = get_components(from_components_val, to_components_val)
  return new_components

def get_standard_basis(p: Float[Array, "N"]) -> BasisVectors:
  """
  Get the standard basis at a given point.
  """
  return BasisVectors(p=p, components=Jet(value=jnp.eye(p.shape[0]), gradient=jnp.zeros((p.shape[0], p.shape[0], p.shape[0])), hessian=jnp.zeros((p.shape[0], p.shape[0], p.shape[0], p.shape[0]))))

def get_standard_dual_basis(p: Float[Array, "N"]) -> BasisVectors:
  """
  Get the standard dual basis (identity covectors) at a given point, represented
  using BasisVectors whose components equal the identity.
  """
  return BasisVectors(p=p, components=Jet(value=jnp.eye(p.shape[0]), gradient=jnp.zeros((p.shape[0], p.shape[0], p.shape[0])), hessian=jnp.zeros((p.shape[0], p.shape[0], p.shape[0], p.shape[0]))))

@dispatch
def change_coordinates(basis: BasisVectors, x_to_z: Callable[[Array], Array], x: Array) -> BasisVectors:
  """
  Change the coordinates of a basis vectors from one set of coordinates to another.
  """
  if basis.components.hessian is not None:
    warnings.warn("The Hessian transform is not implemented for change_coordinates of BasisVectors.")
  x_jet = Jet(value=x, gradient=basis.components.value, hessian=basis.components.gradient)
  z_jet: Jet = change_coordinates(x_jet, x_to_z, x)
  z = x_to_z(x)
  z_components = Jet(value=z_jet.gradient, gradient=z_jet.hessian, hessian=None)
  return BasisVectors(p=z, components=z_components)

@dispatch.abstract
def change_basis(obj: Any, target_basis: BasisVectors) -> Any:
  """
  Change the basis of an object to a new basis.
  """
  pass
