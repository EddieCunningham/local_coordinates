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

class TangentVector(AbstractBatchableObject):
  """
  A tangent vector is a vector in a tangent space. The vector is written
  in components of a basis.
  """
  p: Float[Array, "N"]
  components: Annotated[Jet, "N"]
  basis: BasisVectors

  def __check_init__(self):
    assert isinstance(self.components, Jet), "components must be a Jet"
    if self.components.ndim != self.p.ndim:
      raise ValueError(f"Invalid number of dimensions: {self.components.ndim}")

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.basis.batch_size

  def to_standard_basis(self) -> 'TangentVector':
    """
    Transform the tangent vector to the standard basis.
    """
    standard_basis: BasisVectors = get_standard_basis(self.p)
    return change_basis(self, standard_basis)

  def __call__(self, f: Jet) -> Jet:
    """
    Evaluate the tangent vector at a point.
    """
    assert isinstance(f, Jet), "f must be a Jet"
    assert self.batch_size is None, "TangentVector must be unbatched to be evaluated at a point"

    standard_vector: TangentVector = self.to_standard_basis()

    @jet_decorator
    def derivation(component_val: Array, f_grad: Array) -> Array:
      return jnp.einsum("i,...i->...", component_val, f_grad)

    return derivation(standard_vector.components.get_value_jet(), f.get_gradient_jet())

@dispatch
def change_basis(vector: TangentVector, new_basis: BasisVectors) -> TangentVector:
  """
  Transform a tangent vector from one basis to another.
  """
  # Compute linear transform from the current basis to the new basis.
  T: Jet = get_basis_transform(vector.basis, new_basis)

  # Apply the contravariant transform to the vector components
  new_components = apply_contravariant_transform(T, vector.components)
  return TangentVector(p=vector.p, components=new_components, basis=new_basis)
