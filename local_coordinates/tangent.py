from typing import Any, Callable, Tuple, Annotated, Optional, List, Union
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from linsdex import AbstractBatchableObject
from plum import dispatch
from local_coordinates.jet import Jet, jet_decorator, function_to_jet, change_coordinates
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
    # assert jnp.allclose(self.p, self.basis.p), "p and basis.p must be the same"

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

  def __add__(self, other: 'TangentVector') -> 'TangentVector':
    """
    Add two tangent vectors.
    """
    assert self.batch_size is None and other.batch_size is None, "Use vmap to add batched tangent vectors"
    p = eqx.error_if(self.p, ~(jnp.allclose(self.p, other.p)), "Tangent vectors must be defined at the same point")
    # Write other in terms of the current basis
    other = change_basis(other, self.basis)
    return TangentVector(p, self.components + other.components, self.basis)

  def __neg__(self) -> 'TangentVector':
    """
    Negate a tangent vector.
    """
    return TangentVector(self.p, -self.components, self.basis)

  def __sub__(self, other: 'TangentVector') -> 'TangentVector':
    """
    Subtract two tangent vectors.
    """
    return self + (-other)

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

@dispatch
def pushforward(X: TangentVector, f: Callable) -> TangentVector:
  """
  Pushforward a tangent vector through a smooth map.
  """
  raise NotImplementedError("It is not possible to pushforward tangent vectors whose components are Jets.")
  assert X.batch_size is None, "TangentVector must be unbatched to be pushed forward"

  X_standard: TangentVector = X.to_standard_basis() # X, dX/dx, d²X/dx²
  f_jet: Jet = function_to_jet(f, X.p) # f, df/dx, d²f/dx²
  T: Jet = f_jet.get_gradient_jet() # transformation of components

  # Apply the contravariant transform to the vector components
  fX_components: Jet = apply_contravariant_transform(T, X_standard.components) # f(X), df(X)/dx, d²f(X)/dx²
  # fX_components: Jet = change_coordinates(fX_components, f, X.p)

  new_standard_basis: BasisVectors = get_standard_basis(f_jet.value)
  # import pdb; pdb.set_trace()
  return TangentVector(f_jet.value, fX_components, new_standard_basis)

def lie_bracket(X: TangentVector, Y: TangentVector) -> TangentVector:
  """
  Compute the Lie bracket of two tangent vectors.
  """
  assert X.batch_size is None and Y.batch_size is None, "Use vmap to compute the Lie bracket of batched tangent vectors"
  standard_basis: BasisVectors = get_standard_basis(X.p)

  # Go to the standard basis
  X_standard: TangentVector = change_basis(X, standard_basis)
  Y_standard: TangentVector = change_basis(Y, standard_basis)

  @jet_decorator
  def get_components(
    X_val: Float[Array, "N"],
    Y_val: Float[Array, "N"],
    X_grad: Float[Array, "N N"],
    Y_grad: Float[Array, "N N"]
  ) -> Float[Array, "N"]:
    return Y_grad@X_val - X_grad@Y_val

  X_val = X_standard.components.get_value_jet()
  Y_val = Y_standard.components.get_value_jet()
  X_grad = X_standard.components.get_gradient_jet()
  Y_grad = Y_standard.components.get_gradient_jet()
  components: Jet = get_components(X_val, Y_val, X_grad, Y_grad)
  out = TangentVector(X.p, components, standard_basis)

  # Change back to the original basis
  out = change_basis(out, X.basis)
  return out

def tangent_vectors_are_equivalent(a: TangentVector, b: TangentVector) -> bool:
  """
  Check if two tangent vectors are equal up to a change of basis.
  """
  a_standard: TangentVector = a.to_standard_basis()
  b_standard: TangentVector = b.to_standard_basis()
  assert jnp.allclose(a_standard.p, b_standard.p)
  return jnp.allclose(a_standard.components.value, b_standard.components.value) and jnp.allclose(a_standard.components.gradient, b_standard.components.gradient) and jnp.allclose(a_standard.components.hessian, b_standard.components.hessian)