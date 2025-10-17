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
from local_coordinates.basis import BasisVectors, DualBasis, get_basis_transform, get_standard_basis

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

  def to_dual(self) -> 'DualFrame':
    @jet_decorator
    def to_dual_components(E_vals):
      return jnp.linalg.inv(E_vals)
    comps_val = self.components.get_value_jet()
    dual_components = to_dual_components(comps_val)
    # Build the corresponding dual basis from the primal basis
    @jet_decorator
    def to_dual_basis_components(E_basis_vals):
      return jnp.linalg.inv(E_basis_vals)
    dual_basis_components = to_dual_basis_components(self.basis.components.get_value_jet())
    dual_basis = DualBasis(p=self.p, components=dual_basis_components)
    return DualFrame(p=self.p, components=dual_components, basis=dual_basis)

@dispatch
def change_coordinates(frame: Frame, new_basis: BasisVectors) -> Frame:
  """
  Transform a frame from one basis to another.
  """
  # Compute linear transform from the current basis to the new basis.
  T: Jet = get_basis_transform(frame.basis, new_basis)

  # Apply only the value-level linear map to the frame components while
  # propagating derivatives coming from the frame itself.
  @jet_decorator
  def transform_components(T_value, components):
    # Left-multiply by T (contravariant index transformation).
    return jnp.einsum("...ij,...jk->...ik", T_value, components)

  T_value = T.get_value_jet()
  comps_val = frame.components.get_value_jet()
  new_components = transform_components(T_value, comps_val)

  return Frame(p=frame.p, components=new_components, basis=new_basis)

def get_lie_bracket_components(frame: Frame) -> Jet:
  """
  Get the components of the Lie bracket of the frame.  Returns
  c_{ij}^k where [E_i, E_j] = c_{ij}^k E_k.
  """
  standard_basis = get_standard_basis(frame.p)

  # Go to the standard basis
  frame_standard: Frame = change_coordinates(frame, standard_basis)
  standard_components = frame_standard.components

  @jet_decorator
  def get_components(basis_vals: Array, basis_grads: Array) -> Array:
    term1 = jnp.einsum("ai,kja->kij", basis_vals, basis_grads)
    term2 = jnp.einsum("aj,kia->kij", basis_vals, basis_grads)
    return term1 - term2

  basis_vals = standard_components.get_value_jet()
  basis_grads = standard_components.get_gradient_jet()
  components_euclidean: Jet = get_components(basis_vals, basis_grads)

  # Convert to components in the same basis that we started with
  T_jet: Jet = get_basis_transform(standard_basis, frame.basis)

  @jet_decorator
  def change_basis(components_euclidean_vals: Array, T_val: Array) -> Array:
    return jnp.einsum("ka,aij->kij", T_val, components_euclidean_vals)

  out: Jet = change_basis(components_euclidean.get_value_jet(), T_jet.get_value_jet())
  return out

class DualFrame(AbstractBatchableObject):
  """
  A covector frame (dual basis). The covector components are written
  in the standard Euclidean coordinates.
  """
  p: Float[Array, "N"]
  components: Annotated[Jet, "N D"]
  basis: DualBasis

  def __check_init__(self):
    assert isinstance(self.components, Jet), "components must be a Jet"
    if self.components.ndim != self.p.ndim + 1:
      raise ValueError(f"Invalid number of dimensions: {self.components.ndim}")

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.basis.batch_size

  def to_primal(self) -> Frame:
    @jet_decorator
    def to_primal_components(theta_vals):
      return jnp.linalg.inv(theta_vals)
    comps_val = self.components.get_value_jet()
    primal_components = to_primal_components(comps_val)
    # Recover the corresponding primal basis from the stored dual basis
    @jet_decorator
    def to_primal_basis_components(theta_basis_vals):
      return jnp.linalg.inv(theta_basis_vals)
    primal_basis_components = to_primal_basis_components(self.basis.components.get_value_jet())
    primal_basis = BasisVectors(p=self.p, components=primal_basis_components)
    return Frame(p=self.p, components=primal_components, basis=primal_basis)


@dispatch
def change_coordinates(frame: DualFrame, new_basis: DualBasis) -> DualFrame:
  """
  Transform a covector frame from one basis to another.
  """
  # Use the dual-basis transform; coordinates right-multiply by T_dual
  T = get_basis_transform(frame.basis, new_basis)

  @jet_decorator
  def transform_components(components, T_val):
    # Right-multiply by T_dual so θ_new @ E_new = θ_old @ E_old
    return jnp.einsum("...ij,...jk->...ik", components, T_val)

  comps_val = frame.components.get_value_jet()
  T_val = T.get_value_jet()
  new_components = transform_components(comps_val, T_val)

  return DualFrame(p=frame.p, components=new_components, basis=new_basis)