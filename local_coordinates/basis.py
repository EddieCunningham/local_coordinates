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
from local_coordinates.jet import Jet, jet_decorator, change_coordinates as change_coordinates_jet
from local_coordinates.jacobian import Jacobian
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
def change_coordinates(
  basis: BasisVectors,
  x_to_z_jacobian: Jacobian
) -> BasisVectors:
  """
  Change the coordinates of a set of basis vectors using a precomputed Jacobian
  for the forward map z(x).

  Uses the formulas from notes/change_coordinates.md:
    Value:    E_new[j,i] = E[j,a] * G[i,a]
    Gradient: dE_new[j,i,k] = -G[i,b] H[b,k,m] G[m,a] E[j,a]
                            + G[i,a] J[b,k] dE[j,a,b]
    Hessian:  (see notes for full formula)

  where G = forward Jacobian, J = inverse Jacobian, H = Hessian of inverse map,
  T = third derivative of inverse map.
  """
  # Get forward Jacobian G and inverse Jacobian with its derivatives
  G = x_to_z_jacobian.value  # G[i,a] = dz^i/dx^a
  J_inv = x_to_z_jacobian.get_inverse()
  J = J_inv.value            # J[a,k] = dx^a/dz^k
  H = J_inv.gradient         # H[b,k,m] = d^2x^b/dz^k dz^m
  T = J_inv.hessian          # T[b,k,m,l] = d^3x^b/dz^k dz^m dz^l

  # Original basis components and derivatives
  E = basis.components.value           # E[j,a]
  dE = basis.components.gradient       # dE[j,a,b] = dE^a_j/dx^b
  d2E = basis.components.hessian       # d2E[j,a,b,c] = d^2E^a_j/dx^b dx^c

  # Value: E_new_j^i = E_j^a G_a^i = (G @ E)[i,j]
  # With convention E[a,j] = E_j^a (column j = basis vector j)
  E_new = G @ E

  # Gradient: dE_new_j^i/dz^k = -G^i_b H^b_{km} G^m_a E^a_j + G^i_a J^b_k dE^a_j/dx^b
  # With convention E[a,j], dE[a,j,b], output dE_new[i,j,k]
  if dE is None or H is None:
    dE_new = None
  else:
    term1 = -jnp.einsum("ib,bkm,ma,aj->ijk", G, H, G, E)
    term2 = jnp.einsum("ia,bk,ajb->ijk", G, J, dE)
    dE_new = term1 + term2

  # Hessian: full formula from notes/change_coordinates.md
  # With convention E[a,j], dE[a,j,b], d2E[a,j,b,c], output d2E_new[i,j,k,l]
  if d2E is None or T is None or dE_new is None:
    d2E_new = None
  else:
    # Terms involving E (no derivatives)
    h1 = jnp.einsum("ic,cnl,nb,bkm,ma,aj->ijkl", G, H, G, H, G, E)
    h2 = jnp.einsum("ib,bkm,mc,cnl,na,aj->ijkl", G, H, G, H, G, E)
    h3 = -jnp.einsum("ib,bkml,ma,aj->ijkl", G, T, G, E)

    # Terms involving dE (first derivatives)
    h4 = -jnp.einsum("ib,bkm,ma,cl,ajc->ijkl", G, H, G, J, dE)
    h5 = -jnp.einsum("ic,cnl,na,bk,ajb->ijkl", G, H, G, J, dE)
    h6 = jnp.einsum("ia,bkl,ajb->ijkl", G, H, dE)

    # Term involving d2E (second derivatives)
    h7 = jnp.einsum("ia,bk,cl,ajbc->ijkl", G, J, J, d2E)

    d2E_new = h1 + h2 + h3 + h4 + h5 + h6 + h7

  new_components = Jet(value=E_new, gradient=dE_new, hessian=d2E_new, dim=E_new.shape[0])
  return BasisVectors(p=basis.p, components=new_components)


@dispatch
def change_coordinates(basis: BasisVectors, x_to_z: Callable[[Array], Array], x: Array) -> BasisVectors:
  """
  Change the coordinates of a basis vectors from one set of coordinates to another.
  """
  x = jnp.asarray(x)
  z = x_to_z(x)

  # Build Jacobian of the coordinate change z(x) up to third order
  dzdx = jax.jacrev(x_to_z)(x)                       # (a, i)
  d2zdx2 = jax.jacfwd(jax.jacrev(x_to_z))(x)         # (a, i, j)
  d3zdx3 = jax.jacfwd(jax.jacfwd(jax.jacrev(x_to_z)))(x)  # (a, i, j, k)
  J_zx = Jacobian(value=dzdx, gradient=d2zdx2, hessian=d3zdx3)

  # Use the Jacobian-based overload to transform components, then update p to z.
  transformed: BasisVectors = change_coordinates(basis, J_zx)
  return BasisVectors(p=z, components=transformed.components)

@dispatch.abstract
def change_basis(obj: Any, target_basis: BasisVectors) -> Any:
  """
  Change the basis of an object to a new basis.
  """
  pass

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
