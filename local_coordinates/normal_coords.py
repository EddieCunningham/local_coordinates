from typing import Any, Callable, Tuple, Annotated, Optional, List
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from local_coordinates.base import AbstractBatchableObject
from local_coordinates.basis import BasisVectors, get_basis_transform, get_standard_basis
from local_coordinates.basis import change_coordinates as change_coordinates_basis
from plum import dispatch
from local_coordinates.tensor import Tensor, change_basis
from local_coordinates.tensor import change_coordinates as change_coordinates_tensor
from local_coordinates.jet import Jet
from local_coordinates.jet import jet_decorator
from local_coordinates.jet import change_coordinates as change_coordinates_jet
from local_coordinates.metric import RiemannianMetric, lower_index
from local_coordinates.jet import get_identity_jet
from local_coordinates.frame import Frame, get_lie_bracket_between_frame_pairs, basis_to_frame
from local_coordinates.frame import change_coordinates as change_coordinates_frame
from local_coordinates.tangent import TangentVector
from local_coordinates.tangent import change_coordinates as change_coordinates_tangent
from local_coordinates.tensor import TensorType
from local_coordinates.connection import Connection, get_levi_civita_connection
from local_coordinates.connection import change_coordinates as change_coordinates_connection
from local_coordinates.jacobian import Jacobian

def _get_rnc_jacobian(gamma_bar: Jet, dxdv: Array) -> Jacobian:
  """
  Do the heavy lifting for the Jacobian for the Riemann normal coordinate transformation.

  Returns the forward Jacobian J_v_to_x (dx/dv).
  """

  # Second order: d2x/dv2 = -Gamma^i_ab * J^a_j * J^b_k
  d2xdv2 = -jnp.einsum("abi,aj,bk->ijk", gamma_bar.value, dxdv, dxdv)

  # Third order coefficients from geodesic equation:
  # T^i_abc = Sym_{abc}[-∂_m Γ^i_pq · J^m_a J^p_b J^q_c + 2·Γ^i_pq·Γ^p_mn · J^m_a J^n_b J^q_c]
  #
  # This formula has no explicit curvature term - it comes from differentiating
  # the geodesic equation d²γ/dt² + Γ(γ)·(dγ/dt)² = 0 at t=0.

  # Term 1: -∂_m Γ^i_pq · J^m_a J^p_b J^q_c
  # gamma_bar.gradient[p, q, i, m] = ∂_m Γ^i_pq
  unsym_term1 = -jnp.einsum("pqim,ma,pb,qc->iabc", gamma_bar.gradient, dxdv, dxdv, dxdv)

  # Term 2: +2·Γ^i_pq·Γ^p_mn · J^m_a J^n_b J^q_c
  # gamma_bar.value[p, q, i] = Γ^i_pq
  # gamma_bar.value[m, n, p] = Γ^p_mn  (contracted over p)
  unsym_term2 = 2 * jnp.einsum("pqi,mnp,ma,nb,qc->iabc", gamma_bar.value, gamma_bar.value, dxdv, dxdv, dxdv)

  unsym = unsym_term1 + unsym_term2

  # Symmetrize over the last 3 indices (a, b, c)
  # T^i_abc must be symmetric because partial derivatives commute
  d3xdv3 = (unsym +
            jnp.transpose(unsym, (0, 1, 3, 2)) +  # i,a,c,b
            jnp.transpose(unsym, (0, 2, 1, 3)) +  # i,b,a,c
            jnp.transpose(unsym, (0, 2, 3, 1)) +  # i,b,c,a
            jnp.transpose(unsym, (0, 3, 1, 2)) +  # i,c,a,b
            jnp.transpose(unsym, (0, 3, 2, 1))    # i,c,b,a
           ) / 6

  # Construct Jacobian for the forward map x(v) at v = 0
  # Note: p attribute was removed from Jacobian.
  # The base point v0 is implicitly 0 in the normal coordinate chart.
  J_v_to_x = Jacobian(value=dxdv, gradient=d2xdv2, hessian=d3xdv3)

  return J_v_to_x


def _get_inverse_rnc_jacobian(gamma_bar: Jet, dxdv: Array) -> Jacobian:
  """
  Compute the inverse RNC Jacobian J_x_to_v (dv/dx) directly.

  This function computes the Jacobian for the map v(x) using explicit formulas
  derived from the general inverse-Jacobian formula, simplified for RNC.

  The formulas are:
    - Value: K = J^{-1} (matrix inverse)
    - Gradient: d²v/dx² = K @ Γ (simpler than general O(n^5) formula)
    - Hessian: d³v/dx³ = (1/3) K @ [∂Γ terms + Γ² terms]

  The hessian formula is derived by substituting the RNC-specific forms of the
  forward Jacobian derivatives into the general inverse formula and simplifying.
  The Γ² contributions from term_C (third derivative of forward map) and term_B
  (products of second derivatives) partially cancel, leaving a 1/3 factor.

  Args:
    gamma_bar: Christoffel symbols as a Jet with value Γ^i_pq and gradient ∂_m Γ^i_pq.
    dxdv: The orthonormal frame matrix J = dx/dv.

  Returns:
    J_x_to_v: The inverse Jacobian (dv/dx) at x = p.
  """
  # Value: K = J^{-1}
  # K^i_j = (J^{-1})^i_j = dv^i/dx^j
  dvdx = jnp.linalg.inv(dxdv)

  # Gradient: d²v^i/dx^j dx^k = K^i_c Γ^c_jk
  # This is much simpler than the general inverse formula which requires O(n^5) ops.
  # gamma_bar.value[j, k, c] = Γ^c_jk
  d2vdx2 = jnp.einsum("ic,jkc->ijk", dvdx, gamma_bar.value)

  # Hessian: d³v^i/dx^j dx^k dx^l
  #
  # The formula is derived from the general inverse-Jacobian formula:
  #   S^i_jkl = term_C + term_B
  # where term_C involves the third derivative of the forward map and
  # term_B involves products of second derivatives.
  #
  # After substituting the RNC-specific forms and simplifying:
  #   S^i_jkl = (1/3) K^i_a [∂_l Γ^a_jk + ∂_j Γ^a_kl + ∂_k Γ^a_lj
  #                         + Γ^a_cl Γ^c_jk + Γ^a_cj Γ^c_kl + Γ^a_ck Γ^c_lj]
  #
  # The 1/3 factor arises because:
  # - term_C contributes: (1/3)[∂Γ terms] - (2/3)[Γ² terms]
  # - term_B contributes: [Γ² terms]
  # - Combined Γ² terms: -2/3 + 1 = 1/3

  # Derivative terms: ∂_l Γ^a_jk + ∂_j Γ^a_kl + ∂_k Γ^a_lj
  # gamma_bar.gradient[j, k, a, l] = ∂_l Γ^a_jk
  deriv_term1 = jnp.transpose(gamma_bar.gradient, (2, 3, 0, 1))  # a, l, j, k -> a, j, k, l
  deriv_term2 = jnp.transpose(gamma_bar.gradient, (2, 0, 1, 3))  # a, j, k, l (from [k,l,a,j])
  deriv_term3 = jnp.transpose(gamma_bar.gradient, (2, 1, 3, 0))  # a, j, k, l (from [l,j,a,k])

  deriv_terms = deriv_term1 + deriv_term2 + deriv_term3

  # Γ² terms: Γ^a_cl Γ^c_jk + Γ^a_cj Γ^c_kl + Γ^a_ck Γ^c_lj
  # gamma_bar.value[c, l, a] = Γ^a_cl, gamma_bar.value[j, k, c] = Γ^c_jk
  # So Γ^a_cl Γ^c_jk = einsum over c with result shape (a, j, k, l)
  gamma2_term1 = jnp.einsum("cla,jkc->ajkl", gamma_bar.value, gamma_bar.value)
  gamma2_term2 = jnp.einsum("cja,klc->ajkl", gamma_bar.value, gamma_bar.value)
  gamma2_term3 = jnp.einsum("cka,ljc->ajkl", gamma_bar.value, gamma_bar.value)

  gamma2_terms = gamma2_term1 + gamma2_term2 + gamma2_term3

  # Combine with 1/3 factor and contract with K
  bracket = deriv_terms + gamma2_terms
  d3vdx3 = jnp.einsum("ia,ajkl->ijkl", dvdx, bracket) / 3

  return Jacobian(value=dvdx, gradient=d2vdx2, hessian=d3vdx3)


def _compute_rnc_jacobians(
  metric: RiemannianMetric,
  compute_x_to_v: bool = True,
  compute_v_to_x: bool = True,
  frame_rotation: Optional[Array] = None
) -> Tuple[Optional[Jacobian], Optional[Jacobian]]:
  """
  Compute RNC Jacobians efficiently.

  Jacobians are computed directly using explicit formulas, avoiding
  the expensive general inverse-Jacobian computation.

  Args:
    metric: The Riemannian metric.
    compute_x_to_v: Whether to compute the inverse Jacobian (dv/dx).
    compute_v_to_x: Whether to compute the forward Jacobian (dx/dv).
    frame_rotation: Optional orthogonal matrix Q (satisfying Q^T Q = I in the
      Euclidean sense) that resolves the rotational ambiguity of RNC. The
      orthonormal frame is right-multiplied by Q, so dxdv becomes E @ Q where
      E is the default eigenvector-based frame. Because E already maps from a
      Euclidean tangent space to x-coordinates, Q acts as a rotation in the
      flat tangent space. When None, the identity is used.

  Returns:
    (J_x_to_v, J_v_to_x): The inverse Jacobian (dv/dx) and forward Jacobian (dx/dv).
  """
  # Go to the standard basis.
  standard_basis = get_standard_basis(metric.basis.p)
  metric_std: RiemannianMetric = change_basis(metric, standard_basis)

  connection: Connection = get_levi_civita_connection(metric_std)
  gamma_bar: Jet = connection.christoffel_symbols

  # Construct an orthonormal basis of tangent vectors at the point p.
  gij: Array = metric_std.components.value
  eigenvalues, eigenvectors = jnp.linalg.eigh(gij)

  dxdv = jnp.einsum("ij,j->ij", eigenvectors, jax.lax.rsqrt(eigenvalues))

  if frame_rotation is not None:
    dxdv = dxdv @ frame_rotation

  # Compute Jacobians directly using explicit formulas
  J_v_to_x = _get_rnc_jacobian(gamma_bar, dxdv) if compute_v_to_x else None
  J_x_to_v = _get_inverse_rnc_jacobian(gamma_bar, dxdv) if compute_x_to_v else None

  return J_x_to_v, J_v_to_x


def get_transformation_to_riemann_normal_coordinates(
  metric: RiemannianMetric,
  J_x_to_v: Optional[Jacobian] = None,
  frame_rotation: Optional[Array] = None
) -> Jacobian:
  """
  Get the Jacobian for the transformation TO Riemann normal coordinates (dv/dx).

  Args:
    metric: The Riemannian metric.
    J_x_to_v: Optional pre-computed Jacobian. If provided, returns it directly.
    frame_rotation: Optional orthogonal matrix resolving the RNC rotational ambiguity.

  Returns:
    J_x_to_v: The Jacobian for the inverse map v(x) at x = metric.basis.p.
  """
  if J_x_to_v is not None:
    return J_x_to_v

  J_x_to_v, _ = _compute_rnc_jacobians(metric, compute_v_to_x=False, frame_rotation=frame_rotation)
  return J_x_to_v


def get_transformation_from_riemann_normal_coordinates(
  metric: RiemannianMetric,
  J_v_to_x: Optional[Jacobian] = None,
  frame_rotation: Optional[Array] = None
) -> Jacobian:
  """
  Get the Jacobian for the transformation FROM Riemann normal coordinates (dx/dv).

  Args:
    metric: The Riemannian metric.
    J_v_to_x: Optional pre-computed Jacobian. If provided, returns it directly.
    frame_rotation: Optional orthogonal matrix resolving the RNC rotational ambiguity.

  Returns:
    J_v_to_x: The Jacobian for the forward map x(v) at v = 0.
  """
  if J_v_to_x is not None:
    return J_v_to_x

  _, J_v_to_x = _compute_rnc_jacobians(metric, compute_x_to_v=False, frame_rotation=frame_rotation)
  return J_v_to_x


def get_rnc_jacobians(
  metric: RiemannianMetric,
  frame_rotation: Optional[Array] = None
) -> Tuple[Jacobian, Jacobian]:
  """
  Get both RNC Jacobians efficiently.

  This is the preferred way to get both Jacobians when you need both,
  as it avoids redundant computation.

  Args:
    metric: The Riemannian metric.
    frame_rotation: Optional orthogonal matrix resolving the RNC rotational ambiguity.

  Returns:
    (J_x_to_v, J_v_to_x): The inverse Jacobian (dv/dx) and forward Jacobian (dx/dv).
  """
  return _compute_rnc_jacobians(metric, frame_rotation=frame_rotation)


def _resolve_jacobian_pair(
  metric: RiemannianMetric,
  J_x_to_v: Optional[Jacobian],
  J_v_to_x: Optional[Jacobian],
  frame_rotation: Optional[Array] = None
) -> Tuple[Jacobian, Jacobian]:
  """
  Resolve a pair of Jacobians, computing missing ones as needed.

  If neither is provided, computes standard RNC Jacobians.
  If only one is provided, computes the other as its inverse.
  If both are provided, returns them as-is.
  """
  if J_x_to_v is None and J_v_to_x is None:
    return get_rnc_jacobians(metric, frame_rotation=frame_rotation)
  elif J_x_to_v is None:
    return J_v_to_x.get_inverse(), J_v_to_x
  elif J_v_to_x is None:
    return J_x_to_v, J_x_to_v.get_inverse()
  else:
    return J_x_to_v, J_v_to_x


def get_rnc_basis(
  metric: RiemannianMetric,
  J_v_to_x: Optional[Jacobian] = None,
  frame_rotation: Optional[Array] = None
) -> BasisVectors:
  """
  Get the Riemann normal coordinate basis as a BasisVectors object.

  The RNC basis vectors are ∂/∂v^i, which in x-coordinates have components
  E^a_i = ∂x^a/∂v^i. This function returns the basis with derivatives
  expressed with respect to x (not v).

  Args:
    metric: The Riemannian metric.
    J_v_to_x: Optional pre-computed forward Jacobian (dx/dv). If not provided,
              it will be computed.
    frame_rotation: Optional orthogonal matrix resolving the RNC rotational ambiguity.
  """
  if J_v_to_x is None:
    J_v_to_x = get_transformation_from_riemann_normal_coordinates(metric, frame_rotation=frame_rotation)

  # J_v_to_x has derivatives w.r.t. v, but BasisVectors needs derivatives w.r.t. x.
  # Use change_coordinates_jet to convert from v-derivatives to x-derivatives.
  jacobian_as_jet = Jet(
    value=J_v_to_x.value,
    gradient=J_v_to_x.gradient,
    hessian=J_v_to_x.hessian
  )
  rnc_basis_components = change_coordinates_jet(jacobian_as_jet, J_v_to_x)

  return BasisVectors(p=metric.basis.p, components=rnc_basis_components)

def get_rnc_frame(
  metric: RiemannianMetric,
  J_v_to_x: Optional[Jacobian] = None,
  frame_rotation: Optional[Array] = None
) -> Frame:
  """
  Get the Riemann normal coordinate frame as a Frame object.

  Args:
    metric: The Riemannian metric.
    J_v_to_x: Optional pre-computed forward Jacobian (dx/dv). If not provided,
              it will be computed.
    frame_rotation: Optional orthogonal matrix resolving the RNC rotational ambiguity.
  """
  rnc_basis = get_rnc_basis(metric, J_v_to_x=J_v_to_x, frame_rotation=frame_rotation)
  return Frame(p=metric.basis.p, components=get_identity_jet(metric.basis.p.shape[0]), basis=rnc_basis)

@dispatch
def to_riemann_normal_coordinates(
  metric: RiemannianMetric,
  J_x_to_v: Optional[Jacobian] = None,
  J_v_to_x: Optional[Jacobian] = None,
  frame_rotation: Optional[Array] = None
) -> RiemannianMetric:
  """
  Transform a RiemannianMetric to Riemann normal coordinates.

  In RNC at the origin:
    - The metric components are δ_ij (identity)
    - The first derivatives of the metric vanish
    - The Christoffel symbols vanish

  Args:
    metric: The Riemannian metric.
    J_x_to_v: Optional pre-computed inverse Jacobian (dv/dx).
    J_v_to_x: Optional pre-computed forward Jacobian (dx/dv).
    frame_rotation: Optional orthogonal matrix resolving the RNC rotational ambiguity.
  """
  J_x_to_v, J_v_to_x = _resolve_jacobian_pair(metric, J_x_to_v, J_v_to_x, frame_rotation=frame_rotation)

  rnc_basis = get_rnc_basis(metric, J_v_to_x=J_v_to_x)

  # Change basis to RNC basis (components become identity, derivatives still w.r.t. x)
  metric_rnc_basis = change_basis(metric, rnc_basis)

  # Change coordinates (derivatives now w.r.t. v)
  metric_rnc = change_coordinates_tensor(metric_rnc_basis, J_x_to_v)

  return metric_rnc


@dispatch
def to_riemann_normal_coordinates(
  basis: BasisVectors,
  metric: RiemannianMetric,
  J_x_to_v: Optional[Jacobian] = None,
  frame_rotation: Optional[Array] = None
) -> BasisVectors:
  """
  Transform a BasisVectors object to Riemann normal coordinates.

  The basis vectors are re-expressed in RNC: their components become functions
  of v (the normal coordinates) instead of x (the original coordinates).

  Note: For BasisVectors there is no "change basis" step since BasisVectors
  represents the basis itself. We only re-express them in v-coordinates.

  Args:
    basis: The basis vectors to transform.
    metric: The Riemannian metric.
    J_x_to_v: Optional pre-computed inverse Jacobian (dv/dx).
    frame_rotation: Optional orthogonal matrix resolving the RNC rotational ambiguity.
  """
  J_x_to_v = get_transformation_to_riemann_normal_coordinates(metric, J_x_to_v=J_x_to_v, frame_rotation=frame_rotation)
  return change_coordinates_basis(basis, J_x_to_v)


@dispatch
def to_riemann_normal_coordinates(
  vector: TangentVector,
  metric: RiemannianMetric,
  J_x_to_v: Optional[Jacobian] = None,
  J_v_to_x: Optional[Jacobian] = None,
  frame_rotation: Optional[Array] = None
) -> TangentVector:
  """
  Transform a TangentVector to Riemann normal coordinates.

  The vector is first re-expressed in the RNC basis (∂/∂v^i), then the
  derivatives are transformed to be with respect to v.

  Args:
    vector: The tangent vector to transform.
    metric: The Riemannian metric.
    J_x_to_v: Optional pre-computed inverse Jacobian (dv/dx).
    J_v_to_x: Optional pre-computed forward Jacobian (dx/dv).
    frame_rotation: Optional orthogonal matrix resolving the RNC rotational ambiguity.
  """
  from local_coordinates.tangent import change_basis as change_basis_tangent

  J_x_to_v, J_v_to_x = _resolve_jacobian_pair(metric, J_x_to_v, J_v_to_x, frame_rotation=frame_rotation)

  rnc_basis = get_rnc_basis(metric, J_v_to_x=J_v_to_x)

  # Change basis to RNC basis (components transform, derivatives still w.r.t. x)
  vector_rnc_basis = change_basis_tangent(vector, rnc_basis)

  # Change coordinates (derivatives now w.r.t. v)
  vector_rnc = change_coordinates_tangent(vector_rnc_basis, J_x_to_v)

  return vector_rnc


@dispatch
def to_riemann_normal_coordinates(
  frame: Frame,
  metric: RiemannianMetric,
  J_x_to_v: Optional[Jacobian] = None,
  J_v_to_x: Optional[Jacobian] = None,
  frame_rotation: Optional[Array] = None
) -> Frame:
  """
  Transform a Frame to Riemann normal coordinates.

  The frame is first re-expressed in the RNC basis (∂/∂v^i), then the
  derivatives are transformed to be with respect to v.

  Args:
    frame: The frame to transform.
    metric: The Riemannian metric.
    J_x_to_v: Optional pre-computed inverse Jacobian (dv/dx).
    J_v_to_x: Optional pre-computed forward Jacobian (dx/dv).
    frame_rotation: Optional orthogonal matrix resolving the RNC rotational ambiguity.
  """
  from local_coordinates.frame import change_basis as change_basis_frame

  J_x_to_v, J_v_to_x = _resolve_jacobian_pair(metric, J_x_to_v, J_v_to_x, frame_rotation=frame_rotation)

  rnc_basis = get_rnc_basis(metric, J_v_to_x=J_v_to_x)

  # Change basis to RNC basis
  frame_rnc_basis = change_basis_frame(frame, rnc_basis)

  # Change coordinates
  frame_rnc = change_coordinates_frame(frame_rnc_basis, J_x_to_v)

  return frame_rnc


@dispatch
def to_riemann_normal_coordinates(
  tensor: Tensor,
  metric: RiemannianMetric,
  J_x_to_v: Optional[Jacobian] = None,
  J_v_to_x: Optional[Jacobian] = None,
  frame_rotation: Optional[Array] = None
) -> Tensor:
  """
  Transform a Tensor to Riemann normal coordinates.

  The tensor is first re-expressed in the RNC basis, then the derivatives
  are transformed to be with respect to v.

  Args:
    tensor: The tensor to transform.
    metric: The Riemannian metric.
    J_x_to_v: Optional pre-computed inverse Jacobian (dv/dx).
    J_v_to_x: Optional pre-computed forward Jacobian (dx/dv).
    frame_rotation: Optional orthogonal matrix resolving the RNC rotational ambiguity.
  """
  J_x_to_v, J_v_to_x = _resolve_jacobian_pair(metric, J_x_to_v, J_v_to_x, frame_rotation=frame_rotation)

  rnc_basis = get_rnc_basis(metric, J_v_to_x=J_v_to_x)

  # Change basis to RNC basis
  tensor_rnc_basis = change_basis(tensor, rnc_basis)

  # Change coordinates
  tensor_rnc = change_coordinates_tensor(tensor_rnc_basis, J_x_to_v)

  return tensor_rnc


@dispatch
def to_riemann_normal_coordinates(
  connection: Connection,
  metric: RiemannianMetric,
  J_x_to_v: Optional[Jacobian] = None,
  J_v_to_x: Optional[Jacobian] = None,
  frame_rotation: Optional[Array] = None
) -> Connection:
  """
  Transform a Connection to Riemann normal coordinates.

  The connection is first re-expressed in the RNC basis, then the derivatives
  are transformed to be with respect to v.

  Note: For the Levi-Civita connection of the given metric, the Christoffel
  symbols will vanish at the origin of RNC.

  Args:
    connection: The connection to transform.
    metric: The Riemannian metric.
    J_x_to_v: Optional pre-computed inverse Jacobian (dv/dx).
    J_v_to_x: Optional pre-computed forward Jacobian (dx/dv).
    frame_rotation: Optional orthogonal matrix resolving the RNC rotational ambiguity.
  """
  from local_coordinates.connection import change_basis as change_basis_connection

  J_x_to_v, J_v_to_x = _resolve_jacobian_pair(metric, J_x_to_v, J_v_to_x, frame_rotation=frame_rotation)

  rnc_basis = get_rnc_basis(metric, J_v_to_x=J_v_to_x)

  # Change basis to RNC basis
  connection_rnc_basis = change_basis_connection(connection, rnc_basis)

  # Change coordinates
  connection_rnc = change_coordinates_connection(connection_rnc_basis, J_x_to_v)

  return connection_rnc
