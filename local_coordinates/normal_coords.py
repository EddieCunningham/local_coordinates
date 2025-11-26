from typing import Any, Callable, Tuple, Annotated, Optional, List
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from linsdex import AbstractBatchableObject
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
# Note: Riemann curvature is not needed for the geodesic-derived third derivative formula
from local_coordinates.jacobian import Jacobian, get_inverse

def get_transformation_to_riemann_normal_coordinates(metric: RiemannianMetric) -> Jacobian:
  """
  Get the Riemann normal coordinate basis for a Riemannian metric.
  """

  # Go to the standard basis.
  standard_basis = get_standard_basis(metric.basis.p)
  metric: RiemannianMetric = change_basis(metric, standard_basis)

  connection: Connection = get_levi_civita_connection(metric)
  gamma_bar: Jet = connection.christoffel_symbols

  # Construct an orthonormal basis of tangent vectors at the point p.
  gij: Array = metric.components.value
  eigenvalues, eigenvectors = jnp.linalg.eigh(gij)

  dxdv = jnp.einsum("ij,j->ij", eigenvectors, jax.lax.rsqrt(eigenvalues))

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

  # Invert to obtain Jacobian for the inverse map v(x) at x = metric.basis.p
  J_x_to_v = get_inverse(J_v_to_x)
  return J_x_to_v


def get_rnc_basis(metric: RiemannianMetric) -> BasisVectors:
  """
  Get the Riemann normal coordinate basis as a BasisVectors object.

  The RNC basis vectors are ∂/∂v^i, which in x-coordinates have components
  E^a_i = ∂x^a/∂v^i. This function returns the basis with derivatives
  expressed with respect to x (not v).
  """
  J_x_to_v = get_transformation_to_riemann_normal_coordinates(metric)
  J_v_to_x = get_inverse(J_x_to_v)

  # J_v_to_x has derivatives w.r.t. v, but BasisVectors needs derivatives w.r.t. x.
  # Use change_coordinates_jet to convert from v-derivatives to x-derivatives.
  jacobian_as_jet = Jet(
    value=J_v_to_x.value,
    gradient=J_v_to_x.gradient,
    hessian=J_v_to_x.hessian
  )
  rnc_basis_components = change_coordinates_jet(jacobian_as_jet, J_v_to_x)

  return BasisVectors(p=metric.basis.p, components=rnc_basis_components)

def get_rnc_frame(metric: RiemannianMetric) -> Frame:
  """
  Get the Riemann normal coordinate frame as a Frame object.
  """
  rnc_basis = get_rnc_basis(metric)
  return Frame(p=metric.basis.p, components=get_identity_jet(metric.basis.p.shape[0]), basis=rnc_basis)

@dispatch
def to_riemann_normal_coordinates(metric: RiemannianMetric) -> RiemannianMetric:
  """
  Transform a RiemannianMetric to Riemann normal coordinates.

  In RNC at the origin:
    - The metric components are δ_ij (identity)
    - The first derivatives of the metric vanish
    - The Christoffel symbols vanish
  """
  J_x_to_v = get_transformation_to_riemann_normal_coordinates(metric)
  rnc_basis = get_rnc_basis(metric)

  # Change basis to RNC basis (components become identity, derivatives still w.r.t. x)
  metric_rnc_basis = change_basis(metric, rnc_basis)

  # Change coordinates (derivatives now w.r.t. v)
  metric_rnc = change_coordinates_tensor(metric_rnc_basis, J_x_to_v)

  return metric_rnc


@dispatch
def to_riemann_normal_coordinates(
  basis: BasisVectors,
  metric: RiemannianMetric
) -> BasisVectors:
  """
  Transform a BasisVectors object to Riemann normal coordinates.

  The basis vectors are re-expressed in RNC: their components become functions
  of v (the normal coordinates) instead of x (the original coordinates).

  Note: For BasisVectors there is no "change basis" step since BasisVectors
  represents the basis itself. We only re-express them in v-coordinates.
  """
  J_x_to_v = get_transformation_to_riemann_normal_coordinates(metric)
  return change_coordinates_basis(basis, J_x_to_v)


@dispatch
def to_riemann_normal_coordinates(
  vector: TangentVector,
  metric: RiemannianMetric
) -> TangentVector:
  """
  Transform a TangentVector to Riemann normal coordinates.

  The vector is first re-expressed in the RNC basis (∂/∂v^i), then the
  derivatives are transformed to be with respect to v.
  """
  from local_coordinates.tangent import change_basis as change_basis_tangent

  J_x_to_v = get_transformation_to_riemann_normal_coordinates(metric)
  rnc_basis = get_rnc_basis(metric)

  # Change basis to RNC basis (components transform, derivatives still w.r.t. x)
  vector_rnc_basis = change_basis_tangent(vector, rnc_basis)

  # Change coordinates (derivatives now w.r.t. v)
  vector_rnc = change_coordinates_tangent(vector_rnc_basis, J_x_to_v)

  return vector_rnc


@dispatch
def to_riemann_normal_coordinates(
  frame: Frame,
  metric: RiemannianMetric
) -> Frame:
  """
  Transform a Frame to Riemann normal coordinates.

  The frame is first re-expressed in the RNC basis (∂/∂v^i), then the
  derivatives are transformed to be with respect to v.
  """
  from local_coordinates.frame import change_basis as change_basis_frame

  J_x_to_v = get_transformation_to_riemann_normal_coordinates(metric)
  rnc_basis = get_rnc_basis(metric)

  # Change basis to RNC basis
  frame_rnc_basis = change_basis_frame(frame, rnc_basis)

  # Change coordinates
  frame_rnc = change_coordinates_frame(frame_rnc_basis, J_x_to_v)

  return frame_rnc


@dispatch
def to_riemann_normal_coordinates(
  tensor: Tensor,
  metric: RiemannianMetric
) -> Tensor:
  """
  Transform a Tensor to Riemann normal coordinates.

  The tensor is first re-expressed in the RNC basis, then the derivatives
  are transformed to be with respect to v.
  """
  J_x_to_v = get_transformation_to_riemann_normal_coordinates(metric)
  rnc_basis = get_rnc_basis(metric)

  # Change basis to RNC basis
  tensor_rnc_basis = change_basis(tensor, rnc_basis)

  # Change coordinates
  tensor_rnc = change_coordinates_tensor(tensor_rnc_basis, J_x_to_v)

  return tensor_rnc


@dispatch
def to_riemann_normal_coordinates(
  connection: Connection,
  metric: RiemannianMetric
) -> Connection:
  """
  Transform a Connection to Riemann normal coordinates.

  The connection is first re-expressed in the RNC basis, then the derivatives
  are transformed to be with respect to v.

  Note: For the Levi-Civita connection of the given metric, the Christoffel
  symbols will vanish at the origin of RNC.
  """
  from local_coordinates.connection import change_basis as change_basis_connection

  J_x_to_v = get_transformation_to_riemann_normal_coordinates(metric)
  rnc_basis = get_rnc_basis(metric)

  # Change basis to RNC basis
  connection_rnc_basis = change_basis_connection(connection, rnc_basis)

  # Change coordinates
  connection_rnc = change_coordinates_connection(connection_rnc_basis, J_x_to_v)

  return connection_rnc
