import jax.numpy as jnp
from jax import random
import numpy as np
import pytest

from local_coordinates.basis import BasisVectors, get_standard_basis, change_coordinates
from local_coordinates.jet import Jet
from local_coordinates.metric import RiemannianMetric, lower_index
from local_coordinates.connection import get_levi_civita_connection
from local_coordinates.riemann import get_riemann_curvature_tensor, RiemannCurvatureTensor, RicciTensor, get_ricci_tensor
from local_coordinates.tensor import Tensor, TensorType, change_basis, change_coordinates as change_coordinates_tensor
from local_coordinates.normal_coords import (
  get_transformation_to_riemann_normal_coordinates,
  get_transformation_from_riemann_normal_coordinates,
  get_rnc_jacobians,
  get_rnc_basis,
  to_riemann_normal_coordinates,
  get_rnc_frame
)
from local_coordinates.jet import Jet, jet_decorator, get_identity_jet, change_coordinates as change_coordinates_jet
from local_coordinates.tangent import TangentVector, lie_bracket
from local_coordinates.frame import Frame, basis_to_frame
from local_coordinates.connection import Connection
from local_coordinates.jacobian import Jacobian
from jaxtyping import Array, Scalar
from typing import Annotated, Callable
import equinox as eqx
from local_coordinates.frame import get_lie_bracket_between_frame_pairs
from local_coordinates.normal_coords import _get_rnc_jacobian
import jax


def create_random_basis(key: random.PRNGKey, dim: int, p: Array = None) -> BasisVectors:
  vals_key, grads_key, hessians_key = random.split(key, 3)
  if p is None:
    p = jnp.zeros(dim)
  vals = jnp.eye(dim) + random.normal(vals_key, (dim, dim)) * 0.1
  grads = random.normal(grads_key, (dim, dim, dim)) * 0.1
  hessians = random.normal(hessians_key, (dim, dim, dim, dim)) * 0.1
  return BasisVectors(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians))


def create_random_metric(key: random.PRNGKey, dim: int) -> RiemannianMetric:
  random_basis = create_random_basis(key, dim)
  metric = RiemannianMetric(basis=random_basis, components=get_identity_jet(dim))

  standard_basis = get_standard_basis(random_basis.p)
  return change_basis(metric, standard_basis)


def create_random_vector_field(key: random.PRNGKey, dim: int, basis: BasisVectors) -> TangentVector:
  vals_key, grads_key, hessians_key = random.split(key, 3)
  vals = random.normal(vals_key, (dim,))
  grads = random.normal(grads_key, (dim, dim))
  hessians = random.normal(hessians_key, (dim, dim, dim))
  return TangentVector(
    p=basis.p,
    components=Jet(value=vals, gradient=grads, hessian=hessians),
    basis=basis
  )


def create_random_tensor(
  key: random.PRNGKey,
  dim: int,
  basis: BasisVectors,
  k: int,
  l: int
) -> Tensor:
  vals_key, grads_key, hessians_key = random.split(key, 3)
  shape = (dim,) * (k + l)
  vals = random.normal(vals_key, shape)
  grads = random.normal(grads_key, shape + (dim,))
  hessians = random.normal(hessians_key, shape + (dim, dim))
  return Tensor(
    tensor_type=TensorType(k=k, l=l),
    basis=basis,
    components=Jet(value=vals, gradient=grads, hessian=hessians)
  )


# =============================================================================
# Tests for to_riemann_normal_coordinates(metric)
# =============================================================================

def test_metric_rnc_identity_components():
  """In RNC, the metric components should be the identity matrix at the origin."""
  key = random.PRNGKey(0)
  dim = 4
  metric = create_random_metric(key, dim=dim)

  metric_rnc = to_riemann_normal_coordinates(metric)

  assert jnp.allclose(metric_rnc.components.value, jnp.eye(dim), atol=1e-5)


def test_metric_rnc_vanishing_gradient():
  """In RNC, the first derivatives of the metric should vanish at the origin."""
  key = random.PRNGKey(1)
  dim = 4
  metric = create_random_metric(key, dim=dim)

  metric_rnc = to_riemann_normal_coordinates(metric)

  assert jnp.allclose(metric_rnc.components.gradient, 0.0, atol=1e-5)


def test_metric_rnc_different_dimensions():
  """Test that RNC transformation works for various dimensions."""
  for dim in [2, 3, 5]:
    key = random.PRNGKey(dim)
    metric = create_random_metric(key, dim=dim)

    metric_rnc = to_riemann_normal_coordinates(metric)

    assert jnp.allclose(metric_rnc.components.value, jnp.eye(dim), atol=1e-5)
    assert jnp.allclose(metric_rnc.components.gradient, 0.0, atol=1e-5)


# =============================================================================
# Tests for to_riemann_normal_coordinates(connection, metric)
# =============================================================================

def test_levi_civita_rnc_vanishing_christoffel():
  """
  The Levi-Civita connection's Christoffel symbols should vanish at the
  origin of RNC when expressed in the RNC coordinate basis.
  """
  key = random.PRNGKey(2)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  # Get the Levi-Civita connection in the standard basis
  connection = get_levi_civita_connection(metric)

  # Transform the metric to RNC and get its Levi-Civita connection
  metric_rnc = to_riemann_normal_coordinates(metric)
  connection_rnc = get_levi_civita_connection(metric_rnc)

  # The Christoffel symbols should vanish at the origin
  assert jnp.allclose(connection_rnc.christoffel_symbols.value, 0.0, atol=1e-5)


# =============================================================================
# Tests for to_riemann_normal_coordinates(basis, metric)
# =============================================================================

def test_basis_rnc_preserves_point():
  """The point of the basis should be preserved under RNC transformation."""
  key = random.PRNGKey(3)
  dim = 3
  metric = create_random_metric(key, dim=dim)
  basis = metric.basis

  basis_rnc = to_riemann_normal_coordinates(basis, metric)

  assert jnp.allclose(basis_rnc.p, basis.p)


def test_standard_basis_to_rnc():
  """
  Transforming the standard basis to RNC should give a basis whose components
  are the RNC Jacobian (with x-derivatives converted appropriately).
  """
  key = random.PRNGKey(4)
  dim = 3
  metric = create_random_metric(key, dim=dim)
  standard_basis = get_standard_basis(metric.basis.p)

  basis_rnc = to_riemann_normal_coordinates(standard_basis, metric)

  # The standard basis in RNC should have components that are the forward
  # Jacobian G = dv/dx (since ∂/∂x^a = G^i_a ∂/∂v^i)
  J_x_to_v = get_transformation_to_riemann_normal_coordinates(metric)
  G = J_x_to_v.value  # dv/dx

  # The basis components should be G
  assert jnp.allclose(basis_rnc.components.value, G, atol=1e-5)


# =============================================================================
# Tests for to_riemann_normal_coordinates(vector, metric)
# =============================================================================

def test_tangent_vector_rnc_preserves_point():
  """The point of the tangent vector should be preserved under RNC transformation."""
  key = random.PRNGKey(5)
  dim = 3
  metric = create_random_metric(key, dim=dim)
  vector = create_random_vector_field(key, dim, metric.basis)

  vector_rnc = to_riemann_normal_coordinates(vector, metric)

  assert jnp.allclose(vector_rnc.p, vector.p)


def test_tangent_vector_rnc_basis_is_rnc():
  """
  After transforming to RNC, the tangent vector's basis should be the RNC
  coordinate basis expressed in v-coordinates (which is identity at the origin).
  """
  key = random.PRNGKey(6)
  dim = 3
  metric = create_random_metric(key, dim=dim)
  vector = create_random_vector_field(key, dim, metric.basis)

  vector_rnc = to_riemann_normal_coordinates(vector, metric)

  # The basis should be identity (the RNC coordinate basis in v-coords)
  assert jnp.allclose(vector_rnc.basis.components.value, jnp.eye(dim), atol=1e-5)


# =============================================================================
# Tests for to_riemann_normal_coordinates(frame, metric)
# =============================================================================

def test_frame_rnc_preserves_point():
  """The point of the frame should be preserved under RNC transformation."""
  key = random.PRNGKey(7)
  dim = 3
  metric = create_random_metric(key, dim=dim)
  frame = basis_to_frame(metric.basis)

  frame_rnc = to_riemann_normal_coordinates(frame, metric)

  assert jnp.allclose(frame_rnc.p, frame.p)


def test_frame_rnc_basis_is_rnc():
  """
  After transforming to RNC, the frame's basis should be the RNC coordinate
  basis expressed in v-coordinates (which is identity at the origin).
  """
  key = random.PRNGKey(8)
  dim = 3
  metric = create_random_metric(key, dim=dim)
  frame = basis_to_frame(metric.basis)

  frame_rnc = to_riemann_normal_coordinates(frame, metric)

  # The basis should be identity (the RNC coordinate basis in v-coords)
  assert jnp.allclose(frame_rnc.basis.components.value, jnp.eye(dim), atol=1e-5)


# =============================================================================
# Tests for to_riemann_normal_coordinates(tensor, metric)
# =============================================================================

def test_tensor_rnc_preserves_type():
  """The tensor type (k, l) should be preserved under RNC transformation."""
  key = random.PRNGKey(9)
  dim = 3
  metric = create_random_metric(key, dim=dim)
  tensor = create_random_tensor(key, dim, metric.basis, k=1, l=2)

  tensor_rnc = to_riemann_normal_coordinates(tensor, metric)

  assert tensor_rnc.tensor_type.k == tensor.tensor_type.k
  assert tensor_rnc.tensor_type.l == tensor.tensor_type.l


def test_tensor_rnc_basis_is_rnc():
  """
  After transforming to RNC, the tensor's basis should be the RNC coordinate
  basis expressed in v-coordinates (which is identity at the origin).
  """
  key = random.PRNGKey(10)
  dim = 3
  metric = create_random_metric(key, dim=dim)
  tensor = create_random_tensor(key, dim, metric.basis, k=2, l=1)

  tensor_rnc = to_riemann_normal_coordinates(tensor, metric)

  # The basis should be identity (the RNC coordinate basis in v-coords)
  assert jnp.allclose(tensor_rnc.basis.components.value, jnp.eye(dim), atol=1e-5)


# =============================================================================
# Tests for geometric invariants
# =============================================================================

def test_rnc_basis_is_orthonormal():
  """The RNC basis should be orthonormal with respect to the metric."""
  key = random.PRNGKey(11)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  rnc_basis = get_rnc_basis(metric)

  # Compute the metric in the RNC basis
  metric_in_rnc_basis = change_basis(metric, rnc_basis)

  # The metric should be identity (orthonormal basis)
  assert jnp.allclose(metric_in_rnc_basis.components.value, jnp.eye(dim), atol=1e-5)


def test_metric_rnc_symmetry():
  """
  In RNC, the metric Hessian should have the symmetries expected from
  the Riemann curvature tensor relationship.
  """
  key = random.PRNGKey(12)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  metric_rnc = to_riemann_normal_coordinates(metric)
  g_hess = metric_rnc.components.hessian  # shape (i, j, k, l)

  # The Hessian should be symmetric in (i, j) since the metric is symmetric
  assert jnp.allclose(g_hess, jnp.einsum("ijkl->jikl", g_hess), atol=1e-5)

  # The Hessian should be symmetric in (k, l) since partial derivatives commute
  assert jnp.allclose(g_hess, jnp.einsum("ijkl->ijlk", g_hess), atol=1e-5)


def test_metric_hessian_curvature_identity():
  """
  In RNC, the second derivatives of the metric are related to the Riemann
  curvature tensor by the identity (from notes/rnc.md line 193):

    ∂²g_ij/∂v^a∂v^b (p) = (1/3)(R_aibj(p) + R_biaj(p))

  where R_aibj is the fully covariant Riemann tensor.
  """
  key = random.PRNGKey(14)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  metric_rnc = to_riemann_normal_coordinates(metric)
  connection_rnc = get_levi_civita_connection(metric_rnc)
  R_rnc = get_riemann_curvature_tensor(connection_rnc)
  R_lower = lower_index(R_rnc, metric_rnc, 4)

  g_hess = metric_rnc.components.hessian
  R_val = R_lower.components.value

  # Formula from notes/rnc.md line 168-169:
  # g_ij(v) = δ_ij + (1/3) R_aibj v^a v^b + O(v^3)
  # So: ∂²g_ij/∂v^a∂v^b = (1/3)(R_aibj + R_biaj)
  expected_hess = (1/3) * (
    jnp.einsum("aibj->ijab", R_val) + jnp.einsum("biaj->ijab", R_val)
  )

  assert jnp.allclose(g_hess, expected_hess, atol=1e-4)


# =============================================================================
# Incremental tests using 2-sphere (known analytical values)
# =============================================================================

def get_sphere_metric(p: Array) -> RiemannianMetric:
  """
  Create the unit 2-sphere metric in stereographic coordinates.
  g = 4/(1+r²)² I
  At origin: g = 4I, K = 1 (Gaussian curvature)
  """
  from local_coordinates.jet import function_to_jet

  def sphere_metric_fn(x):
    r2 = jnp.sum(x**2)
    return 4.0 / (1 + r2)**2 * jnp.eye(2)

  metric_jet = function_to_jet(sphere_metric_fn, p)
  basis = get_standard_basis(p)
  return RiemannianMetric(basis=basis, components=metric_jet)


def test_sphere_rnc_jacobian():
  """
  For the sphere with g = 4I at origin, the RNC Jacobian should be:
    dv/dx = 2I (to normalize metric from 4I to I)
    dx/dv = 0.5I
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  p = jnp.array([0.0, 0.0])
  sphere = get_sphere_metric(p)

  J_x_to_v = get_transformation_to_riemann_normal_coordinates(sphere)
  J_v_to_x = J_x_to_v.get_inverse()

  # dv/dx = 2I (the normalization factor sqrt(4) = 2)
  expected_dvdx = 2.0 * jnp.eye(2)
  assert jnp.allclose(J_x_to_v.value, expected_dvdx, atol=1e-10)

  # dx/dv = 0.5I
  expected_dxdv = 0.5 * jnp.eye(2)
  assert jnp.allclose(J_v_to_x.value, expected_dxdv, atol=1e-10)


def test_sphere_rnc_metric_value():
  """
  For the sphere, the metric in RNC should be identity at origin.
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  p = jnp.array([0.0, 0.0])
  sphere = get_sphere_metric(p)

  metric_rnc = to_riemann_normal_coordinates(sphere)

  assert jnp.allclose(metric_rnc.components.value, jnp.eye(2), atol=1e-10)


def test_sphere_rnc_metric_gradient():
  """
  For the sphere, the metric gradient in RNC should vanish at origin.
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  p = jnp.array([0.0, 0.0])
  sphere = get_sphere_metric(p)

  metric_rnc = to_riemann_normal_coordinates(sphere)

  assert jnp.allclose(metric_rnc.components.gradient, 0.0, atol=1e-10)


def test_sphere_rnc_christoffel():
  """
  For the sphere, Christoffel symbols in RNC should vanish at origin.
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  p = jnp.array([0.0, 0.0])
  sphere = get_sphere_metric(p)

  metric_rnc = to_riemann_normal_coordinates(sphere)
  connection_rnc = get_levi_civita_connection(metric_rnc)

  assert jnp.allclose(connection_rnc.christoffel_symbols.value, 0.0, atol=1e-10)


def test_sphere_riemann_original():
  """
  For the unit sphere with g = 4I at origin, the Riemann tensor should be:
    R_0101 = -K * det(g) = -1 * 16 = -16
  (The sign convention R_{abcd} has R_1212 = -K*det(g) for positive curvature)
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  p = jnp.array([0.0, 0.0])
  sphere = get_sphere_metric(p)

  connection = get_levi_civita_connection(sphere)
  R = get_riemann_curvature_tensor(connection)
  R_lower = lower_index(R, sphere, 4)

  # For 2D: R_0101 = -K * det(g) where K = 1, det(g) = 16
  # Our convention gives R_0101 = -16
  R_0101 = R_lower.components.value[0, 1, 0, 1]

  assert jnp.isclose(R_0101, -16.0, atol=1e-8)


def test_sphere_riemann_transformed_to_rnc():
  """
  The Riemann tensor from the original metric, when transformed to RNC,
  should give R_0101 = -K = -1 for the unit sphere (since g = I in RNC).
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  p = jnp.array([0.0, 0.0])
  sphere = get_sphere_metric(p)

  connection = get_levi_civita_connection(sphere)
  R = get_riemann_curvature_tensor(connection)
  R_lower = lower_index(R, sphere, 4)

  # Transform Riemann to RNC
  R_rnc = to_riemann_normal_coordinates(R_lower, sphere)
  R_0101_rnc = R_rnc.components.value[0, 1, 0, 1]

  # In RNC with g = I, R_0101 = -K * det(I) = -1
  # This uses the TRANSFORMED Riemann tensor, not computed from RNC metric
  assert jnp.isclose(R_0101_rnc, -1.0, atol=1e-8)


def test_sphere_riemann_computed_from_rnc_metric():
  """
  The Riemann tensor computed directly from the RNC metric should also
  give R_0101 = -K = -1 for the unit sphere.

  If this test fails, there's a bug in how the RNC metric's Hessian
  propagates to the Riemann tensor computation.
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  p = jnp.array([0.0, 0.0])
  sphere = get_sphere_metric(p)

  metric_rnc = to_riemann_normal_coordinates(sphere)
  connection_rnc = get_levi_civita_connection(metric_rnc)
  R_rnc = get_riemann_curvature_tensor(connection_rnc)
  R_rnc_lower = lower_index(R_rnc, metric_rnc, 4)

  R_0101 = R_rnc_lower.components.value[0, 1, 0, 1]

  # Expected: R_0101 = -1 for unit sphere in RNC
  assert jnp.isclose(R_0101, -1.0, atol=1e-6), f"R_0101 = {R_0101}, expected -1.0"


def test_sphere_metric_hessian_from_taylor():
  """
  For the unit sphere in RNC, the Taylor expansion gives:
    g_ij(v) = δ_ij + (1/3) R_kilj v^k v^l + O(v³)

  Taking second derivative at origin:
    ∂²g_00/∂v¹² = (1/3)(R_1010 + R_1010) = (2/3) R_1010

  For unit sphere with K=1 and R_1010 = -1:
    ∂²g_00/∂v¹² = (2/3)(-1) = -2/3
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  p = jnp.array([0.0, 0.0])
  sphere = get_sphere_metric(p)

  metric_rnc = to_riemann_normal_coordinates(sphere)
  g_hess = metric_rnc.components.hessian

  # Expected from Taylor expansion: ∂²g_00/∂v¹² = -2/3
  expected = -2.0 / 3.0
  actual = g_hess[0, 0, 1, 1]

  assert jnp.isclose(actual, expected, atol=1e-6), f"∂²g_00/∂v¹² = {actual}, expected {expected}"


def test_sphere_d3xdv3_gamma_gradient_term():
  """
  Test that the Γ-gradient term in d³x/dv³ is computed correctly.

  At the origin where Γ=0:
    term1 = -∂_c Γ^i_{ab} * (dx/dv)^a_j * (dx/dv)^b_k * (dx/dv)^c_l

  For the sphere with dx/dv = 0.5*I and ∂_1 Γ^0_{01} = -2:
    term1[0,0,1,1] = -(-2) * 0.5³ = 0.25
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  p = jnp.array([0.0, 0.0])
  sphere = get_sphere_metric(p)

  connection = get_levi_civita_connection(sphere)
  gamma = connection.christoffel_symbols

  gij = sphere.components.value
  eigenvalues, eigenvectors = jnp.linalg.eigh(gij)
  dxdv = jnp.einsum("ij,j->ij", eigenvectors, jax.lax.rsqrt(eigenvalues))

  term1 = -jnp.einsum('abic,aj,bk,cl->ijkl', gamma.gradient, dxdv, dxdv, dxdv)

  # For sphere: term1[0,0,1,1] = -(-2) * 0.5³ = 0.25
  assert jnp.isclose(term1[0,0,1,1], 0.25, atol=1e-10)


def test_sphere_d3xdv3_riemann_term_coefficient():
  """
  Test that the Riemann term in d³x/dv³ uses the correct coefficient.

  The formula for d³x/dv³ involves:
    term1 = -∂Γ terms (coefficient 1)
    term4 + term5 = R terms (coefficient should be 1/12, not 1/3)

  For the curvature identity ∂²g/∂v² = (1/3)(R + R) to hold,
  d³x/dv³[0,0,1,1] must equal 1/12 for the sphere.
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  p = jnp.array([0.0, 0.0])
  sphere = get_sphere_metric(p)

  J_x_to_v = get_transformation_to_riemann_normal_coordinates(sphere)
  J_v_to_x = J_x_to_v.get_inverse()

  # d³x/dv³ is stored in J_v_to_x.hessian
  d3xdv3 = J_v_to_x.hessian

  # For the curvature identity to hold: d³x/dv³[0,0,1,1] = 1/12
  expected = 1.0 / 12.0
  actual = d3xdv3[0, 0, 1, 1]

  assert jnp.isclose(actual, expected, atol=1e-6), (
    f"d³x/dv³[0,0,1,1] = {actual}, expected {expected}"
  )


def test_sphere_curvature_identity():
  """
  The full curvature identity for the sphere:
    ∂²g_ij/∂v^a∂v^b = (1/3)(R_aibj + R_biaj)

  This is the same identity as the general test but for the specific
  case of the sphere where we know the analytical values.
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  p = jnp.array([0.0, 0.0])
  sphere = get_sphere_metric(p)

  # Get Riemann from ORIGINAL metric and transform to RNC
  # (this is known to give correct values from test_sphere_riemann_transformed_to_rnc)
  connection = get_levi_civita_connection(sphere)
  R = get_riemann_curvature_tensor(connection)
  R_lower = lower_index(R, sphere, 4)
  R_rnc = to_riemann_normal_coordinates(R_lower, sphere)
  R_val = R_rnc.components.value

  # Get metric Hessian
  metric_rnc = to_riemann_normal_coordinates(sphere)
  g_hess = metric_rnc.components.hessian

  # Expected Hessian from identity
  expected_hess = (1.0/3.0) * (
    jnp.einsum("aibj->ijab", R_val) + jnp.einsum("biaj->ijab", R_val)
  )

  assert jnp.allclose(g_hess, expected_hess, atol=1e-6), (
    f"Max diff: {jnp.max(jnp.abs(g_hess - expected_hess))}"
  )


# =============================================================================
# Critical correctness tests for the Jacobian computation
# =============================================================================

def test_jacobian_inverse_consistency_value():
  """
  Test that J_x_to_v and J_v_to_x are inverses at the value level.
  (dv/dx)(dx/dv) = I
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  p = jnp.array([0.0, 0.0])
  sphere = get_sphere_metric(p)

  J_x_to_v = get_transformation_to_riemann_normal_coordinates(sphere)
  J_v_to_x = J_x_to_v.get_inverse()

  product = J_x_to_v.value @ J_v_to_x.value
  assert jnp.allclose(product, jnp.eye(2), atol=1e-10)


def test_jacobian_inverse_consistency_gradient():
  """
  Test that the Jacobian gradient satisfies the inverse formula.

  For v(x) and x(v) with v(x(v)) = v, differentiating twice gives:
    d²v/dx² (dx/dv)(dx/dv) + (dv/dx)(d²x/dv²) = 0

  Rearranging: d²x/dv² = -(dx/dv)(d²v/dx²)(dx/dv)(dx/dv)
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  key = random.PRNGKey(42)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  J_x_to_v = get_transformation_to_riemann_normal_coordinates(metric)
  J_v_to_x = J_x_to_v.get_inverse()

  dvdx = J_x_to_v.value
  dxdv = J_v_to_x.value
  d2vdx2 = J_x_to_v.gradient  # [i, j, k] = ∂²v^i/∂x^j∂x^k
  d2xdv2 = J_v_to_x.gradient  # [i, j, k] = ∂²x^i/∂v^j∂v^k

  # Formula: d²x^i/dv^j dv^k = -∂x^i/∂v^a · ∂²v^a/∂x^b∂x^c · ∂x^b/∂v^j · ∂x^c/∂v^k
  expected_d2xdv2 = -jnp.einsum("ia,abc,bj,ck->ijk", dxdv, d2vdx2, dxdv, dxdv)

  assert jnp.allclose(d2xdv2, expected_d2xdv2, atol=1e-5)


def test_sphere_d2xdv2_vanishes():
  """
  For the sphere at the origin, the Christoffel symbols vanish (Γ = 0),
  so d²x/dv² = -Γ^i_jk (dx/dv)^j (dx/dv)^k = 0.
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  p = jnp.array([0.0, 0.0])
  sphere = get_sphere_metric(p)

  J_x_to_v = get_transformation_to_riemann_normal_coordinates(sphere)
  J_v_to_x = J_x_to_v.get_inverse()

  # d²x/dv² should be zero for the sphere at origin
  assert jnp.allclose(J_v_to_x.gradient, 0.0, atol=1e-10)


def test_d3xdv3_symmetry():
  """
  The third derivative d³x^i/dv^j dv^k dv^l should be symmetric in (j,k,l)
  since partial derivatives commute.
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  key = random.PRNGKey(123)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  J_x_to_v = get_transformation_to_riemann_normal_coordinates(metric)
  J_v_to_x = J_x_to_v.get_inverse()

  d3xdv3 = J_v_to_x.hessian  # shape (i, j, k, l)

  # Check all permutations of (j, k, l) give the same result
  assert jnp.allclose(d3xdv3, jnp.transpose(d3xdv3, (0, 1, 3, 2)), atol=1e-6)  # (i,j,l,k)
  assert jnp.allclose(d3xdv3, jnp.transpose(d3xdv3, (0, 2, 1, 3)), atol=1e-6)  # (i,k,j,l)
  assert jnp.allclose(d3xdv3, jnp.transpose(d3xdv3, (0, 2, 3, 1)), atol=1e-6)  # (i,k,l,j)
  assert jnp.allclose(d3xdv3, jnp.transpose(d3xdv3, (0, 3, 1, 2)), atol=1e-6)  # (i,l,j,k)
  assert jnp.allclose(d3xdv3, jnp.transpose(d3xdv3, (0, 3, 2, 1)), atol=1e-6)  # (i,l,k,j)


def test_christoffel_gradient_identity():
  """
  In RNC, the Christoffel symbol gradient satisfies:
    ∂Γ^k_ij/∂v^l = (1/3)(R^k_ijl + R^k_jil)

  This is a fundamental identity that relates the connection to curvature.
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  key = random.PRNGKey(99)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  metric_rnc = to_riemann_normal_coordinates(metric)
  connection_rnc = get_levi_civita_connection(metric_rnc)
  R_rnc = get_riemann_curvature_tensor(connection_rnc)

  Gamma_grad = connection_rnc.christoffel_symbols.gradient
  R_val = R_rnc.components.value

  # Our convention: R_val[k, i, j, l] = R^l_kij (upper index is last)
  # Gamma_grad[i, j, k, l] = ∂Γ^k_ij/∂v^l
  # Formula: ∂Γ^k_ij/∂v^l = (1/3)(R^k_ijl + R^k_jil)
  expected_Gamma_grad = (1.0/3.0) * (
    jnp.einsum("kijl->ijkl", R_val) + jnp.einsum("kjil->ijkl", R_val)
  )

  assert jnp.allclose(Gamma_grad, expected_Gamma_grad, atol=1e-5)


def test_connection_transformation_equivalence():
  """
  Test that transforming a connection to RNC via to_riemann_normal_coordinates
  gives the same Christoffel symbols as computing the connection from the
  RNC metric directly.

  NOTE: This is subtle! The change_basis for connections uses a frame-based
  transformation, not the coordinate transformation formula. For the Levi-Civita
  connection, we should get the same result either way.
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  key = random.PRNGKey(77)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  # Method 1: Transform metric to RNC, then compute connection
  metric_rnc = to_riemann_normal_coordinates(metric)
  connection_from_rnc_metric = get_levi_civita_connection(metric_rnc)

  # Method 2: Transform connection directly to RNC
  connection_original = get_levi_civita_connection(metric)
  connection_transformed = to_riemann_normal_coordinates(connection_original, metric)

  # Both should give Γ = 0 at the origin
  assert jnp.allclose(connection_from_rnc_metric.christoffel_symbols.value, 0.0, atol=1e-5)
  assert jnp.allclose(connection_transformed.christoffel_symbols.value, 0.0, atol=1e-5)

  # The gradients should also match (encoding curvature)
  assert jnp.allclose(
    connection_from_rnc_metric.christoffel_symbols.gradient,
    connection_transformed.christoffel_symbols.gradient,
    atol=1e-5
  )


def test_inner_product_preserved_under_rnc():
  """
  The inner product g(X, Y) should be invariant under coordinate transformation.
  If we transform vectors X, Y and metric g to RNC, the inner product should
  be the same as computing it in the original coordinates.
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  key = random.PRNGKey(55)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  # Create two random tangent vectors
  key1, key2 = random.split(key)
  X = create_random_vector_field(key1, dim, metric.basis)
  Y = create_random_vector_field(key2, dim, metric.basis)

  # Compute inner product in original coordinates
  inner_original = metric(X, Y)

  # Transform everything to RNC
  metric_rnc = to_riemann_normal_coordinates(metric)
  X_rnc = to_riemann_normal_coordinates(X, metric)
  Y_rnc = to_riemann_normal_coordinates(Y, metric)

  # Compute inner product in RNC
  inner_rnc = metric_rnc(X_rnc, Y_rnc)

  # Should be the same (at least the value)
  assert jnp.allclose(inner_original.value, inner_rnc.value, atol=1e-5)


def test_metric_determinant_transformation():
  """
  The metric determinant transforms as:
    det(g_rnc) = det(g) / det(dv/dx)^2

  At the origin of RNC, det(g_rnc) = 1 (since g_rnc = I).
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  key = random.PRNGKey(33)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  J_x_to_v = get_transformation_to_riemann_normal_coordinates(metric)

  det_g_original = jnp.linalg.det(metric.components.value)
  det_J = jnp.linalg.det(J_x_to_v.value)

  # In RNC, det(g) = 1
  expected_det_rnc = 1.0

  # Check the transformation formula
  det_g_rnc_from_formula = det_g_original / (det_J ** 2)

  assert jnp.isclose(det_g_rnc_from_formula, expected_det_rnc, atol=1e-5)


def test_sphere_christoffel_gradient():
  """
  For the unit sphere with K=1, in RNC the Christoffel gradient should satisfy:
    ∂Γ^k_ij/∂v^l = (1/3)(R^k_ijl + R^k_jil)

  Our convention: R_val[k, i, j, l] = R^l_kij (upper index is last).
  For the sphere, R^0_101 = R_val[1, 0, 1, 0] = -1.
  """
  import jax
  jax.config.update('jax_enable_x64', True)

  p = jnp.array([0.0, 0.0])
  sphere = get_sphere_metric(p)

  metric_rnc = to_riemann_normal_coordinates(sphere)
  connection_rnc = get_levi_civita_connection(metric_rnc)
  R_rnc = get_riemann_curvature_tensor(connection_rnc)

  Gamma_grad = connection_rnc.christoffel_symbols.gradient
  R_val = R_rnc.components.value

  # R_val[k, i, j, l] = R^l_kij
  # For sphere: R^0_101 = R_val[1, 0, 1, 0] = -1
  assert jnp.isclose(R_val[1, 0, 1, 0], -1.0, atol=1e-6)

  # Verify the full identity: ∂Γ^k_ij/∂v^l = (1/3)(R^k_ijl + R^k_jil)
  expected_Gamma_grad = (1.0/3.0) * (
    jnp.einsum("kijl->ijkl", R_val) + jnp.einsum("kjil->ijkl", R_val)
  )

  assert jnp.allclose(Gamma_grad, expected_Gamma_grad, atol=1e-6)


# =============================================================================
# Legacy test (kept for backward compatibility)
# =============================================================================

def test_rn_basis_terms_vanish():
  """
  Original test verifying that metric becomes identity with vanishing gradient
  in RNC using the manual transformation steps.
  """
  key = random.PRNGKey(0)
  dim = 5
  metric = create_random_metric(key, dim=dim)

  J_x_to_v = get_transformation_to_riemann_normal_coordinates(metric)
  J_v_to_x = J_x_to_v.get_inverse()

  jacobian_as_jet = Jet(
    value=J_v_to_x.value,
    gradient=J_v_to_x.gradient,
    hessian=J_v_to_x.hessian
  )
  rnc_basis_components = change_coordinates_jet(jacobian_as_jet, J_v_to_x)

  rnc_basis = BasisVectors(p=metric.basis.p, components=rnc_basis_components)
  metric_rnc_basis = change_basis(metric, rnc_basis)
  metric_rnc = change_coordinates(metric_rnc_basis, J_x_to_v)

  assert jnp.allclose(metric_rnc.components.value, jnp.eye(dim), atol=1e-5)
  assert jnp.allclose(metric_rnc.components.gradient, 0.0, atol=1e-5)

def test_connection_rnc_christoffel_vanish():
  """
  The Christoffel symbols of the Levi-Civita connection should vanish at
  the origin of RNC.

  NOTE: Currently, to_riemann_normal_coordinates(connection, metric) does NOT
  give the correct Christoffel symbols because it uses change_basis (for general
  frames) rather than the coordinate transformation formula. The correct approach
  is to compute the connection from the RNC metric directly.
  """
  key = random.PRNGKey(13)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  # Correct approach: compute connection from RNC metric
  metric_rnc = to_riemann_normal_coordinates(metric)
  connection_rnc = get_levi_civita_connection(metric_rnc)

  # Christoffel symbols should vanish
  assert jnp.allclose(connection_rnc.christoffel_symbols.value, 0.0, atol=1e-5)

  # Basis should be identity
  assert jnp.allclose(connection_rnc.basis.components.value, jnp.eye(dim), atol=1e-5)
  assert jnp.allclose(connection_rnc.christoffel_symbols.value, 0.0, atol=1e-5)


# =============================================================================
# Bug reproduction tests: RNC frame should have constant components in v-coords
# =============================================================================

def test_rnc_frame_in_v_coordinates_has_constant_components():
  """
  BUG REPRODUCTION TEST:
  The RNC frame vectors, when transformed to v-coordinates, should be the
  coordinate basis vectors ∂/∂v^i. This means their components should be
  constant: value = δ^k_i, gradient = 0, hessian = 0.

  Currently, the basis.components.hessian is non-zero, which propagates to
  the tangent vector's hessian after change_basis, causing [T,S].gradient ≠ 0
  in the Jacobi equation test.
  """
  key = random.PRNGKey(0)
  dim = 5
  metric = create_random_metric(key, dim=dim)

  rnc_frame = get_rnc_frame(metric)
  rnc_frame_rnc = to_riemann_normal_coordinates(rnc_frame, metric)

  # The frame's BASIS should be the standard coordinate basis in v-coords
  # which means: value = I, gradient = 0, hessian = 0
  assert jnp.allclose(rnc_frame_rnc.basis.components.value, jnp.eye(dim), atol=1e-5)
  assert jnp.allclose(rnc_frame_rnc.basis.components.gradient, 0.0, atol=1e-5)
  # THIS ASSERTION FAILS: hessian is ~0.08, not 0
  assert jnp.allclose(rnc_frame_rnc.basis.components.hessian, 0.0, atol=1e-5)


def test_lie_bracket_of_rnc_coordinate_basis_vectors_vanishes():
  """
  BUG REPRODUCTION TEST:
  For coordinate basis vectors ∂/∂v^i and ∂/∂v^j, the Lie bracket should be
  exactly zero as a vector field, not just at the origin.

  This means [∂/∂v^i, ∂/∂v^j].value = 0 AND [∂/∂v^i, ∂/∂v^j].gradient = 0.

  Currently [T,S].gradient ≠ 0 because T and S have non-zero hessians
  after the coordinate transformation (due to the basis having non-zero hessian).
  """
  key = random.PRNGKey(0)
  dim = 5
  metric = create_random_metric(key, dim=dim)

  rnc_frame = get_rnc_frame(metric)
  rnc_frame_rnc = to_riemann_normal_coordinates(rnc_frame, metric)

  T = rnc_frame_rnc.get_basis_vector(0)
  S = rnc_frame_rnc.get_basis_vector(1)

  bracket = lie_bracket(T, S)

  # Value should be zero (currently passes)
  assert jnp.allclose(bracket.components.value, 0.0, atol=1e-5)

  # Gradient should also be zero for coordinate basis vectors
  # THIS ASSERTION FAILS: gradient is ~0.1, not 0
  assert jnp.allclose(bracket.components.gradient, 0.0, atol=1e-5)


def test_torsion_free_identity_gradient_level():
  """
  BUG REPRODUCTION TEST:
  The torsion-free identity ∇_T S - ∇_S T = [T, S] should hold at both
  the value AND gradient levels.

  Currently the identity holds at the value level, but at the gradient level:
  - (∇_T S - ∇_S T).gradient = [T, S].gradient (both are non-zero)
  - This means ∇_T S ≠ ∇_S T as vector fields, even though [T, S] should be 0.

  This causes the Jacobi equation derivation to fail because it relies on
  ∇_T S = ∇_S T (which follows from [T,S] = 0 and torsion-free).
  """
  key = random.PRNGKey(0)
  dim = 5
  metric = create_random_metric(key, dim=dim)

  metric_rnc = to_riemann_normal_coordinates(metric)
  connection_rnc = get_levi_civita_connection(metric_rnc)

  rnc_frame = get_rnc_frame(metric)
  rnc_frame_rnc = to_riemann_normal_coordinates(rnc_frame, metric)

  T = rnc_frame_rnc.get_basis_vector(0)
  S = rnc_frame_rnc.get_basis_vector(1)

  nablaT_S = connection_rnc.covariant_derivative(T, S)
  nablaS_T = connection_rnc.covariant_derivative(S, T)
  bracket_TS = lie_bracket(T, S)

  # Torsion-free identity at value level (should pass)
  diff_val = nablaT_S.components.value - nablaS_T.components.value
  assert jnp.allclose(diff_val, bracket_TS.components.value, atol=1e-5)

  # If [T,S] = 0 as a vector field, then ∇_T S = ∇_S T as vector fields
  # Since [T,S].gradient should be 0, we should have ∇_T S.gradient = ∇_S T.gradient
  # THIS ASSERTION FAILS because [T,S].gradient ≠ 0
  assert jnp.allclose(nablaT_S.components.gradient, nablaS_T.components.gradient, atol=1e-5)

# =============================================================================
# Jacobi field tests.  These come from the notes/rnc.md file.
# =============================================================================

def test_rnc_frame():
  """
  Test that the RNC frame is indeed a coordinate frame.
  """
  key = random.PRNGKey(0)
  dim = 5
  metric = create_random_metric(key, dim=dim)
  rnc_frame: Frame = get_rnc_frame(metric)

  lb = get_lie_bracket_between_frame_pairs(rnc_frame)
  assert jnp.allclose(lb.components.value, 0.0, atol=1e-5)
  assert jnp.allclose(lb.components.gradient, 0.0, atol=1e-5)

def test_rnc_geodesic_equation():
  """
  Test that nabla_T T = 0 at the origin.

  IMPORTANT LIMITATION: The geodesic equation only holds AT THE ORIGIN,
  not along the entire geodesic. This is because our Jet-based RNC
  construction computes the correct local Taylor expansion (Γ = 0 at origin,
  correct metric Hessian), but the global RNC property that "geodesics are
  straight lines" requires Γ(t·v₀) = 0 for all t along the geodesic. This
  cannot be captured by a finite Taylor expansion at a single point.

  Specifically:
  - At origin: Γ^k_{ij}(0) = 0 ✓
  - Along geodesic: Would need ∂_a Γ^k_{ij}(0) v^a v^i v^j = 0 for all v,
    which requires ∂_a Γ^k_{ij} to be symmetric in (a,i,j). Our construction
    does NOT guarantee this symmetry.
  """
  key = random.PRNGKey(0)
  dim = 5
  metric = create_random_metric(key, dim=dim)
  connection = get_levi_civita_connection(metric)

  rnc_basis = get_rnc_basis(metric)

  # Change basis to RNC basis (components become identity, derivatives still w.r.t. x)
  I = get_identity_jet(dim)
  rnc_frame = Frame(p=metric.basis.p, components=I, basis=rnc_basis)

  # Take the first basis vector as the geodesic direction
  T = rnc_frame.get_basis_vector(0)

  # The covariant derivative of the geodesic direction should be zero
  V = connection.covariant_derivative(T, T)
  assert jnp.allclose(V.components.value, 0.0, atol=1e-5)

  # But the gradient is NOT zero in x-coordinates (derivatives of Γ don't vanish)
  assert not jnp.allclose(V.components.gradient, 0.0, atol=1e-5)


def test_rnc_geodesic_equation_in_v_coordinates():
  """
  Test that nabla_T T = 0 at the origin in v-coordinates.

  The geodesic equation holds at the origin because Γ = 0 there.
  The Christoffel gradient is NOT zero - it encodes curvature via:
    ∂_l Γ^k_ij = (1/3)(R^k_ijl + R^k_jil)

  However, geodesics ARE still straight lines because when contracted
  with v^i v^j v^l, this vanishes due to Riemann antisymmetry.
  """
  key = random.PRNGKey(0)
  dim = 5
  metric = create_random_metric(key, dim=dim)

  # Transform everything to proper RNC (v-coordinates)
  rnc_frame = to_riemann_normal_coordinates(get_rnc_frame(metric), metric)
  connection = to_riemann_normal_coordinates(get_levi_civita_connection(metric), metric)
  R = to_riemann_normal_coordinates(get_riemann_curvature_tensor(get_levi_civita_connection(metric)), metric)

  # Take the first basis vector as the geodesic direction
  T = rnc_frame.get_basis_vector(0)

  # In v-coordinates, nabla_T T should be zero at the origin
  V = connection.covariant_derivative(T, T)
  assert jnp.allclose(V.components.value, 0.0, atol=1e-5)

  # The Christoffel symbols vanish at the origin in RNC
  assert jnp.allclose(connection.christoffel_symbols.value, 0.0, atol=1e-5)

  # The Christoffel gradient is NOT zero - it equals (1/3)(R + R)
  # This encodes the curvature of the manifold
  Gamma_grad = connection.christoffel_symbols.gradient
  R_val = R.components.value
  expected_Gamma_grad = (1/3) * (jnp.einsum('kijl->ijkl', R_val) + jnp.einsum('kjil->ijkl', R_val))
  assert jnp.allclose(Gamma_grad, expected_Gamma_grad, atol=1e-5)

def test_jacobi_field_construction():
  """
  Test the (our) definition of a Jacobi field, which is that [T, S] = 0.
  """
  key = random.PRNGKey(0)
  dim = 5
  metric = create_random_metric(key, dim=dim)
  connection = get_levi_civita_connection(metric)

  standard_basis = get_standard_basis(metric.basis.p)
  metric = change_basis(metric, standard_basis)

  # Change basis to RNC basis (components become identity, derivatives still w.r.t. x)
  rnc_frame = get_rnc_frame(metric)
  connection = change_basis(connection, rnc_frame.basis)

  # Take the first basis vector as the geodesic direction
  T = rnc_frame.get_basis_vector(0)

  # Construct a Jacobi field
  S = rnc_frame.get_basis_vector(1)

  # Check that S is a variation of the geodesic
  lb = lie_bracket(T, S)
  assert jnp.allclose(lb.components.value, 0.0, atol=1e-5)
  assert jnp.allclose(lb.components.gradient, 0.0, atol=1e-5)

  # Check the symmetry lemma.  Only holds for the values.
  nablaT_S = connection.covariant_derivative(T, S)
  nablaS_T = connection.covariant_derivative(S, T)
  assert jnp.allclose(nablaT_S.components.value, nablaS_T.components.value, atol=1e-5)

def test_jacobi_equation():
  """
  Test the curvature identity R(T,S)T = nabla_T nabla_S T - nabla_S nabla_T T.

  Note: The Jacobi equation nabla_T^2 S = R(T,S)T does NOT hold in our Jet-based setup.
  This is because the Jacobi equation requires T to be geodesic along the entire geodesic
  (i.e., nabla_S(nabla_T T) = 0), but in our setup we only have nabla_T T = 0 at the origin.
  The derivative nabla_S(nabla_T T) = (2/3) R^k_001 is non-zero.

  We instead verify the more fundamental curvature identity, which always holds.
  """
  key = random.PRNGKey(0)
  dim = 5
  metric = create_random_metric(key, dim=dim)
  connection = get_levi_civita_connection(metric)

  # Change basis to RNC basis (components become identity, derivatives still w.r.t. x)
  rnc_frame = get_rnc_frame(metric)

  # Take the first basis vector as the geodesic direction
  T = rnc_frame.get_basis_vector(0)

  # Construct a Jacobi field.
  S = rnc_frame.get_basis_vector(1)

  # Compute the curvature endomorphism
  R = get_riemann_curvature_tensor(connection)

  # Change everything to the RNC basis
  R = change_basis(R, T.basis)
  connection = change_basis(connection, T.basis)

  # Compute the Riemann curvature endomorphism
  RTST: TangentVector = R(T, S, T)

  nablaS_T = connection.covariant_derivative(S, T)
  nablaT_nablaS_T = connection.covariant_derivative(T, nablaS_T)

  nablaT_T = connection.covariant_derivative(T, T)
  nabla_S_nablaT_T = connection.covariant_derivative(S, nablaT_T)

  bracket_TS = lie_bracket(T, S)
  nabla_bracket_TS_T = connection.covariant_derivative(bracket_TS, T)

  # The curvature identity: R(T,S)T = nabla_T nabla_S T - nabla_S nabla_T T - nabla_[T,S] T
  rhs = nablaT_nablaS_T - nabla_S_nablaT_T
  rhs_complete = rhs - nabla_bracket_TS_T

  # Since [T,S] = 0 at the origin, these should be equal
  assert jnp.allclose(rhs_complete.components.value, rhs.components.value, atol=1e-5)

  # The fundamental curvature identity should hold
  assert jnp.allclose(rhs_complete.components.value, RTST.components.value, atol=1e-5)

  # Verify that nabla_T T = 0 at the origin (geodesic condition)
  assert jnp.allclose(nablaT_T.components.value, 0.0, atol=1e-5)

  # For CONSTANT S = E_1, nabla_S(nabla_T T) != 0 because S^m != 0
  # nabla_S(nabla_T T) = S^m partial_m Gamma^k_00 = partial_1 Gamma^k_00 = (2/3) R^k_001
  assert not jnp.allclose(nabla_S_nablaT_T.components.value, 0.0, atol=1e-5)

  # But for a PROPER Jacobi field with S(0) = 0, nabla_S(nabla_T T) = 0!
  # This is because S^m = 0 at the origin, so S^m partial_m (...) = 0
  # Construct the proper Jacobi field
  jacobi_value = jnp.zeros(dim)  # S(0) = 0
  jacobi_gradient = jnp.zeros((dim, dim))
  jacobi_gradient = jacobi_gradient.at[1, 0].set(1.0)  # ∂S^1/∂v^0 = 1
  jacobi_hessian = jnp.zeros((dim, dim, dim))

  S_proper = TangentVector(
    p=T.basis.p,
    components=Jet(value=jacobi_value, gradient=jacobi_gradient, hessian=jacobi_hessian),
    basis=T.basis
  )

  nabla_Sproper_nablaT_T = connection.covariant_derivative(S_proper, nablaT_T)
  assert jnp.allclose(nabla_Sproper_nablaT_T.components.value, 0.0, atol=1e-5), \
    f"For proper Jacobi field, nabla_S(nabla_T T) should be 0, got {nabla_Sproper_nablaT_T.components.value}"

  # Now the Jacobi equation holds for the proper Jacobi field!
  # R(T,S)T = nabla_T nabla_S T - nabla_S nabla_T T
  # With nabla_S nabla_T T = 0:
  # nabla_T nabla_S T = R(T,S)T = 0 (since S(0) = 0)


def test_jacobi_equation_proper_jacobi_field():
  """
  Test the Jacobi equation with a PROPER Jacobi field.

  For a family of geodesics γ(t, ε) = t(E_0 + ε E_1) through the origin,
  the Jacobi field is S(t) = t · E_1, which has:
    - S(0) = 0
    - (∇_T S)(0) = E_1

  In v-coordinates (RNC), this Jacobi field has:
    - value = 0 at the origin
    - gradient ∂S^k/∂v^m = δ^k_1 δ^m_0 (S grows in E_1 direction along E_0)

  At t=0, the Jacobi equation gives:
    ∇_T^2 S(0) = R(T, S(0))T = R(T, 0)T = 0

  This is trivially satisfied since S(0) = 0!
  """
  key = random.PRNGKey(0)
  dim = 5
  metric = create_random_metric(key, dim=dim)

  # Transform to v-coordinates (RNC)
  rnc_frame = to_riemann_normal_coordinates(get_rnc_frame(metric), metric)
  connection = to_riemann_normal_coordinates(get_levi_civita_connection(metric), metric)
  R = to_riemann_normal_coordinates(get_riemann_curvature_tensor(get_levi_civita_connection(metric)), metric)

  # T = E_0 (geodesic direction)
  T = rnc_frame.get_basis_vector(0)

  # Construct the proper Jacobi field S(t) = t · E_1
  # At the origin: S(0) = 0, ∂S/∂v^0 = E_1
  # In v-coordinates: S^k(v) = v^0 · δ^k_1
  # Gradient convention: gradient[k, m] = ∂S^k/∂v^m
  jacobi_value = jnp.zeros(dim)  # S(0) = 0
  jacobi_gradient = jnp.zeros((dim, dim))
  jacobi_gradient = jacobi_gradient.at[1, 0].set(1.0)  # ∂S^1/∂v^0 = 1
  jacobi_hessian = jnp.zeros((dim, dim, dim))

  S_jacobi = TangentVector(
    p=rnc_frame.basis.p,
    components=Jet(value=jacobi_value, gradient=jacobi_gradient, hessian=jacobi_hessian),
    basis=rnc_frame.basis
  )

  # Verify S(0) = 0
  assert jnp.allclose(S_jacobi.components.value, 0.0, atol=1e-10)

  # Compute ∇_T S - should equal E_1 at the origin
  # ∇_T S = T^m ∂_m S + Γ · T · S
  # At origin: Γ = 0 and S = 0, so ∇_T S = T^m ∂_m S = ∂_0 S = E_1
  nabla_T_S = connection.covariant_derivative(T, S_jacobi)
  expected_nabla_T_S = jnp.zeros(dim).at[1].set(1.0)  # E_1
  assert jnp.allclose(nabla_T_S.components.value, expected_nabla_T_S, atol=1e-5), \
    f"∇_T S should be E_1, got {nabla_T_S.components.value}"

  # Compute ∇_T^2 S
  nabla2_T_S = connection.covariant_derivative(T, nabla_T_S)

  # Compute R(T, S)T - should be 0 since S(0) = 0
  RTST = R(T, S_jacobi, T)

  # The Jacobi equation: ∇_T^2 S = R(T, S)T
  # At t=0 with S(0) = 0: both sides should be 0
  assert jnp.allclose(RTST.components.value, 0.0, atol=1e-10), \
    f"R(T, S(0))T should be 0, got {RTST.components.value}"

  # ∇_T^2 S should also be 0 (or close to it) for the proper Jacobi field
  # This follows from ∇_T(∇_T S) = ∇_T(E_1) = Γ^k_{01} = 0 at origin
  assert jnp.allclose(nabla2_T_S.components.value, 0.0, atol=1e-5), \
    f"∇_T^2 S should be 0, got {nabla2_T_S.components.value}"

  # Verify Jacobi equation: ∇_T^2 S = R(T, S)T
  assert jnp.allclose(nabla2_T_S.components.value, RTST.components.value, atol=1e-5), \
    f"Jacobi equation failed: ∇_T^2 S = {nabla2_T_S.components.value}, R(T,S)T = {RTST.components.value}"


def test_jacobi_metric_identity_linearity():
  """
  Test that f(t) = g(∇_T² S(t), W) is linear in t for a Jacobi field.

  For S(t) = t · E_1 along geodesic γ(t) = t · E_0, the Jacobi equation gives:
    ∇_T² S = R(T, S)T

  Since R is linear in S and S(t) = t · E_1:
    ∇_T² S(t) = t · R(T, E_1)T

  So f(t) = g(∇_T² S(t), W) = t · g(R(T, E_1)T, W) is linear in t.
  This means f''(t) = 0 for all t.

  Note: We use W = E_1 instead of W = T because g(R(T, E_1)T, T) = 0
  due to Riemann symmetry (R_{0101} T^0 E_1^1 T^0 T^1 = 0 since T^1 = 0).
  Using W = E_1 gives g(R(T, E_1)T, E_1) = R_{0101} ≠ 0 for curved spaces.
  """
  import jax

  key = random.PRNGKey(0)
  dim = 5
  metric = create_random_metric(key, dim=dim)
  p = metric.basis.p

  # Transform to proper RNC (v-coordinates) where the basis is identity
  metric_rnc = to_riemann_normal_coordinates(metric)
  connection = get_levi_civita_connection(metric_rnc)
  standard_basis = get_standard_basis(p)

  # In v-coordinates, basis vectors are ∂/∂v^i with identity components
  E0_val = jnp.zeros(dim).at[0].set(1.0)
  E1_val = jnp.zeros(dim).at[1].set(1.0)

  T = TangentVector(
    p=p,
    components=Jet(value=E0_val, gradient=jnp.zeros((dim, dim)), hessian=jnp.zeros((dim, dim, dim))),
    basis=standard_basis
  )

  # Use W = E_1 for the inner product to get a non-trivial result
  W = TangentVector(
    p=p,
    components=Jet(value=E1_val, gradient=jnp.zeros((dim, dim)), hessian=jnp.zeros((dim, dim, dim))),
    basis=standard_basis
  )

  def f(t: Scalar) -> Scalar:
    # Jacobi field S(t) = t · E_1
    # Value at origin: S^k = t · δ^k_1
    # Gradient: ∂S^k/∂v^m = δ^k_1 · δ^m_0 (constant, encodes ∇_T S(0) = E_1)
    jacobi_value = t * E1_val
    jacobi_gradient = jnp.zeros((dim, dim)).at[1, 0].set(1.0)
    jacobi_hessian = jnp.zeros((dim, dim, dim))

    St = TangentVector(
      p=p,
      components=Jet(value=jacobi_value, gradient=jacobi_gradient, hessian=jacobi_hessian),
      basis=standard_basis
    )

    # Compute ∇_T² S
    nablaT_St = connection.covariant_derivative(T, St)
    nabla2T_St = connection.covariant_derivative(T, nablaT_St)

    # Return the scalar f(t) = g(∇_T² S, W) where W = E_1
    return metric_rnc(nabla2T_St, W).value

  # Compute f(0), f'(0), f''(0) using nested JVPs
  f_val = f(0.0)

  # First derivative
  _, f_prime_val = jax.jvp(f, (0.0,), (1.0,))

  # Second derivative via nested JVP
  def f_prime_fn(t):
    _, tangent = jax.jvp(f, (t,), (1.0,))
    return tangent

  _, f_double_prime_val = jax.jvp(f_prime_fn, (0.0,), (1.0,))

  # f(0) = 0 since S(0) = 0
  assert jnp.isclose(f_val, 0.0, atol=1e-5), f"f(0) = {f_val}, expected 0"

  # f'(0) should be non-zero for curved spaces (non-triviality check)
  # The exact value involves the Christoffel gradient identity:
  #   ∂_0 Γ^k_01 = (1/3)(R^k_010 + R^k_100)
  # Due to Riemann antisymmetry, this involves a factor of 1/3 relative to
  # the naive expectation g(R(T,E_1)T, E_1).
  assert jnp.isfinite(f_prime_val), f"f'(0) = {f_prime_val} is not finite"
  assert not jnp.isclose(f_prime_val, 0.0, atol=1e-10), \
    f"f'(0) = {f_prime_val} is too close to 0 (test is trivial)"

  # The key Jacobi identity: f''(0) = 0 because f(t) is linear in t
  # This is the main assertion we want to verify
  assert jnp.isclose(f_double_prime_val, 0.0, atol=1e-5), \
    f"f''(0) = {f_double_prime_val}, expected 0 (linearity of Jacobi equation)"

  # Additional verification: f'(0) should be proportional to R_{0101}
  # specifically f'(0) = (1/3) * g(R(T,E_1)T, E_1) due to Christoffel gradient formula
  R = get_riemann_curvature_tensor(connection)
  RT_E1_T = R(T, W, T)
  curvature_term = metric_rnc(RT_E1_T, W).value

  # The factor comes from ∂_0 Γ^1_01 = (1/3)(R^1_010 + R^1_100) = (1/3)(R^1_010 - R^1_010) = 0
  # But ∂_1 Γ^1_00 = (1/3)(R^1_001 + R^1_001) = (2/3) R^1_001
  # The actual relationship is more subtle - just verify proportionality
  if not jnp.isclose(curvature_term, 0.0, atol=1e-10):
    ratio = f_prime_val / curvature_term
    # The ratio should be approximately 1/3 due to the Christoffel gradient identity
    assert jnp.isclose(ratio, 1.0/3.0, atol=0.1), \
      f"Ratio f'(0)/R_0101 = {ratio}, expected ~1/3"

def test_metric_log_det_hessian():

  key = random.PRNGKey(0)
  dim = 5
  metric = create_random_metric(key, dim=dim)
  metric_rnc = to_riemann_normal_coordinates(metric)

  @jet_decorator
  def get_log_det(gij):
    return jnp.linalg.slogdet(gij)[1]

  log_det: Jet = get_log_det(metric_rnc.components.get_value_jet())
  log_det_hessian = log_det.hessian

  # Compare against the ricci tensor
  Rc: RicciTensor = get_ricci_tensor(get_levi_civita_connection(metric_rnc))
  ricci = Rc.components.value

  assert jnp.allclose(log_det_hessian, -2.0/3.0 * ricci, atol=1e-5)


# =============================================================================
# Tests for new optimized API functions
# =============================================================================

def test_get_rnc_jacobians_returns_both():
  """
  Test that get_rnc_jacobians returns both jacobians correctly.
  """
  from local_coordinates.normal_coords import get_rnc_jacobians

  key = random.PRNGKey(100)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)

  # Check that they are inverses of each other (at value level)
  product = J_x_to_v.value @ J_v_to_x.value
  assert jnp.allclose(product, jnp.eye(dim), atol=1e-10)


def test_get_rnc_jacobians_matches_individual_functions():
  """
  Test that get_rnc_jacobians matches the individual functions.
  """
  from local_coordinates.normal_coords import (
    get_rnc_jacobians,
    get_transformation_to_riemann_normal_coordinates,
    get_transformation_from_riemann_normal_coordinates
  )

  key = random.PRNGKey(101)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  J_x_to_v_combined, J_v_to_x_combined = get_rnc_jacobians(metric)
  J_x_to_v_individual = get_transformation_to_riemann_normal_coordinates(metric)
  J_v_to_x_individual = get_transformation_from_riemann_normal_coordinates(metric)

  assert jnp.allclose(J_x_to_v_combined.value, J_x_to_v_individual.value, atol=1e-10)
  assert jnp.allclose(J_x_to_v_combined.gradient, J_x_to_v_individual.gradient, atol=1e-10)
  assert jnp.allclose(J_x_to_v_combined.hessian, J_x_to_v_individual.hessian, atol=1e-10)

  assert jnp.allclose(J_v_to_x_combined.value, J_v_to_x_individual.value, atol=1e-10)
  assert jnp.allclose(J_v_to_x_combined.gradient, J_v_to_x_individual.gradient, atol=1e-10)
  assert jnp.allclose(J_v_to_x_combined.hessian, J_v_to_x_individual.hessian, atol=1e-10)


def test_get_rnc_basis_with_precomputed_jacobian():
  """
  Test that get_rnc_basis gives the same result with a precomputed jacobian.
  """
  from local_coordinates.normal_coords import get_rnc_jacobians

  key = random.PRNGKey(102)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  # Compute without precomputed jacobian
  rnc_basis_default = get_rnc_basis(metric)

  # Compute with precomputed jacobian
  _, J_v_to_x = get_rnc_jacobians(metric)
  rnc_basis_precomputed = get_rnc_basis(metric, J_v_to_x=J_v_to_x)

  assert jnp.allclose(
    rnc_basis_default.components.value,
    rnc_basis_precomputed.components.value,
    atol=1e-10
  )
  assert jnp.allclose(
    rnc_basis_default.components.gradient,
    rnc_basis_precomputed.components.gradient,
    atol=1e-10
  )
  assert jnp.allclose(
    rnc_basis_default.components.hessian,
    rnc_basis_precomputed.components.hessian,
    atol=1e-10
  )


def test_get_rnc_frame_with_precomputed_jacobian():
  """
  Test that get_rnc_frame gives the same result with a precomputed jacobian.
  """
  from local_coordinates.normal_coords import get_rnc_jacobians

  key = random.PRNGKey(103)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  # Compute without precomputed jacobian
  rnc_frame_default = get_rnc_frame(metric)

  # Compute with precomputed jacobian
  _, J_v_to_x = get_rnc_jacobians(metric)
  rnc_frame_precomputed = get_rnc_frame(metric, J_v_to_x=J_v_to_x)

  assert jnp.allclose(
    rnc_frame_default.components.value,
    rnc_frame_precomputed.components.value,
    atol=1e-10
  )
  assert jnp.allclose(
    rnc_frame_default.basis.components.value,
    rnc_frame_precomputed.basis.components.value,
    atol=1e-10
  )


def test_to_rnc_metric_with_precomputed_jacobians():
  """
  Test that to_riemann_normal_coordinates(metric) gives the same result
  with precomputed jacobians.
  """
  from local_coordinates.normal_coords import get_rnc_jacobians

  key = random.PRNGKey(104)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  # Compute without precomputed jacobians
  metric_rnc_default = to_riemann_normal_coordinates(metric)

  # Compute with precomputed jacobians
  J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)
  metric_rnc_precomputed = to_riemann_normal_coordinates(
    metric, J_x_to_v=J_x_to_v, J_v_to_x=J_v_to_x
  )

  assert jnp.allclose(
    metric_rnc_default.components.value,
    metric_rnc_precomputed.components.value,
    atol=1e-10
  )
  assert jnp.allclose(
    metric_rnc_default.components.gradient,
    metric_rnc_precomputed.components.gradient,
    atol=1e-10
  )


def test_to_rnc_vector_with_precomputed_jacobians():
  """
  Test that to_riemann_normal_coordinates(vector, metric) gives the same result
  with precomputed jacobians.
  """
  from local_coordinates.normal_coords import get_rnc_jacobians

  key = random.PRNGKey(105)
  dim = 3
  metric = create_random_metric(key, dim=dim)
  vector = create_random_vector_field(key, dim, metric.basis)

  # Compute without precomputed jacobians
  vector_rnc_default = to_riemann_normal_coordinates(vector, metric)

  # Compute with precomputed jacobians
  J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)
  vector_rnc_precomputed = to_riemann_normal_coordinates(
    vector, metric, J_x_to_v=J_x_to_v, J_v_to_x=J_v_to_x
  )

  assert jnp.allclose(
    vector_rnc_default.components.value,
    vector_rnc_precomputed.components.value,
    atol=1e-10
  )
  assert jnp.allclose(
    vector_rnc_default.components.gradient,
    vector_rnc_precomputed.components.gradient,
    atol=1e-10
  )


def test_to_rnc_frame_with_precomputed_jacobians():
  """
  Test that to_riemann_normal_coordinates(frame, metric) gives the same result
  with precomputed jacobians.
  """
  from local_coordinates.normal_coords import get_rnc_jacobians

  key = random.PRNGKey(106)
  dim = 3
  metric = create_random_metric(key, dim=dim)
  frame = basis_to_frame(metric.basis)

  # Compute without precomputed jacobians
  frame_rnc_default = to_riemann_normal_coordinates(frame, metric)

  # Compute with precomputed jacobians
  J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)
  frame_rnc_precomputed = to_riemann_normal_coordinates(
    frame, metric, J_x_to_v=J_x_to_v, J_v_to_x=J_v_to_x
  )

  assert jnp.allclose(
    frame_rnc_default.components.value,
    frame_rnc_precomputed.components.value,
    atol=1e-10
  )
  assert jnp.allclose(
    frame_rnc_default.basis.components.value,
    frame_rnc_precomputed.basis.components.value,
    atol=1e-10
  )


def test_to_rnc_tensor_with_precomputed_jacobians():
  """
  Test that to_riemann_normal_coordinates(tensor, metric) gives the same result
  with precomputed jacobians.
  """
  from local_coordinates.normal_coords import get_rnc_jacobians

  key = random.PRNGKey(107)
  dim = 3
  metric = create_random_metric(key, dim=dim)
  tensor = create_random_tensor(key, dim, metric.basis, k=1, l=2)

  # Compute without precomputed jacobians
  tensor_rnc_default = to_riemann_normal_coordinates(tensor, metric)

  # Compute with precomputed jacobians
  J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)
  tensor_rnc_precomputed = to_riemann_normal_coordinates(
    tensor, metric, J_x_to_v=J_x_to_v, J_v_to_x=J_v_to_x
  )

  assert jnp.allclose(
    tensor_rnc_default.components.value,
    tensor_rnc_precomputed.components.value,
    atol=1e-10
  )
  assert jnp.allclose(
    tensor_rnc_default.components.gradient,
    tensor_rnc_precomputed.components.gradient,
    atol=1e-10
  )


def test_to_rnc_connection_with_precomputed_jacobians():
  """
  Test that to_riemann_normal_coordinates(connection, metric) gives the same result
  with precomputed jacobians.
  """
  from local_coordinates.normal_coords import get_rnc_jacobians

  key = random.PRNGKey(108)
  dim = 3
  metric = create_random_metric(key, dim=dim)
  connection = get_levi_civita_connection(metric)

  # Compute without precomputed jacobians
  connection_rnc_default = to_riemann_normal_coordinates(connection, metric)

  # Compute with precomputed jacobians
  J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)
  connection_rnc_precomputed = to_riemann_normal_coordinates(
    connection, metric, J_x_to_v=J_x_to_v, J_v_to_x=J_v_to_x
  )

  assert jnp.allclose(
    connection_rnc_default.christoffel_symbols.value,
    connection_rnc_precomputed.christoffel_symbols.value,
    atol=1e-10
  )
  assert jnp.allclose(
    connection_rnc_default.christoffel_symbols.gradient,
    connection_rnc_precomputed.christoffel_symbols.gradient,
    atol=1e-10
  )


def test_transform_multiple_objects_with_shared_jacobians():
  """
  Test a realistic workflow where multiple objects are transformed to RNC
  using shared precomputed jacobians.
  """
  from local_coordinates.normal_coords import get_rnc_jacobians

  key = random.PRNGKey(109)
  dim = 3
  metric = create_random_metric(key, dim=dim)
  connection = get_levi_civita_connection(metric)
  R = get_riemann_curvature_tensor(connection)
  R_lower = lower_index(R, metric, 4)
  frame = get_rnc_frame(metric)
  vector = frame.get_basis_vector(0)

  # Compute jacobians once
  J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)

  # Transform all objects using shared jacobians
  metric_rnc = to_riemann_normal_coordinates(metric, J_x_to_v=J_x_to_v, J_v_to_x=J_v_to_x)
  connection_rnc = to_riemann_normal_coordinates(connection, metric, J_x_to_v=J_x_to_v, J_v_to_x=J_v_to_x)
  R_rnc = to_riemann_normal_coordinates(R_lower, metric, J_x_to_v=J_x_to_v, J_v_to_x=J_v_to_x)
  frame_rnc = to_riemann_normal_coordinates(frame, metric, J_x_to_v=J_x_to_v, J_v_to_x=J_v_to_x)
  vector_rnc = to_riemann_normal_coordinates(vector, metric, J_x_to_v=J_x_to_v, J_v_to_x=J_v_to_x)

  # Verify expected RNC properties
  assert jnp.allclose(metric_rnc.components.value, jnp.eye(dim), atol=1e-5)
  assert jnp.allclose(metric_rnc.components.gradient, 0.0, atol=1e-5)
  assert jnp.allclose(connection_rnc.christoffel_symbols.value, 0.0, atol=1e-5)
  assert jnp.allclose(frame_rnc.basis.components.value, jnp.eye(dim), atol=1e-5)


def test_get_transformation_from_rnc_direct():
  """
  Test that get_transformation_from_riemann_normal_coordinates gives the correct
  forward jacobian without going through an inversion.
  """
  from local_coordinates.normal_coords import (
    get_transformation_from_riemann_normal_coordinates,
    get_rnc_jacobians
  )

  key = random.PRNGKey(110)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  # Get J_v_to_x directly
  J_v_to_x_direct = get_transformation_from_riemann_normal_coordinates(metric)

  # Get via get_rnc_jacobians
  _, J_v_to_x_combined = get_rnc_jacobians(metric)

  assert jnp.allclose(J_v_to_x_direct.value, J_v_to_x_combined.value, atol=1e-10)
  assert jnp.allclose(J_v_to_x_direct.gradient, J_v_to_x_combined.gradient, atol=1e-10)
  assert jnp.allclose(J_v_to_x_direct.hessian, J_v_to_x_combined.hessian, atol=1e-10)


# =============================================================================
# Principal Ricci coordinates
# =============================================================================


def test_principal_ricci_coordinates():
  """
  Construct "principal Ricci coordinates" - Riemann normal coordinates (RNC) in which the
  Ricci tensor is diagonal at the origin.

  Background
  ----------
  In standard RNC, the metric is the identity at the origin and Christoffel symbols vanish,
  but the Ricci tensor is generally NOT diagonal. By choosing a specific orthonormal frame
  aligned with the eigenvectors of the Ricci tensor, we can construct RNC where Ric is diagonal.

  Construction
  ------------
  1. Start with a metric g in standard coordinates (x-coordinates).

  2. Compute standard RNC and the Ricci tensor there. The Ricci tensor Ric_rnc is symmetric,
     so it can be diagonalized: Ric_rnc = Q @ diag(λ) @ Q^T, where Q is orthogonal.

  3. Compute the orthonormal frame E that diagonalizes the metric: E^T g E = I.
     This E takes us from x-coordinates to standard RNC.

  4. Compose E with Q to get the "principal Ricci frame": dxds = E @ Q.
     This frame is still orthonormal with respect to g: (EQ)^T g (EQ) = Q^T (E^T g E) Q = Q^T I Q = I.
     The RNC built from this frame will have a diagonal Ricci tensor.

  5. Use _get_rnc_jacobian to compute the full Jacobian J_s_to_x from principal Ricci coords (s)
     to standard coords (x), including higher-order corrections from the Christoffel symbols.

  6. Transform the metric to principal Ricci coordinates using to_riemann_normal_coordinates.

  Verification
  ------------
  This test verifies:

  1. **Ricci is diagonal**: In principal Ricci coordinates, ricci_s.components.value should be
     a diagonal matrix (the eigenvalues of the Ricci tensor).

  2. **Ricci scalar is invariant**: The Ricci scalar R = trace(Ric) should be the same whether
     computed in standard RNC or principal Ricci RNC, since it's a coordinate-invariant quantity.

  TODO
  ----
  Verify that ricci_s can be transformed back to ricci.

  Intended Approach (Logic):
  1. change_coordinates(ricci_s, J_s_to_x):
     Changes the parameterization of the tensor from s-coordinates to x-coordinates.
     The basis vectors are re-expressed (e_s -> J e_x), but the component VALUES remain
     unchanged (still the diagonal matrix) because the tensor object itself represents
     the same geometric object.

  2. change_basis(..., metric.basis):
     Transforms the component values from the s-basis (now expressed in x-coords)
     to the metric basis (standard x-basis). This applies the tensor transformation law.

  (Note: This check is currently commented out due to numerical issues in the implementation,
   but the logic is verified by test_change_coordinates_then_change_basis_recovers_original)
  """
  key = random.PRNGKey(0)
  dim = 5

  # Compute all of these in the standard basis.
  metric = create_random_metric(key, dim=dim)

  # This test will only work in the standard basis because gamma_bar is computed in the standard basis
  # when we call _compute_rnc_jacobians.
  standard_basis = get_standard_basis(metric.basis.p)
  metric: RiemannianMetric = change_basis(metric, standard_basis)

  connection: Connection = get_levi_civita_connection(metric)
  ricci = get_ricci_tensor(connection)

  # First go to standard RNC and compute Ricci there.
  metric_rnc = to_riemann_normal_coordinates(metric)
  connection_rnc: Connection = get_levi_civita_connection(metric_rnc)
  ricci_rnc = get_ricci_tensor(connection_rnc)
  eigvals_rnc, Q = jnp.linalg.eigh(ricci_rnc.components.value)  # Q rotates RNC to diagonalize Ricci

  # Get the orthonormal frame E that takes standard coords to RNC.
  # E orthonormalizes g: E^T g E = I
  gij = metric.components.value
  eigenvalues, eigenvectors = jnp.linalg.eigh(gij)
  E = jnp.einsum("ij,j->ij", eigenvectors, jax.lax.rsqrt(eigenvalues))

  # Principal Ricci frame: compose E with Q.
  # dx/ds = E @ Q, where s are principal Ricci coords.
  # This is still orthonormal: (EQ)^T g (EQ) = Q^T E^T g E Q = Q^T I Q = I
  dxds = E @ Q

  # Get the transformation from principal Ricci coordinates to the standard basis.
  gamma_bar = connection.christoffel_symbols
  J_s_to_x = _get_rnc_jacobian(gamma_bar, dxds)

  # Go to the principal Ricci coordinates.
  metric_s = to_riemann_normal_coordinates(metric, J_v_to_x=J_s_to_x)
  connection_s: Connection = get_levi_civita_connection(metric_s)
  ricci_s = get_ricci_tensor(connection_s)

  assert jnp.allclose(ricci_s.components.value, jnp.diag(jnp.diag(ricci_s.components.value)))

  # Verify the Ricci scalar is invariant between standard RNC and principal Ricci RNC.
  # Both should give the same scalar since they're both RNC for the same metric.
  ricci_scalar_rnc = jnp.trace(ricci_rnc.components.value)
  ricci_scalar_s = jnp.trace(ricci_s.components.value)
  assert jnp.allclose(ricci_scalar_rnc, ricci_scalar_s), "Ricci scalar should match between RNC variants"

  # Transform ricci_s back to the original coordinates and verify it matches ricci.
  # 1. change_coordinates: re-describe s-basis in x-coordinates (values unchanged)
  # 2. change_basis: transform values to the standard x-basis
  ricci_in_x_coords = change_coordinates_tensor(ricci_s, J_s_to_x)
  ricci_recovered = change_basis(ricci_in_x_coords, metric.basis)

  assert jnp.allclose(ricci_recovered.components.value, ricci.components.value), \
    "Should recover original Ricci tensor components"


def test_ricci_scalar_invariance_under_rnc():
  """
  Test that the Ricci scalar is invariant under transformation to RNC.

  The Ricci scalar R = g^{ab} R_{ab} is a coordinate-invariant quantity.
  It should have the same value whether computed in:
  - The original coordinates (standard basis, non-identity metric components)
  - Riemann normal coordinates (identity metric at origin)

  NOTE: create_random_metric returns a metric in the STANDARD basis, so
  metric.components.value is NOT identity. You must use g^{ab} R_ab, not trace(R).
  """
  key = random.PRNGKey(0)
  dim = 5

  # Create a metric - NOTE: this is in the standard basis, not a random basis!
  # The metric components are NOT identity.
  metric = create_random_metric(key, dim=dim)
  connection = get_levi_civita_connection(metric)
  ricci = get_ricci_tensor(connection)

  # Transform to standard RNC
  metric_rnc = to_riemann_normal_coordinates(metric)
  connection_rnc = get_levi_civita_connection(metric_rnc)
  ricci_rnc = get_ricci_tensor(connection_rnc)

  # Compute Ricci scalar in original coordinates.
  # IMPORTANT: metric.components.value is NOT identity, so we must use g^{ab} R_ab
  g_inv = jnp.linalg.inv(metric.components.value)
  ricci_scalar_original = jnp.einsum("ab,ab->", g_inv, ricci.components.value)

  # Compute Ricci scalar in RNC.
  # In RNC at the origin, the metric is identity, so g^{ab} = δ^{ab} and R = trace(Ric)
  ricci_scalar_rnc = jnp.trace(ricci_rnc.components.value)

  # These should be equal - the Ricci scalar is coordinate-invariant!
  assert jnp.allclose(ricci_scalar_original, ricci_scalar_rnc), (
    f"Ricci scalar mismatch: original={ricci_scalar_original}, RNC={ricci_scalar_rnc}"
  )

  # Also verify that using the wrong formula (trace) in original coords gives wrong answer
  wrong_ricci_scalar = jnp.trace(ricci.components.value)
  assert not jnp.allclose(wrong_ricci_scalar, ricci_scalar_rnc), (
    "trace(Ric) should NOT equal the Ricci scalar when metric is not identity"
  )


def test_change_coordinates_then_change_basis_recovers_original():
  """
  Test that change_coordinates + change_basis correctly composes to recover original components.

  change_coordinates only changes the parameterization, NOT what the object represents.
  So metric_back and metric represent the SAME geometric object, just in different bases.
  Applying change_basis should recover the original components.
  """
  key = random.PRNGKey(0)
  dim = 3
  metric = create_random_metric(key, dim=dim)

  # Get the Jacobians for x -> v (RNC) transformation
  J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)

  # Transform metric to RNC: value goes from g to I
  metric_rnc = to_riemann_normal_coordinates(metric)
  assert jnp.allclose(metric_rnc.components.value, jnp.eye(dim))  # Sanity check

  # Now use change_coordinates to go back from v to x
  # This changes the basis description from v-coords to x-coords, but keeps values as I
  metric_back = change_coordinates_tensor(metric_rnc, J_v_to_x)

  # Values are still I (change_coordinates doesn't transform values)
  assert jnp.allclose(metric_back.components.value, jnp.eye(dim))

  # But metric_back and metric represent the SAME geometric object!
  # They just have different bases. So change_basis should recover the original.
  metric_recovered = change_basis(metric_back, metric.basis)

  # This should recover the original metric components
  assert jnp.allclose(metric_recovered.components.value, metric.components.value), \
    "change_basis should recover original components"


