"""
Unit tests for the exponential map module.
"""
import jax
import jax.numpy as jnp
from jax import random
import pytest

from local_coordinates.basis import BasisVectors, get_standard_basis
from local_coordinates.jet import Jet, get_identity_jet, function_to_jet
from local_coordinates.metric import RiemannianMetric
from local_coordinates.tensor import change_basis
from local_coordinates.normal_coords import get_rnc_jacobians
from local_coordinates.tangent import TangentVector
from local_coordinates.exponential_map import (
  exponential_map_taylor,
  exponential_map_ode,
  exponential_map,
  logarithmic_map_taylor,
)
from typing import Callable


def make_metric_fn(g_components_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], RiemannianMetric]:
  """
  Convert a function that returns metric components g_ij(x) into a function
  that returns a full RiemannianMetric object with value, gradient, and hessian.
  """
  def metric_fn(x: jnp.ndarray) -> RiemannianMetric:
    basis = get_standard_basis(x)
    components = function_to_jet(g_components_fn, x)
    return RiemannianMetric(basis=basis, components=components)
  return metric_fn


def create_tangent_vector(p: jnp.ndarray, v: jnp.ndarray) -> TangentVector:
  """Create a TangentVector at point p with velocity v in standard coordinates."""
  dim = p.shape[0]
  basis = get_standard_basis(p)
  # Components as a Jet (we only need the value for ODE, but Jet requires gradient)
  components = Jet(value=v, gradient=jnp.zeros((dim, dim)), hessian=jnp.zeros((dim, dim, dim)))
  return TangentVector(p=p, components=components, basis=basis)


def create_random_basis(key: random.PRNGKey, dim: int, p: jnp.ndarray = None) -> BasisVectors:
  """Create a random basis for testing."""
  vals_key, grads_key, hessians_key = random.split(key, 3)
  if p is None:
    p = jnp.zeros(dim)
  vals = jnp.eye(dim) + random.normal(vals_key, (dim, dim)) * 0.1
  grads = random.normal(grads_key, (dim, dim, dim)) * 0.1
  hessians = random.normal(hessians_key, (dim, dim, dim, dim)) * 0.1
  return BasisVectors(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians))


def create_random_metric(key: random.PRNGKey, dim: int, p: jnp.ndarray = None) -> RiemannianMetric:
  """Create a random metric for testing."""
  if p is None:
    p = jnp.zeros(dim)
  random_basis = create_random_basis(key, dim, p)
  metric = RiemannianMetric(basis=random_basis, components=get_identity_jet(dim))
  standard_basis = get_standard_basis(random_basis.p)
  return change_basis(metric, standard_basis)


def create_nearly_flat_metric(p: jnp.ndarray, epsilon: float = 0.1) -> RiemannianMetric:
  """
  Create a nearly-flat metric g_ij = delta_ij + epsilon * x_i * x_j.

  This is a simple perturbation of the Euclidean metric that still has
  non-trivial curvature.
  """
  dim = p.shape[0]

  # g_ij = delta_ij + epsilon * x_i * x_j
  g_val = jnp.eye(dim) + epsilon * jnp.outer(p, p)

  # d g_ij / dx^k = epsilon * (delta_ik * x_j + x_i * delta_jk)
  g_grad = jnp.zeros((dim, dim, dim))
  for k in range(dim):
    for i in range(dim):
      for j in range(dim):
        g_grad = g_grad.at[i, j, k].set(
          epsilon * ((i == k) * p[j] + p[i] * (j == k))
        )

  # d^2 g_ij / dx^k dx^l = epsilon * (delta_ik * delta_jl + delta_il * delta_jk)
  g_hess = jnp.zeros((dim, dim, dim, dim))
  for k in range(dim):
    for l in range(dim):
      for i in range(dim):
        for j in range(dim):
          g_hess = g_hess.at[i, j, k, l].set(
            epsilon * ((i == k) * (j == l) + (i == l) * (j == k))
          )

  basis = get_standard_basis(p)
  components = Jet(value=g_val, gradient=g_grad, hessian=g_hess)
  return RiemannianMetric(basis=basis, components=components)


def create_euclidean_metric(p: jnp.ndarray) -> RiemannianMetric:
  """Create a flat Euclidean metric g_ij = delta_ij."""
  dim = p.shape[0]

  g_val = jnp.eye(dim)
  g_grad = jnp.zeros((dim, dim, dim))
  g_hess = jnp.zeros((dim, dim, dim, dim))

  basis = get_standard_basis(p)
  components = Jet(value=g_val, gradient=g_grad, hessian=g_hess)
  return RiemannianMetric(basis=basis, components=components)


# =============================================================================
# Tests for exponential_map_taylor
# =============================================================================

def test_exponential_map_taylor_at_zero():
  """exp_p(0) should return p."""
  p = jnp.array([1.0, 2.0])
  metric = create_nearly_flat_metric(p, epsilon=0.1)

  v = create_tangent_vector(p, jnp.zeros(2))
  q = exponential_map_taylor(metric, v)

  assert jnp.allclose(q, p, atol=1e-10)


def test_exponential_map_taylor_euclidean():
  """For flat metric, exp_p(v) = p + v."""
  p = jnp.array([1.0, 2.0, 0.5])
  metric = create_euclidean_metric(p)

  v_arr = jnp.array([0.1, -0.2, 0.3])
  v = create_tangent_vector(p, v_arr)
  q = exponential_map_taylor(metric, v)

  # For Euclidean metric, exp_p(v) = p + v
  expected = p + v_arr
  assert jnp.allclose(q, expected, atol=1e-6)


def test_exponential_map_taylor_small_displacement():
  """For small v, exp_p(v) ≈ p + v (linear approximation)."""
  p = jnp.array([1.0, 0.5])
  metric = create_nearly_flat_metric(p, epsilon=0.1)

  v_arr = jnp.array([0.001, -0.002])
  v = create_tangent_vector(p, v_arr)
  q = exponential_map_taylor(metric, v)

  # Linear approximation: exp_p(v) ≈ p + v for small v
  q_linear = p + v_arr

  # Should agree to high precision for small v
  assert jnp.allclose(q, q_linear, atol=1e-6)


def test_exponential_map_taylor_different_dimensions():
  """Test that the exponential map works for various dimensions."""
  for dim in [2, 3, 4]:
    key = random.PRNGKey(dim)
    p = random.normal(key, (dim,))
    metric = create_random_metric(key, dim, p)

    v_arr = jnp.ones(dim) * 0.1
    v = create_tangent_vector(p, v_arr)
    q = exponential_map_taylor(metric, v)

    assert q.shape == (dim,)
    # q should be close to p for small v
    assert jnp.linalg.norm(q - p) < 1.0


# =============================================================================
# Tests for logarithmic_map_taylor
# =============================================================================

def test_logarithmic_map_taylor_at_p():
  """log_p(p) should return 0."""
  p = jnp.array([1.0, 2.0])
  metric = create_nearly_flat_metric(p, epsilon=0.1)

  v = logarithmic_map_taylor(metric, p)

  assert jnp.allclose(v, jnp.zeros(2), atol=1e-10)


def test_logarithmic_map_taylor_euclidean():
  """For flat metric, log_p(q) = J^{-1} @ (q - p)."""
  p = jnp.array([1.0, 2.0, 0.5])
  metric = create_euclidean_metric(p)

  q = jnp.array([1.1, 1.8, 0.8])
  v = logarithmic_map_taylor(metric, q)

  # For Euclidean metric, log_p(q) = q - p
  expected = q - p
  assert jnp.allclose(v, expected, atol=1e-6)


# =============================================================================
# Tests for exp/log roundtrip
# =============================================================================

def test_exp_log_roundtrip_small_v():
  """log_p(exp_p(v)) should approximately recover v (in RNC) for small v."""
  p = jnp.array([1.0, 0.5])
  metric = create_nearly_flat_metric(p, epsilon=0.1)

  # Create tangent vector in standard coords
  v_std = jnp.array([0.1, -0.05])
  v = create_tangent_vector(p, v_std)
  q = exponential_map_taylor(metric, v)

  # log returns v in RNC, so convert our input to RNC for comparison
  J_x_to_v, _ = get_rnc_jacobians(metric)
  v_rnc = J_x_to_v.value @ v_std

  v_recovered = logarithmic_map_taylor(metric, q)
  assert jnp.allclose(v_rnc, v_recovered, atol=1e-5)


def test_exp_log_roundtrip_random_metric():
  """Test roundtrip for randomly generated metrics."""
  for seed in range(3):
    key = random.PRNGKey(seed)
    dim = 3
    p = random.normal(key, (dim,)) * 0.5
    metric = create_random_metric(key, dim, p)

    v_std = random.normal(key, (dim,)) * 0.1
    v = create_tangent_vector(p, v_std)
    q = exponential_map_taylor(metric, v)

    # Convert to RNC for comparison
    J_x_to_v, _ = get_rnc_jacobians(metric)
    v_rnc = J_x_to_v.value @ v_std

    v_recovered = logarithmic_map_taylor(metric, q)
    assert jnp.allclose(v_rnc, v_recovered, atol=1e-4), f"Roundtrip failed for seed {seed}"


def test_log_exp_roundtrip():
  """exp_p(log_p(q)) should approximately recover q for q near p."""
  p = jnp.array([1.0, 0.5])
  metric = create_nearly_flat_metric(p, epsilon=0.1)

  # q is close to p
  q = jnp.array([1.05, 0.52])
  v_rnc = logarithmic_map_taylor(metric, q)

  # Convert v_rnc to standard coords for TangentVector
  _, J_v_to_x = get_rnc_jacobians(metric)
  v_std = J_v_to_x.value @ v_rnc
  v = create_tangent_vector(p, v_std)

  q_recovered = exponential_map_taylor(metric, v)
  assert jnp.allclose(q, q_recovered, atol=1e-5)


# =============================================================================
# Tests for exponential_map_ode
# =============================================================================

def _euclidean_g_components(x: jnp.ndarray) -> jnp.ndarray:
  """Euclidean metric components g_ij = delta_ij."""
  dim = x.shape[0]
  return jnp.eye(dim)

_euclidean_metric_fn = make_metric_fn(_euclidean_g_components)


def _nearly_flat_metric_fn(epsilon: float = 0.1):
  """Factory for nearly-flat metric g_ij = delta_ij + epsilon * x_i * x_j."""
  def g_components(x: jnp.ndarray) -> jnp.ndarray:
    dim = x.shape[0]
    return jnp.eye(dim) + epsilon * jnp.outer(x, x)
  return make_metric_fn(g_components)


def test_exponential_map_ode_at_zero():
  """ODE solver: exp_p(0) should return p."""
  p = jnp.array([1.0, 2.0])
  metric_fn = _nearly_flat_metric_fn(epsilon=0.1)

  v = create_tangent_vector(p, jnp.zeros(2))
  q = exponential_map_ode(v, metric_fn)

  assert jnp.allclose(q, p, atol=1e-6)


def test_exponential_map_ode_euclidean():
  """For flat metric, ODE solver should give exp_p(v) = p + v."""
  p = jnp.array([1.0, 2.0])

  v = create_tangent_vector(p, jnp.array([0.1, -0.2]))
  q = exponential_map_ode(v, _euclidean_metric_fn)

  expected = p + jnp.array([0.1, -0.2])
  assert jnp.allclose(q, expected, atol=1e-4)


def test_taylor_ode_agreement_small_v():
  """Taylor and ODE should agree for small displacements."""
  p = jnp.array([1.0, 0.5])
  epsilon = 0.1
  metric = create_nearly_flat_metric(p, epsilon=epsilon)
  metric_fn = _nearly_flat_metric_fn(epsilon=epsilon)

  v_std = jnp.array([0.05, -0.03])
  v = create_tangent_vector(p, v_std)

  q_taylor = exponential_map_taylor(metric, v)
  q_ode = exponential_map_ode(v, metric_fn)

  # They should agree reasonably well for small v
  assert jnp.allclose(q_taylor, q_ode, atol=1e-3)


# =============================================================================
# Tests for exponential_map dispatcher
# =============================================================================

def test_exponential_map_dispatch_taylor():
  """Test that dispatch to Taylor method works."""
  p = jnp.array([1.0, 0.5])
  metric = create_nearly_flat_metric(p, epsilon=0.1)

  v = create_tangent_vector(p, jnp.array([0.1, -0.05]))
  q1 = exponential_map(metric, v, method="taylor")
  q2 = exponential_map_taylor(metric, v)

  assert jnp.allclose(q1, q2)


def test_exponential_map_dispatch_ode():
  """Test that dispatch to ODE method works."""
  p = jnp.array([1.0, 0.5])
  epsilon = 0.1
  metric = create_nearly_flat_metric(p, epsilon=epsilon)
  metric_fn = _nearly_flat_metric_fn(epsilon=epsilon)

  v = create_tangent_vector(p, jnp.array([0.1, -0.05]))
  q1 = exponential_map(metric, v, method="ode", metric_fn=metric_fn)
  q2 = exponential_map_ode(v, metric_fn)

  assert jnp.allclose(q1, q2)


def test_exponential_map_invalid_method():
  """Test that invalid method raises ValueError."""
  p = jnp.array([1.0, 0.5])
  metric = create_euclidean_metric(p)
  v = create_tangent_vector(p, jnp.array([0.1, -0.05]))

  with pytest.raises(ValueError, match="Unknown method"):
    exponential_map(metric, v, method="invalid")


def test_exponential_map_ode_requires_metric_fn():
  """Test that ODE method requires metric_fn argument."""
  p = jnp.array([1.0, 0.5])
  metric = create_euclidean_metric(p)
  v = create_tangent_vector(p, jnp.array([0.1, -0.05]))

  with pytest.raises(ValueError, match="requires 'metric_fn'"):
    exponential_map(metric, v, method="ode")


# =============================================================================
# Tests for pre-computed Jacobian
# =============================================================================

def test_exponential_map_with_precomputed_jacobian():
  """Test that passing a pre-computed Jacobian works correctly."""
  p = jnp.array([1.0, 0.5])
  metric = create_nearly_flat_metric(p, epsilon=0.1)

  J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)

  v = create_tangent_vector(p, jnp.array([0.1, -0.05]))

  q1 = exponential_map_taylor(metric, v)
  q2 = exponential_map_taylor(metric, v, J_v_to_x=J_v_to_x)

  assert jnp.allclose(q1, q2)


def test_logarithmic_map_with_precomputed_jacobian():
  """Test that passing a pre-computed Jacobian works correctly."""
  p = jnp.array([1.0, 0.5])
  metric = create_nearly_flat_metric(p, epsilon=0.1)

  J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)

  q = jnp.array([1.05, 0.52])

  v1 = logarithmic_map_taylor(metric, q)
  v2 = logarithmic_map_taylor(metric, q, J_x_to_v=J_x_to_v)

  assert jnp.allclose(v1, v2)


# =============================================================================
# Tests for geodesic properties
# =============================================================================

def test_geodesic_radial_direction():
  """
  Geodesics should initially move in the direction of the tangent vector.

  For small v, exp_p(v) ≈ p + v (in standard coords).
  """
  p = jnp.array([1.0, 0.5])
  metric = create_nearly_flat_metric(p, epsilon=0.05)

  # Small tangent vector in standard coords
  v_std = jnp.array([0.01, 0.0])
  v = create_tangent_vector(p, v_std)

  expected_direction = v_std / jnp.linalg.norm(v_std)

  q = exponential_map_taylor(metric, v)
  actual_direction = (q - p) / jnp.linalg.norm(q - p)

  assert jnp.allclose(expected_direction, actual_direction, atol=1e-3)


def test_geodesic_scaling():
  """
  For small v, exp_p(tv) should approximately be p + t*v.
  """
  p = jnp.array([1.0, 0.5])
  metric = create_nearly_flat_metric(p, epsilon=0.1)

  v_std = jnp.array([0.1, -0.05])

  for t in [0.1, 0.5, 1.0]:
    v = create_tangent_vector(p, t * v_std)
    q = exponential_map_taylor(metric, v)
    # Linear approximation
    q_linear = p + t * v_std

    # For small t, should be close
    if t < 0.5:
      assert jnp.allclose(q, q_linear, atol=0.01)


# =============================================================================
# Tests for differentiability
# =============================================================================

def test_exponential_map_taylor_differentiable():
  """The Taylor exponential map should be differentiable w.r.t. velocity components."""
  p = jnp.array([1.0, 0.5])
  metric = create_nearly_flat_metric(p, epsilon=0.1)

  def exp_fn(v_arr):
    v = create_tangent_vector(p, v_arr)
    return exponential_map_taylor(metric, v)

  v_arr = jnp.array([0.1, -0.05])

  # Should not raise
  jac = jax.jacobian(exp_fn)(v_arr)

  assert jac.shape == (2, 2)
  # For small v near flat metric, Jacobian should be close to identity
  assert jnp.allclose(jac, jnp.eye(2), atol=0.1)


def test_logarithmic_map_taylor_differentiable():
  """The Taylor logarithmic map should be differentiable."""
  p = jnp.array([1.0, 0.5])
  metric = create_nearly_flat_metric(p, epsilon=0.1)

  def log_fn(q):
    return logarithmic_map_taylor(metric, q)

  q = jnp.array([1.05, 0.52])

  # Should not raise
  jac = jax.jacobian(log_fn)(q)

  assert jac.shape == (2, 2)


# =============================================================================
# Tests for boundary cases
# =============================================================================

def test_exponential_map_1d():
  """Test exponential map in 1D."""
  p = jnp.array([1.0])

  # Simple 1D metric g = 1 + epsilon * x^2
  epsilon = 0.1
  g_val = jnp.array([[1.0 + epsilon * p[0]**2]])
  g_grad = jnp.array([[[2 * epsilon * p[0]]]])
  g_hess = jnp.array([[[[2 * epsilon]]]])

  basis = get_standard_basis(p)
  components = Jet(value=g_val, gradient=g_grad, hessian=g_hess)
  metric = RiemannianMetric(basis=basis, components=components)

  v_arr = jnp.array([0.1])
  v = create_tangent_vector(p, v_arr)
  q = exponential_map_taylor(metric, v)

  assert q.shape == (1,)
  # Should be close to p + v for small curvature
  assert jnp.allclose(q, p + v_arr, atol=0.1)


def test_exponential_map_high_dimension():
  """Test that exponential map works in higher dimensions."""
  dim = 5
  key = random.PRNGKey(42)
  p = random.normal(key, (dim,)) * 0.5
  metric = create_random_metric(key, dim, p)

  v_arr = random.normal(key, (dim,)) * 0.05
  v = create_tangent_vector(p, v_arr)
  q = exponential_map_taylor(metric, v)

  assert q.shape == (dim,)
  # Basic sanity: q should be close to p for small v
  assert jnp.linalg.norm(q - p) < 1.0


# =============================================================================
# Tests against known closed-form exponential maps
# =============================================================================

class TestHyperbolicPlane:
  """
  Tests for the hyperbolic plane in the upper half-plane model.

  The upper half-plane H² = {(x, y) : y > 0} with metric:
    ds² = (dx² + dy²) / y²

  Geodesics are:
  - Vertical lines (x = const) for purely vertical initial velocity
  - Semicircles centered on the x-axis for other directions

  For a vertical geodesic from p = (x₀, y₀) with velocity v = (0, v_y):
    exp_p(v) = (x₀, y₀ * exp(v_y / y₀))
  """

  @staticmethod
  def g_components(x: jnp.ndarray) -> jnp.ndarray:
    """Metric components for hyperbolic upper half-plane: g_ij = (1/y²) δ_ij."""
    y = x[1]
    return jnp.eye(2) / (y ** 2)

  metric_fn = staticmethod(make_metric_fn(g_components.__func__))

  @staticmethod
  def exp_vertical_exact(p: jnp.ndarray, v_y: float) -> jnp.ndarray:
    """
    Exact exponential map for vertical geodesic.

    From p = (x₀, y₀) with velocity (0, v_y), the geodesic is:
      γ(t) = (x₀, y₀ * exp(v_y * t / y₀))

    So exp_p((0, v_y)) = (x₀, y₀ * exp(v_y / y₀))
    """
    x0, y0 = p[0], p[1]
    return jnp.array([x0, y0 * jnp.exp(v_y / y0)])

  def test_vertical_geodesic_upward(self):
    """Test vertical geodesic going upward."""
    p = jnp.array([0.5, 1.0])  # Point in upper half-plane
    v_arr = jnp.array([0.0, 0.3])  # Vertical velocity

    q_exact = self.exp_vertical_exact(p, v_arr[1])
    v = create_tangent_vector(p, v_arr)
    q_ode = exponential_map_ode(v, self.metric_fn)

    assert jnp.allclose(q_ode, q_exact, atol=1e-4), \
      f"ODE: {q_ode}, Exact: {q_exact}"

  def test_vertical_geodesic_downward(self):
    """Test vertical geodesic going downward."""
    p = jnp.array([0.0, 2.0])
    v_arr = jnp.array([0.0, -0.5])  # Downward velocity

    q_exact = self.exp_vertical_exact(p, v_arr[1])
    v = create_tangent_vector(p, v_arr)
    q_ode = exponential_map_ode(v, self.metric_fn)

    assert jnp.allclose(q_ode, q_exact, atol=1e-4), \
      f"ODE: {q_ode}, Exact: {q_exact}"

  def test_vertical_geodesic_various_points(self):
    """Test vertical geodesics from various starting points."""
    test_cases = [
      (jnp.array([0.0, 1.0]), 0.2),
      (jnp.array([1.0, 0.5]), 0.1),
      (jnp.array([-0.5, 2.0]), -0.3),
      (jnp.array([0.0, 0.5]), 0.5),
    ]

    for p, v_y in test_cases:
      v_arr = jnp.array([0.0, v_y])
      q_exact = self.exp_vertical_exact(p, v_y)
      v = create_tangent_vector(p, v_arr)
      q_ode = exponential_map_ode(v, self.metric_fn)

      assert jnp.allclose(q_ode, q_exact, atol=1e-4), \
        f"Failed at p={p}, v_y={v_y}: ODE={q_ode}, Exact={q_exact}"


class TestSphere:
  """
  Tests for the 2-sphere in spherical coordinates (θ, φ).

  Metric: ds² = dθ² + sin²(θ) dφ²

  For geodesics along a meridian (φ = const) with velocity (v_θ, 0):
    exp_(θ₀, φ₀)((v_θ, 0)) = (θ₀ + v_θ, φ₀)

  This is because meridians are geodesics and θ is an arc-length parameter.
  """

  @staticmethod
  def g_components(x: jnp.ndarray) -> jnp.ndarray:
    """Metric components for 2-sphere in (θ, φ) coordinates."""
    theta = x[0]
    # g = [[1, 0], [0, sin²θ]]
    # Add small epsilon to avoid singularity at poles
    sin_theta = jnp.sin(theta) + 1e-10
    return jnp.array([[1.0, 0.0], [0.0, sin_theta ** 2]])

  metric_fn = staticmethod(make_metric_fn(g_components.__func__))

  @staticmethod
  def exp_meridian_exact(p: jnp.ndarray, v_theta: float) -> jnp.ndarray:
    """
    Exact exponential map along a meridian.

    Meridians (φ = const) are geodesics, and θ is arc-length.
    exp_(θ₀, φ₀)((v_θ, 0)) = (θ₀ + v_θ, φ₀)
    """
    theta0, phi0 = p[0], p[1]
    return jnp.array([theta0 + v_theta, phi0])

  def test_meridian_geodesic_northward(self):
    """Test geodesic along meridian toward north pole."""
    p = jnp.array([jnp.pi / 2, 0.0])  # On equator
    v_arr = jnp.array([0.3, 0.0])  # Move toward north pole

    q_exact = self.exp_meridian_exact(p, v_arr[0])
    v = create_tangent_vector(p, v_arr)
    q_ode = exponential_map_ode(v, self.metric_fn)

    assert jnp.allclose(q_ode, q_exact, atol=1e-4), \
      f"ODE: {q_ode}, Exact: {q_exact}"

  def test_meridian_geodesic_southward(self):
    """Test geodesic along meridian toward south pole."""
    p = jnp.array([jnp.pi / 3, jnp.pi / 4])  # Northern hemisphere
    v_arr = jnp.array([0.5, 0.0])  # Move toward south

    q_exact = self.exp_meridian_exact(p, v_arr[0])
    v = create_tangent_vector(p, v_arr)
    q_ode = exponential_map_ode(v, self.metric_fn)

    assert jnp.allclose(q_ode, q_exact, atol=1e-4), \
      f"ODE: {q_ode}, Exact: {q_exact}"

  def test_meridian_geodesic_various_points(self):
    """Test meridian geodesics from various starting points."""
    test_cases = [
      (jnp.array([jnp.pi / 4, 0.0]), 0.2),
      (jnp.array([jnp.pi / 2, jnp.pi]), -0.3),
      (jnp.array([2 * jnp.pi / 3, jnp.pi / 2]), 0.4),
      (jnp.array([jnp.pi / 6, -jnp.pi / 4]), -0.1),
    ]

    for p, v_theta in test_cases:
      v_arr = jnp.array([v_theta, 0.0])
      q_exact = self.exp_meridian_exact(p, v_theta)
      v = create_tangent_vector(p, v_arr)
      q_ode = exponential_map_ode(v, self.metric_fn)

      assert jnp.allclose(q_ode, q_exact, atol=1e-4), \
        f"Failed at p={p}, v_θ={v_theta}: ODE={q_ode}, Exact={q_exact}"


class TestPoincareDisk:
  """
  Tests for the Poincaré disk model of hyperbolic geometry.

  The disk is D = {(x, y) : x² + y² < 1} with metric:
    ds² = 4(dx² + dy²) / (1 - r²)²  where r² = x² + y²

  At the origin (where g = 4I), geodesics are straight lines in Euclidean sense.
  For p = (0, 0) and velocity v (in Euclidean coordinates), the hyperbolic
  length is |v|_g = 2|v|. The geodesic in direction v/|v| with this speed is:
    γ(t) = tanh(|v|_g * t / 2) * (v / |v|) = tanh(|v| * t) * (v / |v|)

  So exp_0(v) = tanh(|v|) * (v / |v|) for |v| > 0, and 0 for v = 0.
  """

  @staticmethod
  def g_components(x: jnp.ndarray) -> jnp.ndarray:
    """Metric components for Poincaré disk: g_ij = 4/(1-r²)² δ_ij."""
    r_sq = x[0]**2 + x[1]**2
    scale = 4.0 / (1.0 - r_sq + 1e-10) ** 2
    return scale * jnp.eye(2)

  metric_fn = staticmethod(make_metric_fn(g_components.__func__))

  @staticmethod
  def exp_from_origin_exact(v: jnp.ndarray) -> jnp.ndarray:
    """
    Exact exponential map from the origin.

    At origin, g = 4I, so |v|_hyperbolic = 2|v|_Euclidean.
    The geodesic distance to a point at Euclidean distance d is 2*arctanh(d).
    Solving: d = tanh(|v|_hyperbolic / 2) = tanh(|v|_Euclidean).

    exp_0(v) = tanh(|v|) * (v / |v|) for |v| > 0
    exp_0(0) = 0
    """
    norm_v = jnp.linalg.norm(v)
    # Handle v = 0 case
    safe_norm = jnp.where(norm_v > 1e-10, norm_v, 1.0)
    direction = v / safe_norm
    scale = jnp.tanh(norm_v)
    return jnp.where(norm_v > 1e-10, scale * direction, jnp.zeros(2))

  def test_geodesic_from_origin_x_direction(self):
    """Test geodesic from origin in x-direction."""
    p = jnp.array([0.0, 0.0])
    v_arr = jnp.array([0.5, 0.0])

    q_exact = self.exp_from_origin_exact(v_arr)
    v = create_tangent_vector(p, v_arr)
    q_ode = exponential_map_ode(v, self.metric_fn)

    assert jnp.allclose(q_ode, q_exact, atol=1e-3), \
      f"ODE: {q_ode}, Exact: {q_exact}"

  def test_geodesic_from_origin_y_direction(self):
    """Test geodesic from origin in y-direction."""
    p = jnp.array([0.0, 0.0])
    v_arr = jnp.array([0.0, 0.4])

    q_exact = self.exp_from_origin_exact(v_arr)
    v = create_tangent_vector(p, v_arr)
    q_ode = exponential_map_ode(v, self.metric_fn)

    assert jnp.allclose(q_ode, q_exact, atol=1e-3), \
      f"ODE: {q_ode}, Exact: {q_exact}"

  def test_geodesic_from_origin_diagonal(self):
    """Test geodesic from origin in diagonal direction."""
    p = jnp.array([0.0, 0.0])
    v_arr = jnp.array([0.3, 0.3])

    q_exact = self.exp_from_origin_exact(v_arr)
    v = create_tangent_vector(p, v_arr)
    q_ode = exponential_map_ode(v, self.metric_fn)

    assert jnp.allclose(q_ode, q_exact, atol=1e-3), \
      f"ODE: {q_ode}, Exact: {q_exact}"

  def test_geodesic_from_origin_various_velocities(self):
    """Test geodesics from origin with various velocities."""
    test_cases = [
      jnp.array([0.1, 0.0]),
      jnp.array([0.0, 0.2]),
      jnp.array([0.2, 0.1]),
      jnp.array([-0.3, 0.2]),
      jnp.array([0.4, -0.1]),
    ]

    p = jnp.array([0.0, 0.0])
    for v_arr in test_cases:
      q_exact = self.exp_from_origin_exact(v_arr)
      v = create_tangent_vector(p, v_arr)
      q_ode = exponential_map_ode(v, self.metric_fn)

      assert jnp.allclose(q_ode, q_exact, atol=1e-3), \
        f"Failed at v={v_arr}: ODE={q_ode}, Exact={q_exact}"

  def test_zero_velocity_from_origin(self):
    """Test that exp_0(0) = 0."""
    p = jnp.array([0.0, 0.0])
    v_arr = jnp.array([0.0, 0.0])

    q_exact = self.exp_from_origin_exact(v_arr)
    v = create_tangent_vector(p, v_arr)
    q_ode = exponential_map_ode(v, self.metric_fn)

    assert jnp.allclose(q_ode, q_exact, atol=1e-6), \
      f"ODE: {q_ode}, Exact: {q_exact}"
