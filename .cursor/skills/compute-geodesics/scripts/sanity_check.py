"""
Sanity check script for compute-geodesics skill.

Verifies key invariants of geodesic computations:
1. exp_p(0) = p (zero velocity returns base point)
2. Euclidean: exp_p(v) = p + v (flat space)
3. Round-trip: log_p(exp_p(v)) ≈ v

Run with: uv run python .cursor/skills/compute-geodesics/scripts/sanity_check.py
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from local_coordinates.jet import Jet, function_to_jet
from local_coordinates.basis import get_standard_basis
from local_coordinates.metric import RiemannianMetric
from local_coordinates.tangent import TangentVector
from local_coordinates.exponential_map import (
  exponential_map_taylor,
  logarithmic_map_taylor
)


def check_zero_velocity_returns_base():
  """Verify exp_p(0) = p."""
  print("Checking exp_p(0) = p... ", end="")

  p = jnp.array([1.0, 2.0])

  def metric_components(x):
    return jnp.array([
      [1 + 0.1*x[0]**2, 0.0],
      [0.0, 1 + 0.1*x[1]**2]
    ])

  basis = get_standard_basis(p)
  metric_jet = function_to_jet(metric_components, p)
  metric = RiemannianMetric(basis=basis, components=metric_jet)

  # Zero velocity
  v_components = jnp.array([0.0, 0.0])
  v_jet = Jet(
    value=v_components,
    gradient=jnp.zeros((2, 2)),
    hessian=jnp.zeros((2, 2, 2))
  )
  v = TangentVector(p=p, components=v_jet, basis=basis)

  # Exponential map
  q = exponential_map_taylor(metric, v)

  assert jnp.allclose(q, p, atol=1e-14), f"exp_p(0) != p: got {q}, expected {p}"

  print("PASSED")


def check_euclidean_straight_line():
  """Verify exp_p(v) = p + v for flat (Euclidean) metric."""
  print("Checking Euclidean: exp_p(v) = p + v... ", end="")

  p = jnp.array([1.0, 2.0])
  basis = get_standard_basis(p)

  # Euclidean metric (constant identity)
  metric = RiemannianMetric(
    basis=basis,
    components=Jet(
      value=jnp.eye(2),
      gradient=jnp.zeros((2, 2, 2)),
      hessian=jnp.zeros((2, 2, 2, 2))
    )
  )

  # Some velocity
  v_components = jnp.array([0.5, -0.3])
  v_jet = Jet(
    value=v_components,
    gradient=jnp.zeros((2, 2)),
    hessian=jnp.zeros((2, 2, 2))
  )
  v = TangentVector(p=p, components=v_jet, basis=basis)

  # Exponential map
  q = exponential_map_taylor(metric, v)

  expected = p + v_components
  assert jnp.allclose(q, expected, atol=1e-10), \
    f"Euclidean exp failed: got {q}, expected {expected}"

  print("PASSED")


def check_exp_log_roundtrip():
  """Verify log_p(exp_p(v)) ≈ v for small v."""
  print("Checking round-trip: log_p(exp_p(v)) ≈ v... ", end="")

  p = jnp.array([0.0, 0.0])

  def metric_components(x):
    return jnp.array([
      [1 + 0.1*x[0]**2, 0.0],
      [0.0, 1 + 0.1*x[1]**2]
    ])

  basis = get_standard_basis(p)
  metric_jet = function_to_jet(metric_components, p)
  metric = RiemannianMetric(basis=basis, components=metric_jet)

  # Small velocity
  v_components = jnp.array([0.1, 0.05])
  v_jet = Jet(
    value=v_components,
    gradient=jnp.zeros((2, 2)),
    hessian=jnp.zeros((2, 2, 2))
  )
  v = TangentVector(p=p, components=v_jet, basis=basis)

  # Forward: exp_p(v)
  q = exponential_map_taylor(metric, v)

  # Backward: log_p(q)
  # Note: logarithmic_map_taylor returns v in RNC, which at origin should match
  v_recovered = logarithmic_map_taylor(metric, q)

  # For small v, should recover approximately
  error = jnp.linalg.norm(v_recovered - v_components)
  assert error < 0.01, f"Round-trip error too large: {error}"

  print("PASSED")


def check_small_displacement_accuracy():
  """Verify Taylor approximation is accurate for small displacements."""
  print("Checking small displacement accuracy... ", end="")

  p = jnp.array([0.0, 0.0])

  def metric_components(x):
    scale = 1.0 / (1.0 + 0.1 * jnp.sum(x**2))
    return scale * jnp.eye(2)

  basis = get_standard_basis(p)
  metric_jet = function_to_jet(metric_components, p)
  metric = RiemannianMetric(basis=basis, components=metric_jet)

  # Very small velocity
  v_components = jnp.array([0.01, 0.01])
  v_jet = Jet(
    value=v_components,
    gradient=jnp.zeros((2, 2)),
    hessian=jnp.zeros((2, 2, 2))
  )
  v = TangentVector(p=p, components=v_jet, basis=basis)

  # Exponential map
  q = exponential_map_taylor(metric, v)

  # For very small v and metric close to identity at origin,
  # exp_p(v) should be close to p + v
  simple_approx = p + v_components
  error = jnp.linalg.norm(q - simple_approx)

  # Error should be small (Taylor corrections are higher order)
  assert error < 1e-4, f"Small displacement error too large: {error}"

  print("PASSED")


def check_geodesic_is_differentiable():
  """Verify exponential map is differentiable."""
  print("Checking exponential map is differentiable... ", end="")

  p = jnp.array([0.0, 0.0])

  def metric_components(x):
    return jnp.array([
      [1 + 0.1*x[0]**2, 0.0],
      [0.0, 1 + 0.1*x[1]**2]
    ])

  def exp_wrapper(v_val):
    basis = get_standard_basis(p)
    metric_jet = function_to_jet(metric_components, p)
    metric = RiemannianMetric(basis=basis, components=metric_jet)

    v_jet = Jet(
      value=v_val,
      gradient=jnp.zeros((2, 2)),
      hessian=jnp.zeros((2, 2, 2))
    )
    v = TangentVector(p=p, components=v_jet, basis=basis)
    return exponential_map_taylor(metric, v)

  # Compute Jacobian
  v_val = jnp.array([0.1, 0.1])
  jac = jax.jacfwd(exp_wrapper)(v_val)

  # Should be finite
  assert jnp.all(jnp.isfinite(jac)), "Jacobian has non-finite values"

  # At v=0.1, for a nearly-identity metric, Jacobian should be close to identity
  assert jnp.allclose(jac, jnp.eye(2), atol=0.5), f"Jacobian unexpected:\n{jac}"

  print("PASSED")


if __name__ == "__main__":
  print("=" * 50)
  print("Geodesic Computation Sanity Checks")
  print("=" * 50)

  check_zero_velocity_returns_base()
  check_euclidean_straight_line()
  check_exp_log_roundtrip()
  check_small_displacement_accuracy()
  check_geodesic_is_differentiable()

  print("=" * 50)
  print("All sanity checks PASSED")
  print("=" * 50)
