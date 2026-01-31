"""
Sanity check script for jet-differentiation skill.

Verifies key invariants of the Jet implementation:
1. Jet creation produces correct value, gradient, and Hessian
2. Coordinate transformation round-trip preserves the Jet
3. @jet_decorator correctly propagates derivatives

Run with: uv run python -m local_coordinates.scripts.jet_sanity_check
Or directly: uv run python .cursor/skills/jet-differentiation/scripts/sanity_check.py
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from local_coordinates.jet import Jet, function_to_jet, jet_decorator, change_coordinates


def check_function_to_jet():
  """Verify that function_to_jet produces correct derivatives."""
  print("Checking function_to_jet... ", end="")

  def quad_func(x):
    return jnp.sum(x**2)

  x = jnp.array([1.0, 2.0])
  jet = function_to_jet(quad_func, x)

  # Manual verification
  expected_value = jnp.sum(x**2)  # 1 + 4 = 5
  expected_gradient = 2 * x  # [2, 4]
  expected_hessian = 2 * jnp.eye(2)  # [[2, 0], [0, 2]]

  assert jnp.allclose(jet.value, expected_value), f"Value mismatch: {jet.value} vs {expected_value}"
  assert jnp.allclose(jet.gradient, expected_gradient), f"Gradient mismatch: {jet.gradient} vs {expected_gradient}"
  assert jnp.allclose(jet.hessian, expected_hessian), f"Hessian mismatch: {jet.hessian} vs {expected_hessian}"

  print("PASSED")


def check_jet_decorator_chain_rule():
  """Verify that @jet_decorator correctly applies the chain rule."""
  print("Checking @jet_decorator chain rule... ", end="")

  @jet_decorator
  def square(x):
    return x**2

  # Create identity jet: f(t) = t, so df/dt = 1, d^2f/dt^2 = 0
  jet = function_to_jet(lambda t: t, jnp.array(3.0))
  out = square(jet)

  # For g(f(t)) = f(t)^2 = t^2:
  # value = 9
  # gradient = 2*f * df/dt = 2*3*1 = 6
  # hessian = 2*(df/dt)^2 + 2*f*(d^2f/dt^2) = 2*1 + 0 = 2
  assert jnp.allclose(out.value, 9.0), f"Value mismatch: {out.value}"
  assert jnp.allclose(out.gradient, jnp.array([6.0])), f"Gradient mismatch: {out.gradient}"
  assert jnp.allclose(out.hessian, jnp.array([[2.0]])), f"Hessian mismatch: {out.hessian}"

  print("PASSED")


def check_coordinate_round_trip():
  """Verify that coordinate transformation is invertible."""
  print("Checking coordinate round-trip... ", end="")

  # Define a scalar function
  def f(x):
    return x[0]**2 + x[1]**3

  x = jnp.array([1.0, 2.0])
  jet_original = function_to_jet(f, x)

  # Coordinate transformation: rotation
  theta = jnp.pi / 6
  def x_to_z(x):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([c * x[0] + s * x[1], -s * x[0] + c * x[1]])

  def z_to_x(z):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([c * z[0] - s * z[1], s * z[0] + c * z[1]])

  # Transform to z-coordinates and back
  jet_z = change_coordinates(jet_original, x_to_z, x)
  z = x_to_z(x)
  jet_back = change_coordinates(jet_z, z_to_x, z)

  # Value should be unchanged
  assert jnp.allclose(jet_original.value, jet_back.value), "Value changed after round-trip"

  # Gradient and Hessian should be recovered (up to numerical precision)
  assert jnp.allclose(jet_original.gradient, jet_back.gradient, atol=1e-10), \
    f"Gradient changed: {jet_original.gradient} vs {jet_back.gradient}"
  assert jnp.allclose(jet_original.hessian, jet_back.hessian, atol=1e-10), \
    f"Hessian changed: {jet_original.hessian} vs {jet_back.hessian}"

  print("PASSED")


def check_hessian_symmetry():
  """Verify that Hessian is symmetric."""
  print("Checking Hessian symmetry... ", end="")

  def f(x):
    return x[0]**2 * x[1] + jnp.sin(x[0] * x[1])

  x = jnp.array([1.0, 2.0])
  jet = function_to_jet(f, x)

  # Hessian should be symmetric
  assert jnp.allclose(jet.hessian, jet.hessian.T), \
    f"Hessian not symmetric: {jet.hessian}"

  print("PASSED")


def check_taylor_evaluation():
  """Verify that Taylor evaluation approximates the function."""
  print("Checking Taylor evaluation... ", end="")

  def f(x):
    return jnp.sin(x[0]) + jnp.cos(x[1])

  x = jnp.array([0.5, 0.5])
  jet = function_to_jet(f, x)

  # Small displacement
  dx = jnp.array([0.01, -0.01])

  # Taylor approximation
  taylor_approx = jet(dx)

  # Exact value
  exact = f(x + dx)

  # Should be close for small dx
  error = jnp.abs(taylor_approx - exact)
  assert error < 1e-6, f"Taylor approximation error too large: {error}"

  print("PASSED")


if __name__ == "__main__":
  print("=" * 50)
  print("Jet Differentiation Sanity Checks")
  print("=" * 50)

  check_function_to_jet()
  check_jet_decorator_chain_rule()
  check_coordinate_round_trip()
  check_hessian_symmetry()
  check_taylor_evaluation()

  print("=" * 50)
  print("All sanity checks PASSED")
  print("=" * 50)
