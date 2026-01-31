"""
Sanity check script for pullback-metric skill.

Verifies key invariants of pullback metrics:
1. Polar coordinates: pullback of Euclidean gives diag(1, r^2)
2. Identity map preserves metric
3. Linear map: pullback(Ax, I) = A^T A

Run with: uv run python .cursor/skills/pullback-metric/scripts/sanity_check.py
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from local_coordinates.jet import Jet
from local_coordinates.basis import get_standard_basis
from local_coordinates.metric import RiemannianMetric, pullback_metric


def check_polar_coordinates():
  """Verify polar coords give metric diag(1, r^2)."""
  print("Checking polar coordinate pullback... ", end="")

  def polar_to_cartesian(q):
    r, phi = q[0], q[1]
    return jnp.array([r * jnp.cos(phi), r * jnp.sin(phi)])

  r_val = 2.0
  phi_val = jnp.pi / 4.0
  p_polar = jnp.array([r_val, phi_val])
  p_cart = polar_to_cartesian(p_polar)

  # Euclidean metric in Cartesian
  basis_cart = get_standard_basis(p_cart)
  euclidean = RiemannianMetric(
    basis=basis_cart,
    components=Jet(
      value=jnp.eye(2),
      gradient=jnp.zeros((2, 2, 2)),
      hessian=jnp.zeros((2, 2, 2, 2))
    )
  )

  # Pullback to polar
  polar_metric = pullback_metric(p_polar, polar_to_cartesian, euclidean)
  g = polar_metric.components.value

  # Expected: diag(1, r^2)
  expected = jnp.array([[1.0, 0.0], [0.0, r_val**2]])

  assert jnp.allclose(g, expected, atol=1e-10), \
    f"Polar metric mismatch:\nGot:\n{g}\nExpected:\n{expected}"

  print("PASSED")


def check_identity_map():
  """Verify identity map preserves metric."""
  print("Checking identity map preserves metric... ", end="")

  def identity(x):
    return x

  p = jnp.array([1.0, 2.0])
  basis = get_standard_basis(p)

  # Non-trivial metric
  g_components = jnp.array([[2.0, 0.5], [0.5, 1.0]])
  metric = RiemannianMetric(
    basis=basis,
    components=Jet(
      value=g_components,
      gradient=jnp.zeros((2, 2, 2)),
      hessian=jnp.zeros((2, 2, 2, 2))
    )
  )

  # Pullback under identity
  pulled = pullback_metric(p, identity, metric)

  assert jnp.allclose(pulled.components.value, g_components, atol=1e-10), \
    f"Identity pullback changed metric:\nGot:\n{pulled.components.value}\nExpected:\n{g_components}"

  print("PASSED")


def check_linear_map():
  """Verify linear map: pullback(Ax, I) = A^T A."""
  print("Checking linear map pullback... ", end="")

  A = jnp.array([[2.0, 1.0], [0.5, 1.5]])

  def linear_map(x):
    return A @ x

  p = jnp.array([1.0, 1.0])
  q = linear_map(p)

  # Euclidean metric at target
  basis_target = get_standard_basis(q)
  euclidean = RiemannianMetric(
    basis=basis_target,
    components=Jet(
      value=jnp.eye(2),
      gradient=jnp.zeros((2, 2, 2)),
      hessian=jnp.zeros((2, 2, 2, 2))
    )
  )

  # Pullback
  pulled = pullback_metric(p, linear_map, euclidean)

  # Expected: A^T A
  expected = A.T @ A

  assert jnp.allclose(pulled.components.value, expected, atol=1e-10), \
    f"Linear pullback mismatch:\nGot:\n{pulled.components.value}\nExpected:\n{expected}"

  print("PASSED")


def check_rotation_preserves_euclidean():
  """Verify rotation preserves Euclidean metric."""
  print("Checking rotation preserves Euclidean... ", end="")

  theta = jnp.pi / 6

  def rotate(x):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([c * x[0] - s * x[1], s * x[0] + c * x[1]])

  p = jnp.array([1.0, 0.0])
  q = rotate(p)

  # Euclidean metric at target
  basis_target = get_standard_basis(q)
  euclidean = RiemannianMetric(
    basis=basis_target,
    components=Jet(
      value=jnp.eye(2),
      gradient=jnp.zeros((2, 2, 2)),
      hessian=jnp.zeros((2, 2, 2, 2))
    )
  )

  # Pullback (should be identity since rotation is an isometry)
  pulled = pullback_metric(p, rotate, euclidean)

  assert jnp.allclose(pulled.components.value, jnp.eye(2), atol=1e-10), \
    f"Rotation didn't preserve Euclidean:\nGot:\n{pulled.components.value}"

  print("PASSED")


def check_scaling_gives_scaled_metric():
  """Verify scaling: pullback(kx, I) = k^2 I."""
  print("Checking scaling gives k^2 I... ", end="")

  k = 2.0

  def scale(x):
    return k * x

  p = jnp.array([1.0, 1.0])
  q = scale(p)

  # Euclidean metric at target
  basis_target = get_standard_basis(q)
  euclidean = RiemannianMetric(
    basis=basis_target,
    components=Jet(
      value=jnp.eye(2),
      gradient=jnp.zeros((2, 2, 2)),
      hessian=jnp.zeros((2, 2, 2, 2))
    )
  )

  # Pullback
  pulled = pullback_metric(p, scale, euclidean)

  # Expected: k^2 I
  expected = k**2 * jnp.eye(2)

  assert jnp.allclose(pulled.components.value, expected, atol=1e-10), \
    f"Scaling pullback mismatch:\nGot:\n{pulled.components.value}\nExpected:\n{expected}"

  print("PASSED")


if __name__ == "__main__":
  print("=" * 50)
  print("Pullback Metric Sanity Checks")
  print("=" * 50)

  check_polar_coordinates()
  check_identity_map()
  check_linear_map()
  check_rotation_preserves_euclidean()
  check_scaling_gives_scaled_metric()

  print("=" * 50)
  print("All sanity checks PASSED")
  print("=" * 50)
