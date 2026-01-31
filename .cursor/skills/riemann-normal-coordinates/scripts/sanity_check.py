"""
Sanity check script for riemann-normal-coordinates skill.

Verifies key invariants of RNC:
1. Metric is identity at origin
2. Metric gradient vanishes at origin
3. Christoffel symbols vanish at origin
4. Taylor coefficient symmetry

Run with: uv run python .cursor/skills/riemann-normal-coordinates/scripts/sanity_check.py
"""

import jax
import jax.numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)

from local_coordinates.jet import function_to_jet, Jet, get_identity_jet
from local_coordinates.basis import get_standard_basis, BasisVectors
from local_coordinates.metric import RiemannianMetric
from local_coordinates.connection import get_levi_civita_connection
from local_coordinates.normal_coords import (
  get_rnc_jacobians,
  to_riemann_normal_coordinates
)


def create_curved_metric(p):
  """Create a non-trivial curved metric at point p."""
  def metric_components(x):
    return jnp.array([
      [1 + 0.1*x[0]**2, 0.05*x[0]*x[1]],
      [0.05*x[0]*x[1], 1 + 0.1*x[1]**2]
    ])

  basis = get_standard_basis(p)
  metric_jet = function_to_jet(metric_components, p)
  return RiemannianMetric(basis=basis, components=metric_jet)


def check_metric_identity_at_origin():
  """Verify metric is identity in RNC at origin."""
  print("Checking metric is identity at RNC origin... ", end="")

  p = jnp.array([1.0, 1.0])
  metric = create_curved_metric(p)

  # Transform to RNC
  metric_rnc = to_riemann_normal_coordinates(metric)

  # Should be identity
  expected = jnp.eye(2)
  actual = metric_rnc.components.value

  assert jnp.allclose(actual, expected, atol=1e-10), \
    f"Metric not identity in RNC:\nGot:\n{actual}\nExpected:\n{expected}"

  print("PASSED")


def check_metric_gradient_vanishes():
  """Verify metric gradient vanishes in RNC at origin."""
  print("Checking metric gradient vanishes at RNC origin... ", end="")

  p = jnp.array([1.0, 1.0])
  metric = create_curved_metric(p)

  # Transform to RNC
  metric_rnc = to_riemann_normal_coordinates(metric)

  # Gradient should be zero
  gradient = metric_rnc.components.gradient
  max_grad = jnp.max(jnp.abs(gradient))

  assert max_grad < 1e-10, f"Metric gradient not zero in RNC: max = {max_grad}"

  print("PASSED")


def check_christoffel_vanishes():
  """Verify Christoffel symbols vanish in RNC at origin."""
  print("Checking Christoffel symbols vanish at RNC origin... ", end="")

  p = jnp.array([1.0, 1.0])
  metric = create_curved_metric(p)
  connection = get_levi_civita_connection(metric)

  # Transform to RNC
  connection_rnc = to_riemann_normal_coordinates(connection, metric)

  # Christoffel symbols should vanish
  Gamma = connection_rnc.christoffel_symbols.value
  max_Gamma = jnp.max(jnp.abs(Gamma))

  assert max_Gamma < 1e-10, f"Christoffel symbols not zero in RNC: max = {max_Gamma}"

  print("PASSED")


def check_jacobian_second_derivative_symmetry():
  """Verify d^2x/dv^j dv^k is symmetric."""
  print("Checking Jacobian second derivative symmetry... ", end="")

  p = jnp.array([1.0, 1.0])
  metric = create_curved_metric(p)

  J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)

  # Second derivative d^2x/dv^j dv^k should be symmetric
  H = J_v_to_x.gradient  # shape (dim, dim, dim): H[i, j, k] = d^2x^i/dv^j dv^k

  # Check symmetry in last two indices
  for i in range(2):
    H_i = H[i]
    assert jnp.allclose(H_i, H_i.T, atol=1e-10), \
      f"Second derivative not symmetric for component {i}"

  print("PASSED")


def check_jacobian_third_derivative_symmetry():
  """Verify d^3x/dv^a dv^b dv^c is symmetric."""
  print("Checking Jacobian third derivative symmetry... ", end="")

  p = jnp.array([1.0, 1.0])
  metric = create_curved_metric(p)

  J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)

  # Third derivative d^3x/dv^a dv^b dv^c should be symmetric
  T = J_v_to_x.hessian  # shape (dim, dim, dim, dim): T[i, j, k, l] = d^3x^i/dv^j dv^k dv^l

  # Check symmetry in last three indices (all permutations)
  for i in range(2):
    T_i = T[i]  # shape (dim, dim, dim)
    # Check (j,k,l) = (k,j,l)
    assert jnp.allclose(T_i, jnp.swapaxes(T_i, 0, 1), atol=1e-10), "Not symmetric in first pair"
    # Check (j,k,l) = (j,l,k)
    assert jnp.allclose(T_i, jnp.swapaxes(T_i, 1, 2), atol=1e-10), "Not symmetric in second pair"

  print("PASSED")


def check_ricci_scalar_invariance():
  """Verify Ricci scalar is the same in RNC and original coords."""
  print("Checking Ricci scalar invariance... ", end="")

  from local_coordinates.riemann import get_riemann_curvature_tensor, get_ricci_tensor

  p = jnp.array([1.0, 1.0])
  metric = create_curved_metric(p)
  connection = get_levi_civita_connection(metric)
  ricci = get_ricci_tensor(connection)

  # Compute Ricci scalar in original coordinates
  g_inv_orig = jnp.linalg.inv(metric.components.value)
  R_scalar_orig = jnp.einsum("ij,ij->", g_inv_orig, ricci.components.value)

  # Transform to RNC
  metric_rnc = to_riemann_normal_coordinates(metric)
  connection_rnc = to_riemann_normal_coordinates(connection, metric)
  ricci_rnc = get_ricci_tensor(connection_rnc)

  # Compute Ricci scalar in RNC
  g_inv_rnc = jnp.linalg.inv(metric_rnc.components.value)
  R_scalar_rnc = jnp.einsum("ij,ij->", g_inv_rnc, ricci_rnc.components.value)

  assert jnp.allclose(R_scalar_orig, R_scalar_rnc, atol=1e-8), \
    f"Ricci scalar not invariant: orig={R_scalar_orig}, RNC={R_scalar_rnc}"

  print("PASSED")


if __name__ == "__main__":
  print("=" * 50)
  print("Riemann Normal Coordinates Sanity Checks")
  print("=" * 50)

  check_metric_identity_at_origin()
  check_metric_gradient_vanishes()
  check_christoffel_vanishes()
  check_jacobian_second_derivative_symmetry()
  check_jacobian_third_derivative_symmetry()
  check_ricci_scalar_invariance()

  print("=" * 50)
  print("All sanity checks PASSED")
  print("=" * 50)
