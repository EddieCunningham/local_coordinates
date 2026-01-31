"""
Sanity check for local_coordinates library.

This script demonstrates basic usage of the library by computing
geometric quantities for a simple position-dependent metric.
"""

import jax
import jax.numpy as jnp
from local_coordinates.jet import function_to_jet
from local_coordinates.basis import get_standard_basis
from local_coordinates.metric import RiemannianMetric
from local_coordinates.connection import get_levi_civita_connection
from local_coordinates.riemann import get_riemann_curvature_tensor, get_ricci_tensor

# Enable 64-bit precision (recommended for numerical stability)
jax.config.update("jax_enable_x64", True)


def metric_components(x):
  """A simple position-dependent metric."""
  return jnp.array([
    [1 + 0.1*x[0]**2, 0.0],
    [0.0, 1 + 0.1*x[1]**2]
  ])


def main():
  # Create the metric at a point
  p = jnp.array([1.0, 1.0])
  basis = get_standard_basis(p)
  metric_jet = function_to_jet(metric_components, p)
  metric = RiemannianMetric(basis=basis, components=metric_jet)

  # Compute geometric quantities
  connection = get_levi_civita_connection(metric)
  riemann = get_riemann_curvature_tensor(connection)
  ricci = get_ricci_tensor(connection, R=riemann)

  # Scalar curvature
  g_inv = jnp.linalg.inv(metric.components.value)
  scalar_curvature = jnp.einsum("ij,ij->", g_inv, ricci.components.value)

  print("=" * 60)
  print("local_coordinates Library Sanity Check")
  print("=" * 60)
  print(f"\nPoint: {p}")
  print(f"\nMetric at p:\n{metric.components.value}")
  print(f"\nChristoffel symbols shape: {connection.christoffel_symbols.value.shape}")
  print(f"Riemann tensor shape: {riemann.components.value.shape}")
  print(f"Ricci tensor shape: {ricci.components.value.shape}")
  print(f"\nScalar curvature: {scalar_curvature}")
  print("\nSanity check passed!")
  print("=" * 60)


if __name__ == "__main__":
  main()
