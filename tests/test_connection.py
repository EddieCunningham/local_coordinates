import jax.numpy as jnp
from local_coordinates.basis import get_standard_basis
from local_coordinates.connection import Connection, get_levi_civita_connection
from local_coordinates.jet import Jet
from local_coordinates.metric import RiemannianMetric


def test_connection_basic_construction():
  p = jnp.array([0.0, 0.0])
  basis = get_standard_basis(p)

  Gamma = jnp.zeros((2, 2, 2))
  Gamma_jet = Jet(value=Gamma, gradient=None, hessian=None)

  conn = Connection(basis=basis, christoffel_symbols=Gamma_jet)

  assert conn.basis is basis
  assert conn.christoffel_symbols.shape == (2, 2, 2)
  assert jnp.allclose(conn.christoffel_symbols.value, 0.0)


def test_get_levi_civita_connection_euclidean_cartesian_zero():
  p = jnp.array([0.3, -1.2])
  basis = get_standard_basis(p)

  # Euclidean metric in Cartesian coordinates: identity with zero derivatives
  metric_components = Jet(
      value=jnp.eye(2),
      gradient=jnp.zeros((2, 2, 2)),
    hessian=jnp.zeros((2, 2, 2, 2)),
  )
  metric = RiemannianMetric(basis=basis, components=metric_components)

  conn = get_levi_civita_connection(metric)

  # Compare to known zero connection in this chart
  assert conn.christoffel_symbols.shape == (2, 2, 2)
  assert jnp.allclose(conn.christoffel_symbols.value, 0.0)