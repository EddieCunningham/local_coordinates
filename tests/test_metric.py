import jax.numpy as jnp
import pytest
from local_coordinates.metric import RiemannianMetric
from local_coordinates.basis import BasisVectors
from local_coordinates.jet import Jet

def test_riemannian_metric_creation():
  """
  Tests the creation of a simple RiemannianMetric instance.
  """
  p = jnp.array([1., 2.])
  basis_components = Jet(value=jnp.eye(2), gradient=None, hessian=None)
  basis = BasisVectors(p=p, components=basis_components)

  metric_components_jet = Jet(value=jnp.eye(2), gradient=None, hessian=None)
  metric = RiemannianMetric(basis=basis, components=metric_components_jet)

  assert metric.basis is basis
  assert jnp.array_equal(metric.components.value, metric_components_jet.value)
  assert metric.batch_size is None

def test_metric_creation_fails_with_non_jet_components():
  """
  Tests that creating a RiemannianMetric with non-Jet components raises an error.
  """
  p = jnp.array([1., 2.])
  basis_components = Jet(value=jnp.eye(2), gradient=None, hessian=None)
  basis = BasisVectors(p=p, components=basis_components)

  metric_components = jnp.eye(2) # Not a Jet

  with pytest.raises(AssertionError):
    RiemannianMetric(basis=basis, components=metric_components)

def test_metric_creation_fails_with_wrong_ndim():
  """
  Tests that creating a RiemannianMetric with wrong ndim for components raises an error.
  """
  p = jnp.array([1., 2.])
  basis_components = Jet(value=jnp.eye(2), gradient=None, hessian=None)
  basis = BasisVectors(p=p, components=basis_components)

  metric_components_jet = Jet(value=jnp.ones((2, 2, 2)), gradient=None, hessian=None)

  with pytest.raises(ValueError):
    RiemannianMetric(basis=basis, components=metric_components_jet)

def test_metric_creation_fails_with_non_square_components():
  """
  Tests that creating a RiemannianMetric with non-square components raises an error.
  """
  p = jnp.array([1., 2.])
  basis_components = Jet(value=jnp.eye(2), gradient=None, hessian=None)
  basis = BasisVectors(p=p, components=basis_components)

  metric_components_jet = Jet(value=jnp.ones((2, 3)), gradient=None, hessian=None)

  with pytest.raises(ValueError):
    RiemannianMetric(basis=basis, components=metric_components_jet)

def test_metric_batching():
  """
  Tests the creation of a batched RiemannianMetric instance.
  """
  p_batch = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
  basis_components_jet = Jet(value=jnp.stack([jnp.eye(2)] * 3), gradient=None, hessian=None)
  basis = BasisVectors(p=p_batch, components=basis_components_jet)

  metric_components_jet = Jet(value=jnp.stack([jnp.eye(2)] * 3), gradient=None, hessian=None)
  metric = RiemannianMetric(basis=basis, components=metric_components_jet)

  assert metric.batch_size == 3
  assert metric.basis.p.shape == (3, 2)
  assert metric.components.value.shape == (3, 2, 2)
