import jax.numpy as jnp
import pytest
from local_coordinates.metric import RiemannianMetric, raise_index, lower_index
from local_coordinates.tensor import change_basis
# from local_coordinates.tensor import change_basis
from local_coordinates.basis import BasisVectors, get_dual_basis_transform
from local_coordinates.jet import Jet

def test_riemannian_metric_creation():
  """
  Tests the creation of a simple RiemannianMetric instance.
  """
  p = jnp.array([1., 2.])
  basis_components = Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2)
  basis = BasisVectors(p=p, components=basis_components)

  metric_components_jet = Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2)
  metric = RiemannianMetric(basis=basis, components=metric_components_jet)

  assert metric.basis is basis
  assert jnp.array_equal(metric.components.value, metric_components_jet.value)
  assert metric.batch_size is None

def test_metric_creation_fails_with_non_jet_components():
  """
  Tests that creating a RiemannianMetric with non-Jet components raises an error.
  """
  p = jnp.array([1., 2.])
  basis_components = Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2)
  basis = BasisVectors(p=p, components=basis_components)

  metric_components = jnp.eye(2) # Not a Jet

  with pytest.raises(AssertionError):
    RiemannianMetric(basis=basis, components=metric_components)

def test_metric_creation_fails_with_wrong_ndim():
  """
  Tests that creating a RiemannianMetric with wrong ndim for components raises an error.
  """
  p = jnp.array([1., 2.])
  basis_components = Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2)
  basis = BasisVectors(p=p, components=basis_components)

  metric_components_jet = Jet(value=jnp.ones((2, 2, 2)), gradient=None, hessian=None, dim=2)

  with pytest.raises(ValueError):
    RiemannianMetric(basis=basis, components=metric_components_jet)

def test_metric_creation_fails_with_non_square_components():
  """
  Tests that creating a RiemannianMetric with non-square components raises an error.
  """
  p = jnp.array([1., 2.])
  basis_components = Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2)
  basis = BasisVectors(p=p, components=basis_components)

  metric_components_jet = Jet(value=jnp.ones((2, 3)), gradient=None, hessian=None, dim=2)

  with pytest.raises(ValueError):
    RiemannianMetric(basis=basis, components=metric_components_jet)

def test_metric_batching():
  """
  Tests the creation of a batched RiemannianMetric instance.
  """
  p_batch = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
  basis_components_jet = Jet(value=jnp.stack([jnp.eye(2)] * 3), gradient=None, hessian=None, dim=2)
  basis = BasisVectors(p=p_batch, components=basis_components_jet)

  metric_components_jet = Jet(value=jnp.stack([jnp.eye(2)] * 3), gradient=None, hessian=None, dim=2)
  metric = RiemannianMetric(basis=basis, components=metric_components_jet)

  assert metric.batch_size == 3
  assert metric.basis.p.shape == (3, 2)
  assert metric.components.value.shape == (3, 2, 2)


def test_change_coordinates_metric_dual_basis():
  p = jnp.array([0., 0.])
  # Define two dual bases from vector bases B1, B2
  B1 = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  B2 = jnp.array([[0.0, 1.0], [1.0, 0.0]])
  theta1 = BasisVectors(p=p, components=Jet(value=B1, gradient=None, hessian=None, dim=2))
  theta2 = BasisVectors(p=p, components=Jet(value=B2, gradient=None, hessian=None, dim=2))

  # Metric in basis theta1
  g = jnp.array([[2.0, 0.5], [0.5, 1.0]])
  metric = RiemannianMetric(basis=theta1, components=Jet(value=g, gradient=None, hessian=None, dim=2))

  # Transform
  metric2 = change_basis(metric, theta2)

  # Expected: g' = T_dual^T g T_dual, where T_dual = inv(B1) @ B2
  T_dual = get_dual_basis_transform(theta1, theta2).value
  expected = T_dual.T @ g @ T_dual
  assert jnp.allclose(metric2.components.value, expected)


def test_raise_index_covector_to_vector():
  # Basis
  p = jnp.array([0., 0.])
  basis = BasisVectors(p=p, components=Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2))

  # Metric (nontrivial)
  g = jnp.array([[2.0, 0.5], [0.5, 1.0]])
  metric = RiemannianMetric(basis=basis, components=Jet(value=g, gradient=None, hessian=None, dim=2))

  # Covector (k=1,l=0)
  from local_coordinates.tensor import Tensor, TensorType
  covec = jnp.array([1.0, -1.0])
  covec_jet = Jet(value=covec, gradient=None, hessian=None, dim=2)
  covec_tensor = Tensor(tensor_type=TensorType(k=1, l=0), basis=basis, components=covec_jet)

  # Raise overall index 1 (the only covariant index)
  raised = raise_index(covec_tensor, metric, index=1)

  # Expected: v^i = g^{ij} alpha_j
  g_inv = jnp.linalg.inv(g)
  expected = g_inv @ covec

  assert raised.tensor_type.k == 0 and raised.tensor_type.l == 1
  assert jnp.allclose(raised.components.value, expected)


def test_lower_index_vector_to_covector():
  # Basis
  p = jnp.array([0., 0.])
  basis = BasisVectors(p=p, components=Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2))

  # Metric (nontrivial)
  g = jnp.array([[2.0, 0.5], [0.5, 1.0]])
  metric = RiemannianMetric(basis=basis, components=Jet(value=g, gradient=None, hessian=None, dim=2))

  # Vector (k=0,l=1)
  from local_coordinates.tensor import Tensor, TensorType
  vec = jnp.array([1.0, -1.0])
  vec_jet = Jet(value=vec, gradient=None, hessian=None, dim=2)
  vec_tensor = Tensor(tensor_type=TensorType(k=0, l=1), basis=basis, components=vec_jet)

  # Lower overall index 1 (the only contravariant index)
  lowered = lower_index(vec_tensor, metric, index=1)

  # Expected: alpha_i = g_{ij} v^j
  expected = g @ vec

  assert lowered.tensor_type.k == 1 and lowered.tensor_type.l == 0
  assert jnp.allclose(lowered.components.value, expected)
