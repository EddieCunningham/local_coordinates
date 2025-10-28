import jax.numpy as jnp
from jax import random
import numpy as np

from local_coordinates.metric import RiemannianMetric, lower_index
from local_coordinates.basis import get_standard_basis
from local_coordinates.jet import function_to_jet
from local_coordinates.connection import get_levi_civita_connection
from local_coordinates.riemann import get_riemann_curvature_tensor
from local_coordinates.basis import BasisVectors
from local_coordinates.frame import get_lie_bracket_between_frame_pairs, basis_to_frame
from local_coordinates.jet import Jet, jet_decorator, get_identity_jet
from local_coordinates.tensor import change_basis
from local_coordinates.tangent import TangentVector, lie_bracket
from jaxtyping import Array
from typing import Annotated

def create_random_basis(key: random.PRNGKey, dim: int) -> BasisVectors:
  p_key, vals_key, grads_key, hessians_key = random.split(key, 4)
  p = jnp.zeros(dim)
  vals = random.normal(vals_key, (dim, dim))*0.1
  grads = random.normal(grads_key, (dim, dim, dim))*0.1
  hessians = random.normal(hessians_key, (dim, dim, dim, dim))*0.1
  return BasisVectors(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians))

def create_random_metric(key: random.PRNGKey, dim: int) -> RiemannianMetric:
  random_basis = create_random_basis(key, dim)
  return RiemannianMetric(basis=random_basis, components=get_identity_jet(dim))

def create_random_vector_field(key: random.PRNGKey, dim: int) -> TangentVector:
  p_key, basis_key, vals_key, grads_key, hessians_key = random.split(key, 5)
  p = jnp.zeros(dim)
  random_basis = create_random_basis(basis_key, dim)
  vals = random.normal(vals_key, (dim,))
  grads = random.normal(grads_key, (dim, dim))
  hessians = random.normal(hessians_key, (dim, dim, dim))
  return TangentVector(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians), basis=random_basis)

def test_riemann_curvature_tensor_definition():
  """
  Test that the Riemann curvature tensor is defined correctly
  by comparing it to the definition using the covariant derivative.
  """
  key = random.PRNGKey(42)
  dim = 5
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)

  k1, k2, k3 = random.split(key, 3)
  X = change_basis(create_random_vector_field(k1, dim), connection.basis)
  Y = change_basis(create_random_vector_field(k2, dim), connection.basis)
  Z = change_basis(create_random_vector_field(k3, dim), connection.basis)

  nablaY_Z = connection.covariant_derivative(Y, Z)
  nablaX_Z = connection.covariant_derivative(X, Z)
  bracket_XY = lie_bracket(X, Y)
  nablaX_nablaY_Z = connection.covariant_derivative(X, nablaY_Z)
  nablaY_nablaX_Z = connection.covariant_derivative(Y, nablaX_Z)
  nabla_bracket_XY_Z = connection.covariant_derivative(bracket_XY, Z)

  R_XYZ = nablaX_nablaY_Z - nablaY_nablaX_Z - nabla_bracket_XY_Z

  # Construct the Riemann curvature tensor
  riemann_tensor = get_riemann_curvature_tensor(connection)

  @jet_decorator
  def apply_riemann_tensor(R_val: Array, X_val: Array, Y_val: Array, Z_val: Array) -> Array:
    # einsum is for R_{ijk}^l X^i Y^j Z^k
    return jnp.einsum("ijkl,i,j,k->l", R_val, X_val, Y_val, Z_val)

  R_val = riemann_tensor.components.get_value_jet()
  X_val = X.components.get_value_jet()
  Y_val = Y.components.get_value_jet()
  Z_val = Z.components.get_value_jet()
  out = apply_riemann_tensor(R_val, X_val, Y_val, Z_val)

  assert jnp.allclose(R_XYZ.components.value, out.value)


def test_riemann_symmetries():
  """
  Tests that the calculated RiemannCurvatureTensor satisfies its
  fundamental symmetries for a randomly generated metric.
  """
  key = random.PRNGKey(42)
  dim = 5
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)
  k1, k2, k3 = random.split(key, 3)
  X = change_basis(create_random_vector_field(k1, dim), connection.basis)
  Y = change_basis(create_random_vector_field(k2, dim), connection.basis)
  Z = change_basis(create_random_vector_field(k3, dim), connection.basis)

  riemann_tensor = get_riemann_curvature_tensor(connection)
  R_lower = lower_index(riemann_tensor, metric, 4)

  R = R_lower.components.value  # Components R_{ijkl}

  # Skew symmetry 1
  assert jnp.allclose(R, -R.swapaxes(0, 1))

  # Skew symmetry 2
  assert jnp.allclose(R, -R.swapaxes(-1, -2))

  # Interchange
  assert jnp.allclose(R, R.transpose((2, 3, 0, 1)))

  # First Bianchi identity
  assert jnp.allclose(R + R.transpose((0, 2, 3, 1)) + R.transpose((0, 3, 1, 2)), 0.0)

def test_ricci_scalar_basis_independence():
  """
  Tests that the Ricci scalar is independent of the basis chosen.
  """
  key = random.PRNGKey(42)
  dim = 5
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)
  riemann_tensor = get_riemann_curvature_tensor(connection)

  metric_standard = change_basis(metric, get_standard_basis(metric.basis.p))
  riemann_tensor_standard = get_riemann_curvature_tensor(get_levi_civita_connection(metric_standard))


  # Lower the upper index: R_{ijkl}
  R_lower = lower_index(riemann_tensor, metric, 4)
  R = R_lower.components.value  # (i,j,k,l)
  g = metric.components.value   # (i,j)
  g_inv = jnp.linalg.inv(g)
  scalar_curvature = jnp.einsum("ijkl,il,jk->", R, g_inv, g_inv)

  R_lower_standard = lower_index(riemann_tensor_standard, metric_standard, 4)
  R_standard = R_lower_standard.components.value  # (i,j,k,l)
  g_standard = metric_standard.components.value   # (i,j)
  g_inv_standard = jnp.linalg.inv(g_standard)
  scalar_curvature_standard = jnp.einsum("ijkl,il,jk->", R_standard, g_inv_standard, g_inv_standard)

  assert jnp.allclose(scalar_curvature, scalar_curvature_standard)