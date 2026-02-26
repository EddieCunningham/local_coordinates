import jax
import jax.numpy as jnp
import jax.random as random
import pytest
from local_coordinates.metric import RiemannianMetric, raise_index, lower_index, pullback_metric
from local_coordinates.tensor import change_basis, change_coordinates
from local_coordinates.basis import BasisVectors, get_dual_basis_transform, get_standard_basis
from local_coordinates.jet import Jet
from local_coordinates.jacobian import function_to_jacobian

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


# ============================================================================
# Tests for metric evaluation: g(X, Y)
# ============================================================================

def test_metric_call_identity_metric():
  """
  Test that g(X, Y) = X · Y for the identity metric.
  """
  from local_coordinates.tangent import TangentVector

  p = jnp.array([0., 0.])
  basis = BasisVectors(p=p, components=Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2))
  metric = RiemannianMetric(basis=basis, components=Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2))

  X = TangentVector(p=p, components=Jet(value=jnp.array([1.0, 2.0]), gradient=None, hessian=None, dim=2), basis=basis)
  Y = TangentVector(p=p, components=Jet(value=jnp.array([3.0, 4.0]), gradient=None, hessian=None, dim=2), basis=basis)

  result = metric(X, Y)
  expected = 1.0 * 3.0 + 2.0 * 4.0  # X · Y = 11.0

  assert jnp.allclose(result.value, expected)


def test_metric_call_nontrivial_metric():
  """
  Test g(X, Y) = X^i g_{ij} Y^j for a non-trivial metric.
  """
  from local_coordinates.tangent import TangentVector

  p = jnp.array([0., 0.])
  basis = BasisVectors(p=p, components=Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2))
  g = jnp.array([[2.0, 0.5], [0.5, 1.0]])
  metric = RiemannianMetric(basis=basis, components=Jet(value=g, gradient=None, hessian=None, dim=2))

  X = TangentVector(p=p, components=Jet(value=jnp.array([1.0, 2.0]), gradient=None, hessian=None, dim=2), basis=basis)
  Y = TangentVector(p=p, components=Jet(value=jnp.array([3.0, 4.0]), gradient=None, hessian=None, dim=2), basis=basis)

  result = metric(X, Y)
  expected = jnp.einsum("i,ij,j->", jnp.array([1.0, 2.0]), g, jnp.array([3.0, 4.0]))

  assert jnp.allclose(result.value, expected)


def test_metric_call_symmetry():
  """
  Test that g(X, Y) = g(Y, X) (symmetry of metric).
  """
  from local_coordinates.tangent import TangentVector

  p = jnp.array([0., 0.])
  basis = BasisVectors(p=p, components=Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2))
  g = jnp.array([[2.0, 0.5], [0.5, 1.0]])
  metric = RiemannianMetric(basis=basis, components=Jet(value=g, gradient=None, hessian=None, dim=2))

  X = TangentVector(p=p, components=Jet(value=jnp.array([1.0, 2.0]), gradient=None, hessian=None, dim=2), basis=basis)
  Y = TangentVector(p=p, components=Jet(value=jnp.array([3.0, -1.0]), gradient=None, hessian=None, dim=2), basis=basis)

  result_XY = metric(X, Y)
  result_YX = metric(Y, X)

  assert jnp.allclose(result_XY.value, result_YX.value)


def test_metric_call_vectors_in_different_basis():
  """
  Test that g(X, Y) works when vectors are in a different basis than the metric.
  """
  from local_coordinates.tangent import TangentVector

  p = jnp.array([0., 0.])
  basis1 = BasisVectors(p=p, components=Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2))
  B2 = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  basis2 = BasisVectors(p=p, components=Jet(value=B2, gradient=None, hessian=None, dim=2))

  g = jnp.array([[2.0, 0.5], [0.5, 1.0]])
  metric = RiemannianMetric(basis=basis1, components=Jet(value=g, gradient=None, hessian=None, dim=2))

  # Create vectors in basis2
  X_comp = jnp.array([1.0, 0.0])
  Y_comp = jnp.array([0.0, 1.0])
  X = TangentVector(p=p, components=Jet(value=X_comp, gradient=None, hessian=None, dim=2), basis=basis2)
  Y = TangentVector(p=p, components=Jet(value=Y_comp, gradient=None, hessian=None, dim=2), basis=basis2)

  result = metric(X, Y)

  # Manually compute: transform X, Y to basis1, then compute g(X, Y)
  # X in basis2 has coords [1, 0], which means X = 1 * e1_basis2 = 1 * (1*e1 + 0*e2) = e1 in coords
  # Actually, basis2 columns are the basis vectors in coordinates
  # X = X_comp[0] * basis2[:,0] + X_comp[1] * basis2[:,1] in coordinates
  X_coords = B2 @ X_comp  # [1, 0] in standard coordinates
  Y_coords = B2 @ Y_comp  # [0.5, 1] in standard coordinates

  # Since metric is in basis1 (standard basis), we need X, Y components in basis1
  # For standard basis, components = coordinates
  expected = jnp.einsum("i,ij,j->", X_coords, g, Y_coords)

  assert jnp.allclose(result.value, expected)


def test_metric_call_with_gradients():
  """
  Test that g(X, Y) correctly computes gradients via jet arithmetic.
  """
  from local_coordinates.tangent import TangentVector

  dim = 2
  p = jnp.array([0., 0.])
  basis = BasisVectors(p=p, components=Jet(
    value=jnp.eye(dim),
    gradient=jnp.zeros((dim, dim, dim)),
    hessian=jnp.zeros((dim, dim, dim, dim))
  ))

  g_val = jnp.array([[2.0, 0.5], [0.5, 1.0]])
  g_grad = jnp.array([[[0.1, 0.0], [0.0, 0.1]], [[0.0, 0.1], [0.1, 0.0]]])
  metric = RiemannianMetric(basis=basis, components=Jet(
    value=g_val,
    gradient=g_grad,
    hessian=jnp.zeros((dim, dim, dim, dim))
  ))

  X_val = jnp.array([1.0, 0.0])
  Y_val = jnp.array([0.0, 1.0])
  X = TangentVector(p=p, components=Jet(
    value=X_val,
    gradient=jnp.zeros((dim, dim)),
    hessian=jnp.zeros((dim, dim, dim))
  ), basis=basis)
  Y = TangentVector(p=p, components=Jet(
    value=Y_val,
    gradient=jnp.zeros((dim, dim)),
    hessian=jnp.zeros((dim, dim, dim))
  ), basis=basis)

  result = metric(X, Y)

  # g(X, Y) = X^i g_{ij} Y^j = g_{01} = 0.5
  assert jnp.allclose(result.value, 0.5)

  # Gradient: d/dx^k (X^i g_{ij} Y^j) = X^i (dg_{ij}/dx^k) Y^j
  # = 1 * g_grad[0, 1, k] * 1 = g_grad[0, 1, :]
  expected_grad = g_grad[0, 1, :]
  assert jnp.allclose(result.gradient, expected_grad)


# ============================================================================
# Tests for change_coordinates on RiemannianMetric
# ============================================================================

def polar_to_cartesian(q):
  """Map from polar (r, phi) to Cartesian (x, y)."""
  r, phi = q[0], q[1]
  x = r * jnp.cos(phi)
  y = r * jnp.sin(phi)
  return jnp.array([x, y])

def cartesian_to_polar(p):
  """Map from Cartesian (x, y) to polar (r, phi)."""
  x, y = p[0], p[1]
  r = jnp.sqrt(x**2 + y**2)
  phi = jnp.arctan2(y, x)
  return jnp.array([r, phi])


def test_metric_change_coordinates_preserves_type():
  """
  Test that change_coordinates on a RiemannianMetric returns a RiemannianMetric.
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])

  basis = get_standard_basis(p_cart)
  metric_val = jnp.eye(dim)
  gradient = jnp.zeros((dim, dim, dim))
  hessian = jnp.zeros((dim, dim, dim, dim))
  metric = RiemannianMetric(
    basis=basis,
    components=Jet(value=metric_val, gradient=gradient, hessian=hessian)
  )

  jac = function_to_jacobian(cartesian_to_polar, p_cart)
  metric_polar = change_coordinates(metric, jac)

  assert isinstance(metric_polar, RiemannianMetric)


def test_metric_change_coordinates_value_unchanged():
  """
  Test that change_coordinates preserves the metric component values.
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])

  basis = get_standard_basis(p_cart)
  key = random.PRNGKey(42)
  k1, k2, k3 = random.split(key, 3)
  metric_val = random.normal(k1, (dim, dim))
  metric_val = metric_val @ metric_val.T  # Make symmetric positive definite
  gradient = random.normal(k2, (dim, dim, dim))
  hessian = random.normal(k3, (dim, dim, dim, dim))
  metric = RiemannianMetric(
    basis=basis,
    components=Jet(value=metric_val, gradient=gradient, hessian=hessian)
  )

  jac = function_to_jacobian(cartesian_to_polar, p_cart)
  metric_polar = change_coordinates(metric, jac)

  # Value should be unchanged
  assert jnp.allclose(metric_polar.components.value, metric_val)


def test_metric_change_coordinates_round_trip():
  """
  Test that changing coordinates and changing back gives the original metric.
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])
  p_polar = cartesian_to_polar(p_cart)

  basis = get_standard_basis(p_cart)
  key = random.PRNGKey(43)
  k1, k2, k3 = random.split(key, 3)
  metric_val = random.normal(k1, (dim, dim))
  metric_val = metric_val @ metric_val.T
  gradient = random.normal(k2, (dim, dim, dim))
  hessian = random.normal(k3, (dim, dim, dim, dim))
  metric = RiemannianMetric(
    basis=basis,
    components=Jet(value=metric_val, gradient=gradient, hessian=hessian)
  )

  # Round trip: Cartesian -> Polar -> Cartesian
  jac_to_polar = function_to_jacobian(cartesian_to_polar, p_cart)
  metric_polar = change_coordinates(metric, jac_to_polar)

  jac_to_cart = function_to_jacobian(polar_to_cartesian, p_polar)
  metric_back = change_coordinates(metric_polar, jac_to_cart)

  assert jnp.allclose(metric_back.components.value, metric.components.value)
  assert jnp.allclose(metric_back.components.gradient, metric.components.gradient, atol=1e-5)


def test_metric_change_coordinates_gradient_chain_rule():
  """
  Test that the gradient transforms according to the chain rule.
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])

  basis = get_standard_basis(p_cart)
  key = random.PRNGKey(44)
  k1, k2, k3 = random.split(key, 3)
  metric_val = random.normal(k1, (dim, dim))
  metric_val = metric_val @ metric_val.T
  gradient = random.normal(k2, (dim, dim, dim))
  hessian = random.normal(k3, (dim, dim, dim, dim))
  metric = RiemannianMetric(
    basis=basis,
    components=Jet(value=metric_val, gradient=gradient, hessian=hessian)
  )

  jac = function_to_jacobian(cartesian_to_polar, p_cart)
  metric_polar = change_coordinates(metric, jac)

  # Expected gradient via chain rule: dg/dz^k = J^a_k * dg/dx^a
  J_inv = jac.get_inverse()
  J = J_inv.value  # J[a,k] = dx^a/dz^k
  expected_gradient = jnp.einsum("ija,ak->ijk", gradient, J)

  assert jnp.allclose(metric_polar.components.gradient, expected_gradient)


def test_metric_change_coordinates_3d():
  """
  Test change_coordinates for a metric in 3D (spherical to Cartesian).
  """
  def spherical_to_cartesian(q):
    r, theta, phi = q[0], q[1], q[2]
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return jnp.array([x, y, z])

  def cartesian_to_spherical(p):
    x, y, z = p[0], p[1], p[2]
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.arccos(z / r)
    phi = jnp.arctan2(y, x)
    return jnp.array([r, theta, phi])

  dim = 3
  p_cart = jnp.array([1.0, 0.5, 0.3])
  p_sph = cartesian_to_spherical(p_cart)

  basis = get_standard_basis(p_cart)
  key = random.PRNGKey(45)
  k1, k2, k3 = random.split(key, 3)
  metric_val = random.normal(k1, (dim, dim))
  metric_val = metric_val @ metric_val.T
  gradient = random.normal(k2, (dim, dim, dim))
  hessian = random.normal(k3, (dim, dim, dim, dim))
  metric = RiemannianMetric(
    basis=basis,
    components=Jet(value=metric_val, gradient=gradient, hessian=hessian)
  )

  # Round trip
  jac_to_sph = function_to_jacobian(cartesian_to_spherical, p_cart)
  metric_sph = change_coordinates(metric, jac_to_sph)

  jac_to_cart = function_to_jacobian(spherical_to_cartesian, p_sph)
  metric_back = change_coordinates(metric_sph, jac_to_cart)

  assert isinstance(metric_back, RiemannianMetric)
  assert jnp.allclose(metric_back.components.value, metric.components.value)
  assert jnp.allclose(metric_back.components.gradient, metric.components.gradient, atol=1e-5)

# ============================================================================
# Tests for pullback_metric
# ============================================================================

def test_pullback_metric_identity_linear_map():
  """
  Test pullback of identity metric under a linear map f(x) = Ax is A^T A.
  """
  dim_m = 2
  dim_n = 1
  key = random.PRNGKey(46)
  A = random.normal(key, (dim_n, dim_m))

  def f(x):
    return A @ x

  p_m = jnp.array([1.0, 2.0])
  p_n = f(p_m)

  basis_m = get_standard_basis(p_m)
  basis_n = get_standard_basis(p_n)

  # Identity metric on N
  h = RiemannianMetric(
    basis=basis_n,
    components=Jet(value=jnp.eye(dim_n), gradient=jnp.zeros((dim_n, dim_n, dim_n)), hessian=jnp.zeros((dim_n, dim_n, dim_n, dim_n)))
  )

  # Pullback metric g = f*h
  g = pullback_metric(p_m, f, h)

  # Expected g = A^T I A = A^T A
  expected_g = A.T @ A
  assert jnp.allclose(g.components.value, expected_g)

def test_pullback_metric_polar_to_cartesian():
  """
  Test pullback of Euclidean metric under polar to cartesian map.
  The result should be the standard polar metric: diag(1, r^2).
  """
  def polar_to_cartesian_map(q):
    r, phi = q[0], q[1]
    return jnp.array([r * jnp.cos(phi), r * jnp.sin(phi)])

  r_val, phi_val = 2.0, jnp.pi / 4.0
  p_polar = jnp.array([r_val, phi_val])
  p_cart = polar_to_cartesian_map(p_polar)

  basis_cart = get_standard_basis(p_cart)

  # Euclidean metric on R^2
  h = RiemannianMetric(
    basis=basis_cart,
    components=Jet(value=jnp.eye(2), gradient=jnp.zeros((2, 2, 2)), hessian=jnp.zeros((2, 2, 2, 2)))
  )

  # Pullback metric g = f*h
  g = pullback_metric(p_polar, polar_to_cartesian_map, h)

  # Expected g = diag(1, r^2)
  expected_g = jnp.diag(jnp.array([1.0, r_val**2]))
  assert jnp.allclose(g.components.value, expected_g)

def test_pullback_metric_with_gradients():
  """
  Test that pullback_metric correctly propagates gradients of the map.
  """
  # Let f(x; theta) = theta * x, h = I.
  # g = theta^2 * I.
  # dg/dtheta = 2 * theta * I.

  def f_param(x, theta):
    return theta * x

  theta_val = 2.0
  p_m = jnp.array([1.0, 0.0])
  p_n = f_param(p_m, theta_val)

  basis_n = get_standard_basis(p_n)

  h = RiemannianMetric(
    basis=basis_n,
    components=Jet(value=jnp.eye(1), gradient=jnp.zeros((1, 1, 1)), hessian=jnp.zeros((1, 1, 1, 1)))
  )

  # We want to differentiate w.r.t. theta.
  # In our framework, we can wrap this in a function of theta.
  def get_pullback_value(theta):
    f = lambda x: f_param(x, theta)
    g = pullback_metric(p_m, f, h)
    return g.components.value

  dg_dtheta = jax.jacobian(get_pullback_value)(theta_val)

  # Expected dg/dtheta = 2 * theta * I = 2 * 2.0 * 1.0 = 4.0
  assert jnp.allclose(dg_dtheta, jnp.array([[4.0]]))

def test_compare_pullback_metric_computations():
  """
  Test pullback_metric with non-standard bases and random metric components.
  """
  key = random.PRNGKey(47)
  k1, k2, k3, k4, k5, k6 = random.split(key, 6)

  def f(q):
    r, phi = q[0], q[1]
    return jnp.array([r * jnp.cos(phi), r * jnp.sin(phi)])

  r_val, phi_val = 2.0, jnp.pi / 4.0
  p_polar = jnp.array([r_val, phi_val])
  p_cart = f(p_polar)

  # Non-standard bases: rotated/scaled
  E_cart_val = random.normal(k2, (2, 2))

  basis_cart = BasisVectors(
    p=p_cart,
    components=Jet(value=E_cart_val, gradient=jnp.zeros((2, 2, 2)), hessian=jnp.zeros((2, 2, 2, 2)))
  )

  # Euclidean metric components on N (Cartesian space)
  h = RiemannianMetric(
    basis=basis_cart,
    components=Jet(value=jnp.eye(2), gradient=jnp.zeros((2, 2, 2)), hessian=jnp.zeros((2, 2, 2, 2)))
  )

  # Pullback metric g = f*h
  g = pullback_metric(p_polar, f, h)


  # Expected g calculation:
  # G_jac = df/dq (Jacobian in standard coordinates)
  G_jac = jax.jacobian(f)(p_polar)

  # g_expected = G_jac^T @ h_standard @ G_jac
  # h_standard = inv(E_cart)^T @ h_val @ inv(E_cart)
  # But here h is defined in basis_cart, so we need to transform it to standard basis first
  h_standard = change_basis(h, get_standard_basis(p_cart)).components.value
  expected_g_val = G_jac.T @ h_standard @ G_jac

  assert jnp.allclose(g.components.value, expected_g_val)


# ============================================================================
# Tests for pullback_metric with non-dimension-preserving maps
# ============================================================================

def test_pullback_metric_dimension_reducing_projection():
  """
  Pullback of Euclidean metric on R^2 under projection R^3 -> R^2.
  The Jacobian is [[1,0,0],[0,1,0]], so the pullback is diag(1,1,0).
  """
  def project(x):
    return x[:2]

  x = jnp.array([1.0, 2.0, 3.0])
  y = project(x)

  h = RiemannianMetric(
    basis=get_standard_basis(y),
    components=Jet(
      value=jnp.eye(2),
      gradient=jnp.zeros((2, 2, 2)),
      hessian=jnp.zeros((2, 2, 2, 2)),
    ),
  )

  g = pullback_metric(x, project, h)

  J = jnp.array([[1., 0., 0.], [0., 1., 0.]])
  expected = J.T @ J
  assert g.components.value.shape == (3, 3)
  assert jnp.allclose(g.components.value, expected)


def test_pullback_metric_dimension_expanding_embedding():
  """
  Pullback of Euclidean metric on R^3 under a paraboloid embedding R^2 -> R^3.
  f(x) = (x0, x1, x0^2 + x1^2).  At x = (1, 0) the Jacobian is
  [[1,0],[0,1],[2,0]] so the pullback is [[5,0],[0,1]].
  """
  def embed(x):
    return jnp.array([x[0], x[1], x[0]**2 + x[1]**2])

  x = jnp.array([1.0, 0.0])
  y = embed(x)

  h = RiemannianMetric(
    basis=get_standard_basis(y),
    components=Jet(
      value=jnp.eye(3),
      gradient=jnp.zeros((3, 3, 3)),
      hessian=jnp.zeros((3, 3, 3, 3)),
    ),
  )

  g = pullback_metric(x, embed, h)

  J = jnp.array([[1., 0.], [0., 1.], [2., 0.]])
  expected = J.T @ J
  assert g.components.value.shape == (2, 2)
  assert jnp.allclose(g.components.value, expected)


def test_pullback_metric_dimension_reducing_linear():
  """
  Pullback of identity metric under a general linear map R^3 -> R^2.
  g = A^T A.
  """
  A = jnp.array([[1., 0., 1.], [0., 1., -1.]])

  def f(x):
    return A @ x

  x = jnp.array([1.0, 2.0, 3.0])
  y = f(x)

  h = RiemannianMetric(
    basis=get_standard_basis(y),
    components=Jet(
      value=jnp.eye(2),
      gradient=jnp.zeros((2, 2, 2)),
      hessian=jnp.zeros((2, 2, 2, 2)),
    ),
  )

  g = pullback_metric(x, f, h)

  expected = A.T @ A
  assert g.components.value.shape == (3, 3)
  assert jnp.allclose(g.components.value, expected)
