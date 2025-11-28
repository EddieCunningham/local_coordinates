import jax.numpy as jnp
from jax import random
import numpy as np

from local_coordinates.metric import RiemannianMetric, lower_index
from local_coordinates.basis import get_standard_basis
from local_coordinates.jet import function_to_jet
from local_coordinates.connection import get_levi_civita_connection
from local_coordinates.riemann import RiemannCurvatureTensor, get_riemann_curvature_tensor, RicciTensor, get_ricci_tensor
from local_coordinates.basis import BasisVectors
from local_coordinates.frame import get_lie_bracket_between_frame_pairs, basis_to_frame
from local_coordinates.jet import Jet, jet_decorator, get_identity_jet
from local_coordinates.tensor import TensorType, change_basis, change_coordinates
from local_coordinates.tangent import TangentVector, lie_bracket
from local_coordinates.jacobian import function_to_jacobian
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


def test_kretschmann_zero_iff_flat_and_basis_invariant():
  """
  Kretschmann scalar K = R_{abcd} R^{abcd} is:
  - Nonnegative and coordinate-free
  - Zero iff the full Riemann tensor vanishes

  For the Euclidean metric (identity components), curvature must vanish.
  We compute K and verify it is ~0, that Riemann is ~0, and that K is
  invariant under change of basis to the standard basis.
  """
  key = random.PRNGKey(0)
  dim = 4

  # Build a truly flat metric: identity components in the standard basis
  p = jnp.zeros(dim)
  metric = RiemannianMetric(basis=get_standard_basis(p), components=get_identity_jet(dim))
  connection = get_levi_civita_connection(metric)
  riemann_tensor = get_riemann_curvature_tensor(connection)

  # Lower the upper index to get R_{ijkl}
  R_lower = lower_index(riemann_tensor, metric, 4)
  R = R_lower.components.value  # (i,j,k,l)
  g = metric.components.value   # (i,j)
  g_inv = jnp.linalg.inv(g)

  # Kretschmann scalar K = R_{ijkl} R_{abcd} g^{ia} g^{jb} g^{kc} g^{ld}
  K = jnp.einsum("ijkl,abcd,ia,jb,kc,ld->", R, R, g_inv, g_inv, g_inv, g_inv)

  # In a flat Euclidean metric, curvature must vanish
  assert jnp.allclose(K, 0.0)
  assert jnp.allclose(R, 0.0)

  # Basis invariance: transform to a random smooth basis and recompute K
  metric_std = change_basis(metric, create_random_basis(key, dim))
  connection_std = get_levi_civita_connection(metric_std)
  riemann_std = get_riemann_curvature_tensor(connection_std)
  R_lower_std = lower_index(riemann_std, metric_std, 4)
  R_std = R_lower_std.components.value
  g_std = metric_std.components.value
  g_inv_std = jnp.linalg.inv(g_std)
  K_std = jnp.einsum("ijkl,abcd,ia,jb,kc,ld->", R_std, R_std, g_inv_std, g_inv_std, g_inv_std, g_inv_std)

  assert jnp.allclose(K, K_std)


# ============================================================================
# Tests for Riemann tensor evaluation: R(X, Y, Z)
# ============================================================================

def test_riemann_call_basic():
  """
  Test that R(X, Y, Z) computes R_{ijk}^l X^i Y^j Z^k correctly.
  """
  key = random.PRNGKey(60)
  dim = 4
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)
  riemann_tensor = get_riemann_curvature_tensor(connection)

  k1, k2, k3 = random.split(key, 3)
  X = change_basis(create_random_vector_field(k1, dim), riemann_tensor.basis)
  Y = change_basis(create_random_vector_field(k2, dim), riemann_tensor.basis)
  Z = change_basis(create_random_vector_field(k3, dim), riemann_tensor.basis)

  result = riemann_tensor(X, Y, Z)

  # Result should be a TangentVector
  assert isinstance(result, TangentVector)

  # Manual computation
  R_val = riemann_tensor.components.value
  X_val = X.components.value
  Y_val = Y.components.value
  Z_val = Z.components.value
  expected = jnp.einsum("ijkl,i,j,k->l", R_val, X_val, Y_val, Z_val)

  assert jnp.allclose(result.components.value, expected)


def test_riemann_call_skew_symmetry():
  """
  Test that R(X, Y, Z) = -R(Y, X, Z) (skew-symmetry in first two arguments).
  """
  key = random.PRNGKey(61)
  dim = 4
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)
  riemann_tensor = get_riemann_curvature_tensor(connection)

  k1, k2, k3 = random.split(key, 3)
  X = change_basis(create_random_vector_field(k1, dim), riemann_tensor.basis)
  Y = change_basis(create_random_vector_field(k2, dim), riemann_tensor.basis)
  Z = change_basis(create_random_vector_field(k3, dim), riemann_tensor.basis)

  R_XYZ = riemann_tensor(X, Y, Z)
  R_YXZ = riemann_tensor(Y, X, Z)

  assert jnp.allclose(R_XYZ.components.value, -R_YXZ.components.value)


def test_riemann_call_vectors_in_different_basis():
  """
  Test that R(X, Y, Z) works when vectors are in a different basis than R.
  """
  key = random.PRNGKey(62)
  dim = 4
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)
  riemann_tensor = get_riemann_curvature_tensor(connection)

  k1, k2, k3 = random.split(key, 3)
  # Create vectors in their own random bases (not riemann_tensor.basis)
  X = create_random_vector_field(k1, dim)
  Y = create_random_vector_field(k2, dim)
  Z = create_random_vector_field(k3, dim)

  result = riemann_tensor(X, Y, Z)

  # Manually change basis and compute
  X_cb = change_basis(X, riemann_tensor.basis)
  Y_cb = change_basis(Y, riemann_tensor.basis)
  Z_cb = change_basis(Z, riemann_tensor.basis)
  R_val = riemann_tensor.components.value
  expected = jnp.einsum("ijkl,i,j,k->l", R_val, X_cb.components.value, Y_cb.components.value, Z_cb.components.value)

  assert jnp.allclose(result.components.value, expected)


def test_riemann_call_matches_definition():
  """
  Test that R(X, Y, Z) matches the definition via covariant derivatives:
  R(X, Y)Z = nabla_X nabla_Y Z - nabla_Y nabla_X Z - nabla_[X,Y] Z
  """
  key = random.PRNGKey(63)
  dim = 4
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)
  riemann_tensor = get_riemann_curvature_tensor(connection)

  k1, k2, k3 = random.split(key, 3)
  X = change_basis(create_random_vector_field(k1, dim), connection.basis)
  Y = change_basis(create_random_vector_field(k2, dim), connection.basis)
  Z = change_basis(create_random_vector_field(k3, dim), connection.basis)

  # Compute via covariant derivatives
  nablaY_Z = connection.covariant_derivative(Y, Z)
  nablaX_Z = connection.covariant_derivative(X, Z)
  bracket_XY = lie_bracket(X, Y)
  nablaX_nablaY_Z = connection.covariant_derivative(X, nablaY_Z)
  nablaY_nablaX_Z = connection.covariant_derivative(Y, nablaX_Z)
  nabla_bracket_XY_Z = connection.covariant_derivative(bracket_XY, Z)
  R_XYZ_def = nablaX_nablaY_Z.components.value - nablaY_nablaX_Z.components.value - nabla_bracket_XY_Z.components.value

  # Compute via __call__
  R_XYZ = riemann_tensor(X, Y, Z)

  assert jnp.allclose(R_XYZ.components.value, R_XYZ_def)


def test_riemann_call_with_gradients():
  """
  Test that R(X, Y, Z) returns a TangentVector with correctly shaped gradient.
  """
  key = random.PRNGKey(64)
  dim = 3
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)
  riemann_tensor = get_riemann_curvature_tensor(connection)

  k1, k2, k3 = random.split(key, 3)
  X = change_basis(create_random_vector_field(k1, dim), riemann_tensor.basis)
  Y = change_basis(create_random_vector_field(k2, dim), riemann_tensor.basis)
  Z = change_basis(create_random_vector_field(k3, dim), riemann_tensor.basis)

  result = riemann_tensor(X, Y, Z)

  # The result should be a TangentVector with gradient of correct shape
  assert isinstance(result, TangentVector)
  assert result.components.gradient is not None
  assert result.components.value.shape == (dim,)  # R(X,Y,Z) is a vector
  assert result.components.gradient.shape == (dim, dim)  # gradient of vector is (dim, dim)


def test_riemann_call_first_bianchi():
  """
  Test the first Bianchi identity via __call__:
  R(X, Y)Z + R(Y, Z)X + R(Z, X)Y = 0
  """
  key = random.PRNGKey(65)
  dim = 4
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)
  riemann_tensor = get_riemann_curvature_tensor(connection)

  k1, k2, k3 = random.split(key, 3)
  X = change_basis(create_random_vector_field(k1, dim), riemann_tensor.basis)
  Y = change_basis(create_random_vector_field(k2, dim), riemann_tensor.basis)
  Z = change_basis(create_random_vector_field(k3, dim), riemann_tensor.basis)

  R_XYZ = riemann_tensor(X, Y, Z)
  R_YZX = riemann_tensor(Y, Z, X)
  R_ZXY = riemann_tensor(Z, X, Y)

  bianchi_sum = R_XYZ.components.value + R_YZX.components.value + R_ZXY.components.value

  assert jnp.allclose(bianchi_sum, 0.0)


# ============================================================================
# Tests for change_coordinates on RiemannCurvatureTensor
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


def test_riemann_change_coordinates_preserves_type():
  """
  Test that change_coordinates on a RiemannCurvatureTensor returns a RiemannCurvatureTensor.
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])

  basis = get_standard_basis(p_cart)
  key = random.PRNGKey(50)
  k1, k2, k3 = random.split(key, 3)
  riemann_val = random.normal(k1, (dim, dim, dim, dim))
  gradient = random.normal(k2, (dim, dim, dim, dim, dim))
  hessian = random.normal(k3, (dim, dim, dim, dim, dim, dim))
  riemann = RiemannCurvatureTensor(
    tensor_type=TensorType(k=3, l=1),
    basis=basis,
    components=Jet(value=riemann_val, gradient=gradient, hessian=hessian)
  )

  jac = function_to_jacobian(cartesian_to_polar, p_cart)
  riemann_polar = change_coordinates(riemann, jac)

  assert isinstance(riemann_polar, RiemannCurvatureTensor)


def test_riemann_change_coordinates_value_unchanged():
  """
  Test that change_coordinates preserves the Riemann tensor component values.
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])

  basis = get_standard_basis(p_cart)
  key = random.PRNGKey(51)
  k1, k2, k3 = random.split(key, 3)
  riemann_val = random.normal(k1, (dim, dim, dim, dim))
  gradient = random.normal(k2, (dim, dim, dim, dim, dim))
  hessian = random.normal(k3, (dim, dim, dim, dim, dim, dim))
  riemann = RiemannCurvatureTensor(
    tensor_type=TensorType(k=3, l=1),
    basis=basis,
    components=Jet(value=riemann_val, gradient=gradient, hessian=hessian)
  )

  jac = function_to_jacobian(cartesian_to_polar, p_cart)
  riemann_polar = change_coordinates(riemann, jac)

  # Value should be unchanged
  assert jnp.allclose(riemann_polar.components.value, riemann_val)


def test_riemann_change_coordinates_round_trip():
  """
  Test that changing coordinates and changing back gives the original tensor.
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])
  p_polar = cartesian_to_polar(p_cart)

  basis = get_standard_basis(p_cart)
  key = random.PRNGKey(52)
  k1, k2, k3 = random.split(key, 3)
  riemann_val = random.normal(k1, (dim, dim, dim, dim))
  gradient = random.normal(k2, (dim, dim, dim, dim, dim))
  hessian = random.normal(k3, (dim, dim, dim, dim, dim, dim))
  riemann = RiemannCurvatureTensor(
    tensor_type=TensorType(k=3, l=1),
    basis=basis,
    components=Jet(value=riemann_val, gradient=gradient, hessian=hessian)
  )

  # Round trip: Cartesian -> Polar -> Cartesian
  jac_to_polar = function_to_jacobian(cartesian_to_polar, p_cart)
  riemann_polar = change_coordinates(riemann, jac_to_polar)

  jac_to_cart = function_to_jacobian(polar_to_cartesian, p_polar)
  riemann_back = change_coordinates(riemann_polar, jac_to_cart)

  assert jnp.allclose(riemann_back.components.value, riemann.components.value)
  assert jnp.allclose(riemann_back.components.gradient, riemann.components.gradient, atol=1e-5)


def test_riemann_change_coordinates_gradient_chain_rule():
  """
  Test that the gradient transforms according to the chain rule.
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])

  basis = get_standard_basis(p_cart)
  key = random.PRNGKey(53)
  k1, k2, k3 = random.split(key, 3)
  riemann_val = random.normal(k1, (dim, dim, dim, dim))
  gradient = random.normal(k2, (dim, dim, dim, dim, dim))
  hessian = random.normal(k3, (dim, dim, dim, dim, dim, dim))
  riemann = RiemannCurvatureTensor(
    tensor_type=TensorType(k=3, l=1),
    basis=basis,
    components=Jet(value=riemann_val, gradient=gradient, hessian=hessian)
  )

  jac = function_to_jacobian(cartesian_to_polar, p_cart)
  riemann_polar = change_coordinates(riemann, jac)

  # Expected gradient via chain rule: dR/dz^k = J^a_k * dR/dx^a
  J_inv = jac.get_inverse()
  J = J_inv.value  # J[a,k] = dx^a/dz^k
  expected_gradient = jnp.einsum("ijkla,am->ijklm", gradient, J)

  assert jnp.allclose(riemann_polar.components.gradient, expected_gradient)


def test_riemann_change_coordinates_3d():
  """
  Test change_coordinates for Riemann tensor in 3D (spherical to Cartesian).
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
  key = random.PRNGKey(54)
  k1, k2, k3 = random.split(key, 3)
  riemann_val = random.normal(k1, (dim, dim, dim, dim))
  gradient = random.normal(k2, (dim, dim, dim, dim, dim))
  hessian = random.normal(k3, (dim, dim, dim, dim, dim, dim))
  riemann = RiemannCurvatureTensor(
    tensor_type=TensorType(k=3, l=1),
    basis=basis,
    components=Jet(value=riemann_val, gradient=gradient, hessian=hessian)
  )

  # Round trip
  jac_to_sph = function_to_jacobian(cartesian_to_spherical, p_cart)
  riemann_sph = change_coordinates(riemann, jac_to_sph)

  jac_to_cart = function_to_jacobian(spherical_to_cartesian, p_sph)
  riemann_back = change_coordinates(riemann_sph, jac_to_cart)

  assert isinstance(riemann_back, RiemannCurvatureTensor)
  assert jnp.allclose(riemann_back.components.value, riemann.components.value)
  assert jnp.allclose(riemann_back.components.gradient, riemann.components.gradient, atol=1e-5)


def test_riemann_tensor_type_preserved():
  """
  Test that the tensor type (3,1) is preserved after change_coordinates.
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])

  basis = get_standard_basis(p_cart)
  key = random.PRNGKey(55)
  k1, k2, k3 = random.split(key, 3)
  riemann_val = random.normal(k1, (dim, dim, dim, dim))
  gradient = random.normal(k2, (dim, dim, dim, dim, dim))
  hessian = random.normal(k3, (dim, dim, dim, dim, dim, dim))
  riemann = RiemannCurvatureTensor(
    tensor_type=TensorType(k=3, l=1),
    basis=basis,
    components=Jet(value=riemann_val, gradient=gradient, hessian=hessian)
  )

  jac = function_to_jacobian(cartesian_to_polar, p_cart)
  riemann_polar = change_coordinates(riemann, jac)

  assert riemann_polar.tensor_type.k == 3
  assert riemann_polar.tensor_type.l == 1


# ============================================================================
# Tests for Ricci tensor
# ============================================================================

def test_ricci_tensor_returns_correct_type():
  """
  Test that get_ricci_tensor returns a RicciTensor with correct tensor type.
  """
  key = random.PRNGKey(70)
  dim = 4
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)

  ricci = get_ricci_tensor(connection)

  assert isinstance(ricci, RicciTensor)
  assert ricci.tensor_type.k == 2
  assert ricci.tensor_type.l == 0
  assert ricci.components.value.shape == (dim, dim)


def test_ricci_tensor_contraction():
  """
  Test that Ricci tensor is the correct contraction of Riemann: R_{ab} = R^i_{aib}.
  """
  key = random.PRNGKey(71)
  dim = 4
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)
  riemann = get_riemann_curvature_tensor(connection)

  ricci = get_ricci_tensor(connection, R=riemann)

  # Manual contraction: R_{ab} = R^i_{aib} = R_{iabi} with upper index last
  R = riemann.components.value
  expected = jnp.einsum("iabi->ab", R)

  assert jnp.allclose(ricci.components.value, expected)


def test_ricci_tensor_symmetry():
  """
  Test that Ricci tensor is symmetric for Levi-Civita connection.
  """
  key = random.PRNGKey(72)
  dim = 5
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)

  ricci = get_ricci_tensor(connection)
  Ric = ricci.components.value

  assert jnp.allclose(Ric, Ric.T)


def test_ricci_tensor_with_and_without_precomputed_riemann():
  """
  Test that passing R explicitly gives the same result as computing it internally.
  """
  key = random.PRNGKey(73)
  dim = 4
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)
  riemann = get_riemann_curvature_tensor(connection)

  ricci_with_R = get_ricci_tensor(connection, R=riemann)
  ricci_without_R = get_ricci_tensor(connection)

  assert jnp.allclose(ricci_with_R.components.value, ricci_without_R.components.value)


def test_ricci_scalar_basis_independence():
  """
  Test that the Ricci scalar R = g^{ab} R_{ab} is independent of the basis.
  """
  key = random.PRNGKey(74)
  dim = 4
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)
  ricci = get_ricci_tensor(connection)

  g = metric.components.value
  g_inv = jnp.linalg.inv(g)
  Ric = ricci.components.value
  ricci_scalar = jnp.einsum("ab,ab->", g_inv, Ric)

  # Change to standard basis and recompute
  metric_std = change_basis(metric, get_standard_basis(metric.basis.p))
  connection_std = get_levi_civita_connection(metric_std)
  ricci_std = get_ricci_tensor(connection_std)

  g_std = metric_std.components.value
  g_inv_std = jnp.linalg.inv(g_std)
  Ric_std = ricci_std.components.value
  ricci_scalar_std = jnp.einsum("ab,ab->", g_inv_std, Ric_std)

  assert jnp.allclose(ricci_scalar, ricci_scalar_std)


def test_ricci_tensor_flat_metric():
  """
  Test that Ricci tensor vanishes for a flat metric.
  """
  dim = 4
  p = jnp.zeros(dim)
  metric = RiemannianMetric(basis=get_standard_basis(p), components=get_identity_jet(dim))
  connection = get_levi_civita_connection(metric)

  ricci = get_ricci_tensor(connection)

  assert jnp.allclose(ricci.components.value, 0.0)