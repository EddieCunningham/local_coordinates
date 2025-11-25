import jax.numpy as jnp
import pytest
from local_coordinates.basis import BasisVectors, get_basis_transform, get_standard_basis
from local_coordinates.tensor import Tensor, TensorType, change_basis, change_coordinates
from local_coordinates.jet import Jet, function_to_jet
from local_coordinates.jacobian import Jacobian, function_to_jacobian, get_inverse
import jax
import jax.random as random

@pytest.fixture
def a_basis():
    p = jnp.array([0., 0.])
    basis_vectors = jnp.eye(2)
    gradient = jnp.zeros((2, 2, 2))
    hessian = jnp.zeros((2, 2, 2, 2))
    components_jet = Jet(value=basis_vectors, gradient=gradient, hessian=hessian, dim=2)
    return BasisVectors(p=p, components=components_jet)

@pytest.fixture
def another_basis():
    p = jnp.array([1., 1.])
    angle = jnp.pi / 4
    rot = jnp.array([
        [jnp.cos(angle), -jnp.sin(angle)],
        [jnp.sin(angle), jnp.cos(angle)]
    ])
    gradient = jnp.zeros((2, 2, 2))
    hessian = jnp.zeros((2, 2, 2, 2))
    components_jet = Jet(value=rot, gradient=gradient, hessian=hessian, dim=2)
    return BasisVectors(p=p, components=components_jet)

def test_tensortype_methods():
    tt = TensorType(k=2, l=3)
    assert tt.total_dims() == 5
    assert tt.k_names == ['k0', 'k1']
    assert tt.l_names == ['l0', 'l1', 'l2']
    assert tt.get_coordinate_indices() == 'k0 k1 l0 l1 l2'

def test_tensor_creation(a_basis):
    tt = TensorType(k=1, l=1)
    components = jnp.eye(2)
    key = jax.random.PRNGKey(0)
    gradient = jax.random.normal(key, (2, 2, 2))
    hessian = jax.random.normal(key, (2, 2, 2, 2))
    components_jet = Jet(value=components, gradient=gradient, hessian=hessian)
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=components_jet)
    assert tensor.tensor_type == tt
    assert tensor.basis == a_basis
    assert jnp.array_equal(tensor.components.value, components)
    assert jnp.array_equal(tensor.components.gradient, gradient)
    expected_hessian = 0.5 * (hessian + jnp.swapaxes(hessian, -1, -2))
    assert jnp.allclose(tensor.components.hessian, expected_hessian)

def test_tensor_batch_size(a_basis):
    # No batch
    tt = TensorType(k=1, l=1)
    components = jnp.eye(2)
    key = jax.random.PRNGKey(1)
    gradient = jax.random.normal(key, (2, 2, 2))
    hessian = jax.random.normal(key, (2, 2, 2, 2))
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=Jet(value=components, gradient=gradient, hessian=hessian))
    assert tensor.batch_size is None

    # 1D batch
    components_batch = jnp.stack([jnp.eye(2)] * 5)
    gradient_batch = jax.random.normal(key, (5, 2, 2, 2))
    hessian_batch = jax.random.normal(key, (5, 2, 2, 2, 2))
    tensor_batch = Tensor(tensor_type=tt, basis=a_basis, components=Jet(value=components_batch, gradient=gradient_batch, hessian=hessian_batch))
    assert tensor_batch.batch_size == 5

    # Multi-D batch
    components_multibatch = jnp.stack([components_batch] * 3)
    gradient_multibatch = jax.random.normal(key, (3, 5, 2, 2, 2))
    hessian_multibatch = jax.random.normal(key, (3, 5, 2, 2, 2, 2))
    tensor_multibatch = Tensor(tensor_type=tt, basis=a_basis, components=Jet(value=components_multibatch, gradient=gradient_multibatch, hessian=hessian_multibatch))
    assert tensor_multibatch.batch_size == (3, 5)


def test_change_coordinates_vector(a_basis, another_basis):
    # Vector is a contravariant tensor of rank 1
    tt = TensorType(k=0, l=1)
    vec = jnp.array([1., 0.])
    key = jax.random.PRNGKey(2)
    gradient = jax.random.normal(key, (2, 2))
    hessian = jax.random.normal(key, (2, 2, 2))
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=Jet(value=vec, gradient=gradient, hessian=hessian))

    new_tensor = change_basis(tensor, another_basis)

    # Manual transformation
    T = get_basis_transform(a_basis, another_basis)
    manual_new_vec = jnp.einsum('l,ml->m', vec, T.value)
    manual_new_grad = jnp.einsum('lr,ml->mr', gradient, T.value)
    manual_new_hess = jnp.einsum('lrs,ml->mrs', hessian, T.value)


    assert new_tensor.basis == another_basis
    assert jnp.allclose(new_tensor.components.value, manual_new_vec)
    assert jnp.allclose(new_tensor.components.gradient, manual_new_grad)
    manual_new_hess = 0.5 * (manual_new_hess + jnp.swapaxes(manual_new_hess, -1, -2))
    assert jnp.allclose(new_tensor.components.hessian, manual_new_hess)


def test_change_coordinates_covector(a_basis, another_basis):
    # Covector is a covariant tensor of rank 1
    tt = TensorType(k=1, l=0)
    covec = jnp.array([1., 0.])
    key = jax.random.PRNGKey(3)
    gradient = jax.random.normal(key, (2, 2))
    hessian = jax.random.normal(key, (2, 2, 2))
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=Jet(value=covec, gradient=gradient, hessian=hessian))

    new_tensor = change_basis(tensor, another_basis)

    # Manual transformation
    T = get_basis_transform(a_basis, another_basis)
    Tinv = jnp.linalg.inv(T.value)
    manual_new_covec = jnp.einsum('k,kj->j', covec, Tinv)
    manual_new_grad = jnp.einsum('kr,kj->jr', gradient, Tinv)
    manual_new_hess = jnp.einsum('krs,kj->jrs', hessian, Tinv)

    assert new_tensor.basis == another_basis
    assert jnp.allclose(new_tensor.components.value, manual_new_covec)
    assert jnp.allclose(new_tensor.components.gradient, manual_new_grad)
    manual_new_hess = 0.5 * (manual_new_hess + jnp.swapaxes(manual_new_hess, -1, -2))
    assert jnp.allclose(new_tensor.components.hessian, manual_new_hess)

def test_change_coordinates_metric_tensor(a_basis, another_basis):
    # Metric tensor is a covariant tensor of rank 2
    tt = TensorType(k=2, l=0)
    metric = jnp.eye(2)
    key = jax.random.PRNGKey(4)
    gradient = jax.random.normal(key, (2, 2, 2))
    hessian = jax.random.normal(key, (2, 2, 2, 2))
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=Jet(value=metric, gradient=gradient, hessian=hessian))

    new_tensor = change_basis(tensor, another_basis)

    # Manual transformation
    T = get_basis_transform(a_basis, another_basis)
    Tinv = jnp.linalg.inv(T.value)
    manual_new_metric = jnp.einsum('kl,ki,lj->ij', metric, Tinv, Tinv)
    manual_new_grad = jnp.einsum('klr,ki,lj->ijr', gradient, Tinv, Tinv)
    manual_new_hess = jnp.einsum('klrs,ki,lj->ijrs', hessian, Tinv, Tinv)

    assert new_tensor.basis == another_basis
    assert jnp.allclose(new_tensor.components.value, manual_new_metric)
    assert jnp.allclose(new_tensor.components.gradient, manual_new_grad)
    manual_new_hess = 0.5 * (manual_new_hess + jnp.swapaxes(manual_new_hess, -1, -2))
    assert jnp.allclose(new_tensor.components.hessian, manual_new_hess)

def test_change_coordinates_tensor_0_2(a_basis, another_basis):
    tt = TensorType(k=0, l=2)
    components = jnp.array([[1., 2.], [3., 4.]])
    key = jax.random.PRNGKey(5)
    gradient = jax.random.normal(key, (2, 2, 2))
    hessian = jax.random.normal(key, (2, 2, 2, 2))
    tensor = Tensor(tensor_type=tt, basis=a_basis, components=Jet(value=components, gradient=gradient, hessian=hessian))

    new_tensor = change_basis(tensor, another_basis)

    # Manual transformation
    T = get_basis_transform(a_basis, another_basis)
    Tinv = jnp.linalg.inv(T.value)
    manual_new_components = jnp.einsum('ij,ki,lj->kl', components, T.value, T.value)
    manual_new_grad = jnp.einsum('ijr,ki,lj->klr', gradient, T.value, T.value)
    manual_new_hess = jnp.einsum('ijrs,ki,lj->klrs', hessian, T.value, T.value)

    assert new_tensor.basis == another_basis
    assert jnp.allclose(new_tensor.components.value, manual_new_components)
    assert jnp.allclose(new_tensor.components.gradient, manual_new_grad)
    manual_new_hess = 0.5 * (manual_new_hess + jnp.swapaxes(manual_new_hess, -1, -2))
    assert jnp.allclose(new_tensor.components.hessian, manual_new_hess)


# ============================================================================
# Tests for change_coordinates
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


def test_change_coordinates_value_unchanged():
  """
  Test that change_coordinates preserves the tensor component values.
  Only the derivatives should change (via chain rule).
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])

  # Create a tensor in Cartesian coordinates
  basis = get_standard_basis(p_cart)
  tt = TensorType(k=1, l=1)
  key = jax.random.PRNGKey(42)
  k1, k2, k3 = random.split(key, 3)
  components = random.normal(k1, (dim, dim))
  gradient = random.normal(k2, (dim, dim, dim))
  hessian = random.normal(k3, (dim, dim, dim, dim))
  tensor = Tensor(
    tensor_type=tt,
    basis=basis,
    components=Jet(value=components, gradient=gradient, hessian=hessian)
  )

  # Change coordinates to polar
  jac = function_to_jacobian(cartesian_to_polar, p_cart)
  tensor_polar = change_coordinates(tensor, jac)

  # Value should be unchanged
  assert jnp.allclose(tensor_polar.components.value, components)


def test_change_coordinates_round_trip_scalar():
  """
  Test that changing coordinates and changing back gives the original tensor
  for a scalar (0,0)-tensor.
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])
  p_polar = cartesian_to_polar(p_cart)

  # Create a scalar tensor
  basis = get_standard_basis(p_cart)
  tt = TensorType(k=0, l=0)
  key = jax.random.PRNGKey(100)
  k1, k2, k3 = random.split(key, 3)
  # Scalar has shape ()
  components = random.normal(k1, ())
  gradient = random.normal(k2, (dim,))
  hessian = random.normal(k3, (dim, dim))
  tensor = Tensor(
    tensor_type=tt,
    basis=basis,
    components=Jet(value=components, gradient=gradient, hessian=hessian)
  )

  # Round trip: Cartesian -> Polar -> Cartesian
  jac_to_polar = function_to_jacobian(cartesian_to_polar, p_cart)
  tensor_polar = change_coordinates(tensor, jac_to_polar)

  jac_to_cart = function_to_jacobian(polar_to_cartesian, p_polar)
  tensor_back = change_coordinates(tensor_polar, jac_to_cart)

  # Should recover original
  assert jnp.allclose(tensor_back.components.value, tensor.components.value)
  assert jnp.allclose(tensor_back.components.gradient, tensor.components.gradient, atol=1e-5)


def test_change_coordinates_round_trip_vector():
  """
  Test round-trip for a contravariant vector (0,1)-tensor.
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])
  p_polar = cartesian_to_polar(p_cart)

  basis = get_standard_basis(p_cart)
  tt = TensorType(k=0, l=1)
  key = jax.random.PRNGKey(101)
  k1, k2, k3 = random.split(key, 3)
  components = random.normal(k1, (dim,))
  gradient = random.normal(k2, (dim, dim))
  hessian = random.normal(k3, (dim, dim, dim))
  tensor = Tensor(
    tensor_type=tt,
    basis=basis,
    components=Jet(value=components, gradient=gradient, hessian=hessian)
  )

  # Round trip
  jac_to_polar = function_to_jacobian(cartesian_to_polar, p_cart)
  tensor_polar = change_coordinates(tensor, jac_to_polar)

  jac_to_cart = function_to_jacobian(polar_to_cartesian, p_polar)
  tensor_back = change_coordinates(tensor_polar, jac_to_cart)

  assert jnp.allclose(tensor_back.components.value, tensor.components.value)
  assert jnp.allclose(tensor_back.components.gradient, tensor.components.gradient, atol=1e-5)


def test_change_coordinates_round_trip_covector():
  """
  Test round-trip for a covariant covector (1,0)-tensor.
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])
  p_polar = cartesian_to_polar(p_cart)

  basis = get_standard_basis(p_cart)
  tt = TensorType(k=1, l=0)
  key = jax.random.PRNGKey(102)
  k1, k2, k3 = random.split(key, 3)
  components = random.normal(k1, (dim,))
  gradient = random.normal(k2, (dim, dim))
  hessian = random.normal(k3, (dim, dim, dim))
  tensor = Tensor(
    tensor_type=tt,
    basis=basis,
    components=Jet(value=components, gradient=gradient, hessian=hessian)
  )

  # Round trip
  jac_to_polar = function_to_jacobian(cartesian_to_polar, p_cart)
  tensor_polar = change_coordinates(tensor, jac_to_polar)

  jac_to_cart = function_to_jacobian(polar_to_cartesian, p_polar)
  tensor_back = change_coordinates(tensor_polar, jac_to_cart)

  assert jnp.allclose(tensor_back.components.value, tensor.components.value)
  assert jnp.allclose(tensor_back.components.gradient, tensor.components.gradient, atol=1e-5)


def test_change_coordinates_round_trip_metric():
  """
  Test round-trip for a (2,0) covariant tensor (metric-like).
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])
  p_polar = cartesian_to_polar(p_cart)

  basis = get_standard_basis(p_cart)
  tt = TensorType(k=2, l=0)
  key = jax.random.PRNGKey(103)
  k1, k2, k3 = random.split(key, 3)
  components = random.normal(k1, (dim, dim))
  gradient = random.normal(k2, (dim, dim, dim))
  hessian = random.normal(k3, (dim, dim, dim, dim))
  tensor = Tensor(
    tensor_type=tt,
    basis=basis,
    components=Jet(value=components, gradient=gradient, hessian=hessian)
  )

  # Round trip
  jac_to_polar = function_to_jacobian(cartesian_to_polar, p_cart)
  tensor_polar = change_coordinates(tensor, jac_to_polar)

  jac_to_cart = function_to_jacobian(polar_to_cartesian, p_polar)
  tensor_back = change_coordinates(tensor_polar, jac_to_cart)

  assert jnp.allclose(tensor_back.components.value, tensor.components.value)
  assert jnp.allclose(tensor_back.components.gradient, tensor.components.gradient, atol=1e-5)


def test_change_coordinates_round_trip_mixed_tensor():
  """
  Test round-trip for a (1,1) mixed tensor.
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])
  p_polar = cartesian_to_polar(p_cart)

  basis = get_standard_basis(p_cart)
  tt = TensorType(k=1, l=1)
  key = jax.random.PRNGKey(104)
  k1, k2, k3 = random.split(key, 3)
  components = random.normal(k1, (dim, dim))
  gradient = random.normal(k2, (dim, dim, dim))
  hessian = random.normal(k3, (dim, dim, dim, dim))
  tensor = Tensor(
    tensor_type=tt,
    basis=basis,
    components=Jet(value=components, gradient=gradient, hessian=hessian)
  )

  # Round trip
  jac_to_polar = function_to_jacobian(cartesian_to_polar, p_cart)
  tensor_polar = change_coordinates(tensor, jac_to_polar)

  jac_to_cart = function_to_jacobian(polar_to_cartesian, p_polar)
  tensor_back = change_coordinates(tensor_polar, jac_to_cart)

  assert jnp.allclose(tensor_back.components.value, tensor.components.value)
  assert jnp.allclose(tensor_back.components.gradient, tensor.components.gradient, atol=1e-5)


def test_change_coordinates_basis_transforms_correctly():
  """
  Test that the basis is correctly transformed under change_coordinates.
  The basis should be expressed in the new coordinate system.
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])

  basis = get_standard_basis(p_cart)
  tt = TensorType(k=1, l=1)
  key = jax.random.PRNGKey(105)
  k1, k2, k3 = random.split(key, 3)
  components = random.normal(k1, (dim, dim))
  gradient = random.normal(k2, (dim, dim, dim))
  hessian = random.normal(k3, (dim, dim, dim, dim))
  tensor = Tensor(
    tensor_type=tt,
    basis=basis,
    components=Jet(value=components, gradient=gradient, hessian=hessian)
  )

  jac = function_to_jacobian(cartesian_to_polar, p_cart)
  tensor_polar = change_coordinates(tensor, jac)

  # The basis should be transformed by G (forward Jacobian)
  G = jac.value  # G[i,a] = dz^i/dx^a
  expected_basis_value = G @ basis.components.value
  assert jnp.allclose(tensor_polar.basis.components.value, expected_basis_value)


def test_change_coordinates_gradient_chain_rule():
  """
  Test that the gradient transforms according to the chain rule.
  For scalar jets: dF/dz^i = (dx^a/dz^i) * dF/dx^a
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])

  basis = get_standard_basis(p_cart)
  # Use a scalar tensor (0,0) for simplicity
  tt = TensorType(k=0, l=0)
  key = jax.random.PRNGKey(106)
  k1, k2, k3 = random.split(key, 3)
  components = random.normal(k1, ())
  gradient = random.normal(k2, (dim,))
  hessian = random.normal(k3, (dim, dim))
  tensor = Tensor(
    tensor_type=tt,
    basis=basis,
    components=Jet(value=components, gradient=gradient, hessian=hessian)
  )

  jac = function_to_jacobian(cartesian_to_polar, p_cart)
  tensor_polar = change_coordinates(tensor, jac)

  # Expected gradient via chain rule: dF/dz^i = J^a_i * dF/dx^a
  # where J^a_i = dx^a/dz^i (inverse Jacobian)
  J_inv = get_inverse(jac)
  J = J_inv.value  # J[a,i] = dx^a/dz^i
  expected_gradient = jnp.einsum("a,ai->i", gradient, J)

  assert jnp.allclose(tensor_polar.components.gradient, expected_gradient)


def test_change_coordinates_higher_rank_tensor():
  """
  Test change_coordinates for a higher-rank (2,2) tensor.
  """
  dim = 2
  p_cart = jnp.array([1.5, 0.8])
  p_polar = cartesian_to_polar(p_cart)

  basis = get_standard_basis(p_cart)
  tt = TensorType(k=2, l=2)
  key = jax.random.PRNGKey(107)
  k1, k2, k3 = random.split(key, 3)
  components = random.normal(k1, (dim, dim, dim, dim))
  gradient = random.normal(k2, (dim, dim, dim, dim, dim))
  hessian = random.normal(k3, (dim, dim, dim, dim, dim, dim))
  tensor = Tensor(
    tensor_type=tt,
    basis=basis,
    components=Jet(value=components, gradient=gradient, hessian=hessian)
  )

  # Round trip
  jac_to_polar = function_to_jacobian(cartesian_to_polar, p_cart)
  tensor_polar = change_coordinates(tensor, jac_to_polar)

  jac_to_cart = function_to_jacobian(polar_to_cartesian, p_polar)
  tensor_back = change_coordinates(tensor_polar, jac_to_cart)

  assert jnp.allclose(tensor_back.components.value, tensor.components.value)
  assert jnp.allclose(tensor_back.components.gradient, tensor.components.gradient, atol=1e-5)


def test_change_coordinates_3d():
  """
  Test change_coordinates in 3D (spherical to Cartesian).
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
  tt = TensorType(k=1, l=1)
  key = jax.random.PRNGKey(108)
  k1, k2, k3 = random.split(key, 3)
  components = random.normal(k1, (dim, dim))
  gradient = random.normal(k2, (dim, dim, dim))
  hessian = random.normal(k3, (dim, dim, dim, dim))
  tensor = Tensor(
    tensor_type=tt,
    basis=basis,
    components=Jet(value=components, gradient=gradient, hessian=hessian)
  )

  # Round trip
  jac_to_sph = function_to_jacobian(cartesian_to_spherical, p_cart)
  tensor_sph = change_coordinates(tensor, jac_to_sph)

  jac_to_cart = function_to_jacobian(spherical_to_cartesian, p_sph)
  tensor_back = change_coordinates(tensor_sph, jac_to_cart)

  assert jnp.allclose(tensor_back.components.value, tensor.components.value)
  assert jnp.allclose(tensor_back.components.gradient, tensor.components.gradient, atol=1e-5)
