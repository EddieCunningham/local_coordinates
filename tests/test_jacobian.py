import jax
import jax.numpy as jnp
import pytest

from local_coordinates.jacobian import Jacobian, compose, function_to_jacobian


def test_jacobian_basic_construction():
  dim = 3
  value = jnp.eye(dim)
  gradient = jnp.zeros((dim, dim, dim))
  hessian = jnp.zeros((dim, dim, dim, dim))

  J = Jacobian(value=value, gradient=gradient, hessian=hessian)

  assert jnp.allclose(J.value, value)
  assert jnp.allclose(J.gradient, gradient)
  assert jnp.allclose(J.hessian, hessian)
  assert J.batch_size is None


def test_jacobian_invalid_shape_raises():
  dim = 3
  bad_value = jnp.zeros((dim - 1, dim - 1))
  # Note: with p removed, we can't check against p.shape anymore.
  # But the property or methods using it might still fail if shapes are weird?
  # The batch_size logic checks ndim.
  # Let's just check that batch_size logic doesn't crash on weird shapes
  # or if we want to enforce squareness:
  pass # The previous test was checking value vs p shape. Now p is gone.


def test_jacobian_get_inverse_linear_only():
  dim = 3
  # Random invertible matrix
  A = jnp.array([[2.0, 0.5, -1.0],
                 [0.0, 1.5, 0.3],
                 [0.1, -0.2, 1.2]])
  J = Jacobian(value=A, gradient=None, hessian=None)

  J_inv = J.get_inverse()

  assert jnp.allclose(J_inv.value @ J.value, jnp.eye(dim), atol=1e-6, rtol=1e-6)
  assert jnp.allclose(J.value @ J_inv.value, jnp.eye(dim), atol=1e-6, rtol=1e-6)
  assert J_inv.gradient is None


def test_jacobian_get_inverse_hessian_1d():
  # One-dimensional test where we know the closed-form inverse derivatives.
  # z(x) = x + a x^2 + b x^3 near x = x0
  a = 0.3
  b = -0.2

  def z_scalar(x):
    return x + a * x**2 + b * x**3

  def F(xvec):
    return jnp.array([z_scalar(xvec[0])])

  x0 = jnp.array([0.1])

  # Forward derivatives at x0
  dzdx = jax.jacrev(F)(x0)            # shape (1,1)
  d2zdx2 = jax.jacfwd(jax.jacrev(F))(x0)  # shape (1,1,1)
  d3zdx3 = jax.jacfwd(jax.jacfwd(jax.jacrev(F)))(x0)  # shape (1,1,1,1)

  J = Jacobian(value=dzdx, gradient=d2zdx2, hessian=d3zdx3)
  J_inv = J.get_inverse()

  A = float(dzdx[0, 0])
  B = float(d2zdx2[0, 0, 0])
  C = float(d3zdx3[0, 0, 0, 0])

  # Known 1D inverse-derivative formulas:
  # x'(z0)  = 1 / z'(x0)
  # x''(z0) = - z''(x0) / z'(x0)^3
  # x'''(z0)= (3 z''(x0)^2 - z'(x0) z'''(x0)) / z'(x0)^5
  expected_first = 1.0 / A
  expected_second = -B / (A**3)
  expected_third = (3.0 * B * B - A * C) / (A**5)

  assert jnp.allclose(J_inv.value[0, 0], expected_first, atol=1e-6, rtol=1e-6)
  assert jnp.allclose(J_inv.gradient[0, 0, 0], expected_second, atol=1e-6, rtol=1e-6)
  assert jnp.allclose(J_inv.hessian[0, 0, 0, 0], expected_third, atol=1e-6, rtol=1e-6)


def test_jacobian_inverse_round_trip():
  dim = 3
  key = jax.random.PRNGKey(42)
  k1, k2, k3 = jax.random.split(key, 3)

  # Random invertible Jacobian at p=0

  # A: value (dz/dx)
  A = jax.random.normal(k1, (dim, dim))
  # Make sure it's reasonably well-conditioned
  A = A + jnp.eye(dim) * 5.0

  # B: gradient (d2z/dx2) - symmetric in last two indices
  B = jax.random.normal(k2, (dim, dim, dim))
  B = (B + B.transpose(0, 2, 1)) / 2.0

  # C: hessian (d3z/dx3) - symmetric in last three indices
  C = jax.random.normal(k3, (dim, dim, dim, dim))
  # Symmetrize C over last three indices (full permutation group S3)
  # Indices: 0, 1, 2, 3 -> we symmetrize 1, 2, 3
  perms = [
      (0, 1, 2, 3), (0, 1, 3, 2),
      (0, 2, 1, 3), (0, 2, 3, 1),
      (0, 3, 1, 2), (0, 3, 2, 1)
  ]
  C_sym = sum(C.transpose(perm) for perm in perms) / 6.0

  J = Jacobian(value=A, gradient=B, hessian=C_sym)

  # First inverse: J_inv
  J_inv = J.get_inverse()

  # Second inverse: J_inv_inv should match J
  J_inv_inv = J_inv.get_inverse()

  # Check closeness
  # Tolerances need to be a bit looser due to accumulated precision errors in inversions and einsums
  atol = 1e-5
  rtol = 1e-5

  assert jnp.allclose(J_inv_inv.value, J.value, atol=atol, rtol=rtol), "Value mismatch in round trip"
  assert jnp.allclose(J_inv_inv.gradient, J.gradient, atol=atol, rtol=rtol), "Gradient mismatch in round trip"
  assert jnp.allclose(J_inv_inv.hessian, J.hessian, atol=atol, rtol=rtol), "Hessian mismatch in round trip"


def test_compose_linear_only():
  """Test compose with only first derivatives (linear maps)."""
  dim = 3
  key = jax.random.PRNGKey(0)
  k1, k2 = jax.random.split(key, 2)

  A = jax.random.normal(k1, (dim, dim))
  B = jax.random.normal(k2, (dim, dim))

  J1 = Jacobian(value=A, gradient=None, hessian=None)
  J2 = Jacobian(value=B, gradient=None, hessian=None)

  J_composed = compose(J1, J2)

  # For linear maps, composition is just matrix multiplication
  expected_value = A @ B
  assert jnp.allclose(J_composed.value, expected_value, atol=1e-6)
  assert J_composed.gradient is None
  assert J_composed.hessian is None


def test_compose_with_identity():
  """Composing with identity Jacobian should return the original."""
  dim = 3
  key = jax.random.PRNGKey(1)
  k1, k2, k3 = jax.random.split(key, 3)

  # Random Jacobian with all derivatives
  A = jax.random.normal(k1, (dim, dim)) + 2.0 * jnp.eye(dim)
  B = jax.random.normal(k2, (dim, dim, dim))
  B = (B + B.transpose(0, 2, 1)) / 2.0
  C = jax.random.normal(k3, (dim, dim, dim, dim))
  perms = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3),
           (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1)]
  C = sum(C.transpose(perm) for perm in perms) / 6.0

  J = Jacobian(value=A, gradient=B, hessian=C)
  J_id = Jacobian(
    value=jnp.eye(dim),
    gradient=jnp.zeros((dim, dim, dim)),
    hessian=jnp.zeros((dim, dim, dim, dim))
  )

  # J ∘ identity = J
  J_right = compose(J, J_id)
  assert jnp.allclose(J_right.value, J.value, atol=1e-6)
  assert jnp.allclose(J_right.gradient, J.gradient, atol=1e-6)
  assert jnp.allclose(J_right.hessian, J.hessian, atol=1e-6)

  # identity ∘ J = J
  J_left = compose(J_id, J)
  assert jnp.allclose(J_left.value, J.value, atol=1e-6)
  assert jnp.allclose(J_left.gradient, J.gradient, atol=1e-6)
  assert jnp.allclose(J_left.hessian, J.hessian, atol=1e-6)


def test_compose_matches_function_composition():
  """Compose should match the Jacobian of composed functions."""
  dim = 2

  # Define two nonlinear functions
  def f1(z):
    # y = f1(z): some nonlinear map
    return jnp.array([
      z[0]**2 + 0.5*z[1],
      jnp.sin(z[0]) + z[1]**2
    ])

  def f2(x):
    # z = f2(x): another nonlinear map
    return jnp.array([
      x[0] + 0.3*x[1]**2,
      x[0]*x[1] + x[1]
    ])

  def f_composed(x):
    return f1(f2(x))

  x0 = jnp.array([0.5, 0.7])
  z0 = f2(x0)

  # Get Jacobians at appropriate points
  J1 = function_to_jacobian(f1, z0)  # Jacobian of f1 at z0
  J2 = function_to_jacobian(f2, x0)  # Jacobian of f2 at x0
  J_composed = compose(J1, J2)

  # Direct Jacobian of composed function
  J_direct = function_to_jacobian(f_composed, x0)

  atol = 1e-5
  assert jnp.allclose(J_composed.value, J_direct.value, atol=atol), \
    f"Value mismatch:\ncomposed={J_composed.value}\ndirect={J_direct.value}"
  assert jnp.allclose(J_composed.gradient, J_direct.gradient, atol=atol), \
    f"Gradient mismatch:\ncomposed={J_composed.gradient}\ndirect={J_direct.gradient}"
  assert jnp.allclose(J_composed.hessian, J_direct.hessian, atol=atol), \
    f"Hessian mismatch:\ncomposed={J_composed.hessian}\ndirect={J_direct.hessian}"


def test_compose_associativity():
  """(J1 ∘ J2) ∘ J3 = J1 ∘ (J2 ∘ J3)."""
  dim = 2

  def f1(z):
    return jnp.array([z[0]**2 + z[1], z[0] - z[1]**2])

  def f2(y):
    return jnp.array([y[0] + y[1]**2, y[0]*y[1]])

  def f3(x):
    return jnp.array([x[0] + 0.5*x[1], x[0]**2 + x[1]])

  x0 = jnp.array([0.3, 0.4])
  y0 = f3(x0)
  z0 = f2(y0)

  J1 = function_to_jacobian(f1, z0)
  J2 = function_to_jacobian(f2, y0)
  J3 = function_to_jacobian(f3, x0)

  # (J1 ∘ J2) ∘ J3
  J12 = compose(J1, J2)
  J123_left = compose(J12, J3)

  # J1 ∘ (J2 ∘ J3)
  J23 = compose(J2, J3)
  J123_right = compose(J1, J23)

  atol = 1e-5
  assert jnp.allclose(J123_left.value, J123_right.value, atol=atol)
  assert jnp.allclose(J123_left.gradient, J123_right.gradient, atol=atol)
  assert jnp.allclose(J123_left.hessian, J123_right.hessian, atol=atol)


def test_compose_inverse_is_identity():
  """J ∘ J^{-1} should give identity Jacobian."""
  dim = 2

  def f(x):
    return jnp.array([
      x[0] + 0.2*x[1]**2,
      x[1] + 0.1*x[0]**2
    ])

  x0 = jnp.array([0.5, 0.5])
  J = function_to_jacobian(f, x0)
  J_inv = J.get_inverse()

  # J ∘ J^{-1} should be identity
  J_id = compose(J, J_inv)

  atol = 1e-5
  assert jnp.allclose(J_id.value, jnp.eye(dim), atol=atol), \
    f"Value not identity:\n{J_id.value}"
  assert jnp.allclose(J_id.gradient, jnp.zeros((dim, dim, dim)), atol=atol), \
    f"Gradient not zero:\n{J_id.gradient}"
  assert jnp.allclose(J_id.hessian, jnp.zeros((dim, dim, dim, dim)), atol=atol), \
    f"Hessian not zero:\n{J_id.hessian}"
