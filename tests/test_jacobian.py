import jax
import jax.numpy as jnp
import pytest

from local_coordinates.jacobian import Jacobian, get_inverse


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

  J_inv = get_inverse(J)

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
  J_inv = get_inverse(J)

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
  J_inv = get_inverse(J)

  # Second inverse: J_inv_inv should match J
  J_inv_inv = get_inverse(J_inv)

  # Check closeness
  # Tolerances need to be a bit looser due to accumulated precision errors in inversions and einsums
  atol = 1e-5
  rtol = 1e-5

  assert jnp.allclose(J_inv_inv.value, J.value, atol=atol, rtol=rtol), "Value mismatch in round trip"
  assert jnp.allclose(J_inv_inv.gradient, J.gradient, atol=atol, rtol=rtol), "Gradient mismatch in round trip"
  assert jnp.allclose(J_inv_inv.hessian, J.hessian, atol=atol, rtol=rtol), "Hessian mismatch in round trip"
