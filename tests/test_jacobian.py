import jax.numpy as jnp
import pytest

from local_coordinates.jacobian import Jacobian


def test_jacobian_basic_construction():
  dim = 3
  p = jnp.zeros(dim)
  value = jnp.eye(dim)
  gradient = jnp.zeros((dim, dim, dim))
  hessian = jnp.zeros((dim, dim, dim, dim))

  J = Jacobian(p=p, value=value, gradient=gradient, hessian=hessian)

  assert jnp.allclose(J.p, p)
  assert jnp.allclose(J.value, value)
  assert jnp.allclose(J.gradient, gradient)
  assert jnp.allclose(J.hessian, hessian)
  assert J.batch_size is None


def test_jacobian_invalid_shape_raises():
  dim = 3
  p = jnp.zeros(dim)
  bad_value = jnp.zeros((dim - 1, dim - 1))

  with pytest.raises(ValueError):
    Jacobian(p=p, value=bad_value, gradient=None, hessian=None)


