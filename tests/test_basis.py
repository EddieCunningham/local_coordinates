import jax.numpy as jnp
from local_coordinates.basis import BasisVectors, get_basis_transform, make_coordinate_basis
from local_coordinates.jet import Jet
import pytest
import jax

def test_basis_vectors_creation():
  p = jnp.array([1., 2.])
  basis_vectors = jnp.eye(2)
  jet = Jet(value=p, gradient=basis_vectors, hessian=None)
  cs = BasisVectors(_jet=jet)
  assert jnp.array_equal(cs.p, p)
  assert jnp.array_equal(cs.basis_vectors, basis_vectors)

def test_get_coordinate_transform_simple():
  # cs1 is standard basis
  p1 = jnp.array([0., 0.])
  b1 = jnp.eye(2)
  jet1 = Jet(value=p1, gradient=b1, hessian=None)
  cs1 = BasisVectors(_jet=jet1)

  # cs2 has swapped basis vectors
  p2 = jnp.array([0., 0.])
  b2 = jnp.array([
    [0., 1.],
    [1., 0.]
  ])
  jet2 = Jet(value=p2, gradient=b2, hessian=None)
  cs2 = BasisVectors(_jet=jet2)

  # Transform from cs1 to cs2
  transform = get_basis_transform(cs1, cs2)

  # Expected transform
  # v2 = inv(b2) @ b1 @ v1
  # T = inv(b2)
  expected_transform = jnp.linalg.inv(b2)

  assert jnp.allclose(transform, expected_transform)

def test_get_coordinate_transform_rotated():
  # cs1 is standard basis
  p1 = jnp.array([0., 0.])
  b1 = jnp.eye(2)
  jet1 = Jet(value=p1, gradient=b1, hessian=None)
  cs1 = BasisVectors(_jet=jet1)

  # cs2 is rotated by 45 degrees
  p2 = jnp.array([0., 0.])
  angle = jnp.pi / 4
  b2 = jnp.array([
    [jnp.cos(angle), -jnp.sin(angle)],
    [jnp.sin(angle), jnp.cos(angle)]
  ])
  jet2 = Jet(value=p2, gradient=b2, hessian=None)
  cs2 = BasisVectors(_jet=jet2)

  # Transform from cs1 to cs2
  transform = get_basis_transform(cs1, cs2)

  # Expected transform
  # v2 = inv(b2) @ b1 @ v1
  # T = inv(b2) @ b1
  expected_transform = jnp.linalg.inv(b2) @ b1

  assert jnp.allclose(transform, expected_transform)

def test_basis_vectors_second_derivatives():
  p = jnp.array([1., 2.])
  basis_vectors = jnp.eye(2)
  hessian = jnp.ones((2, 2, 2))  # Example second derivatives
  jet = Jet(value=p, gradient=basis_vectors, hessian=hessian)
  cs = BasisVectors(_jet=jet)
  assert jnp.array_equal(cs.second_derivatives, hessian)

def test_basis_vectors_batching():
  # Batch of 3 points
  p_batch = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
  # Batch of 3 corresponding identity basis vectors
  b_batch = jnp.stack([jnp.eye(2)] * 3)
  jet = Jet(value=p_batch, gradient=b_batch, hessian=None)
  cs = BasisVectors(_jet=jet)

  assert cs.batch_size == 3
  assert cs.p.shape == (3, 2)
  assert cs.basis_vectors.shape == (3, 2, 2)

def test_get_coordinate_transform_skewed():
  # cs1 is standard basis
  p1 = jnp.array([0., 0.])
  b1 = jnp.eye(2)
  jet1 = Jet(value=p1, gradient=b1, hessian=None)
  cs1 = BasisVectors(_jet=jet1)

  # cs2 has skewed (non-orthogonal) basis vectors
  p2 = jnp.array([0., 0.])
  b2 = jnp.array([
    [1.0, 0.5],  # Skewed basis
    [0.0, 1.0]
  ])
  jet2 = Jet(value=p2, gradient=b2, hessian=None)
  cs2 = BasisVectors(_jet=jet2)

  # Transform from cs1 to cs2
  transform = get_basis_transform(cs1, cs2)

  # The transform should still be the inverse of the target basis matrix
  # since the source basis is identity.
  expected_transform = jnp.linalg.inv(b2)

  assert jnp.allclose(transform, expected_transform)

def test_make_coordinate_basis_is_idempotent_on_coordinate_basis():
  """
  Tests that make_coordinate_basis does not change a basis that is already
  a coordinate basis. It uses a polar coordinate chart as an example.
  """
  # Define polar coordinates chart: (r, θ) -> (x, y)
  def chart(u):
    r, theta = u
    return jnp.array([r * jnp.cos(theta), r * jnp.sin(theta)])

  # Point in parameter space
  u = jnp.array([2.0, jnp.pi / 4])

  # Use JAX to get the point, basis vectors (Jacobian), and their derivatives (Hessian)
  p = chart(u)
  basis_vectors = jax.jacfwd(chart)(u)
  second_derivatives = jax.jacfwd(jax.jacrev(chart))(u)

  # Create the BasisVectors object. By construction, this is a coordinate basis.
  jet = Jet(value=p, gradient=basis_vectors, hessian=second_derivatives)
  coord_basis = BasisVectors(_jet=jet)

  # Apply the function
  new_basis = make_coordinate_basis(coord_basis)

  # The second derivatives should be unchanged (up to float precision)
  assert jnp.allclose(coord_basis.second_derivatives, new_basis.second_derivatives)
  # The point and basis vectors should also be unchanged
  assert jnp.allclose(coord_basis.p, new_basis.p)
  assert jnp.allclose(coord_basis.basis_vectors, new_basis.basis_vectors)


def test_make_coordinate_basis_symmetrizes():
  """
  Tests that the function correctly symmetrizes the derivatives of a
  non-commuting frame.
  """
  p = jnp.array([0., 0.])
  frame = jnp.eye(2)

  # Create a non-symmetric derivative tensor
  # d(E_j)^i / dx^r
  dframe_dx = jnp.zeros((2, 2, 2))
  dframe_dx = dframe_dx.at[0, 1, 0].set(1.0)  # ∂(E_1)^0/∂x^0 = 1

  # This corresponds to ∂E_1/∂z^0 != ∂E_0/∂z^1, so it's not a coordinate basis
  jet = Jet(value=p, gradient=frame, hessian=dframe_dx)
  non_coord_basis = BasisVectors(_jet=jet)

  # Apply the function
  new_basis = make_coordinate_basis(non_coord_basis)

  # Check that the new derivatives are symmetric in the frame's own basis
  dframe_dx_new = new_basis.second_derivatives
  frame_new = new_basis.basis_vectors # This is unchanged

  # d(E_j)^i / dz^k = ∑_r (d(E_j)^i / dx^r) * (E_k)^r
  dframe_dz_new = jnp.einsum('ijr,rk->ijk', dframe_dx_new, frame_new)

  # Assert that d(E_j)/dz^k is symmetric in j and k
  assert jnp.allclose(dframe_dz_new, jnp.swapaxes(dframe_dz_new, 1, 2))
