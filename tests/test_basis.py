import jax.numpy as jnp
from local_coordinates.basis import BasisVectors, get_basis_transform, make_coordinate_basis
from local_coordinates.jet import Jet, function_to_jet
import pytest
import jax

def test_basis_vectors_creation():
  p = jnp.array([1., 2.])
  basis_vectors = jnp.eye(2)
  components_jet = Jet(value=basis_vectors, gradient=None, hessian=None)
  cs = BasisVectors(p=p, components=components_jet)
  assert jnp.array_equal(cs.p, p)
  assert jnp.array_equal(cs.basis_vectors, basis_vectors)

def test_get_coordinate_transform_simple():
  # cs1 is standard basis
  p1 = jnp.array([0., 0.])
  b1 = jnp.eye(2)
  components1 = Jet(value=b1, gradient=None, hessian=None)
  cs1 = BasisVectors(p=p1, components=components1)

  # cs2 has swapped basis vectors
  p2 = jnp.array([0., 0.])
  b2 = jnp.array([
    [0., 1.],
    [1., 0.]
  ])
  components2 = Jet(value=b2, gradient=None, hessian=None)
  cs2 = BasisVectors(p=p2, components=components2)

  # Transform from cs1 to cs2
  transform = get_basis_transform(cs1, cs2)

  # Expected transform
  # v2 = inv(b2) @ b1 @ v1
  # T = inv(b2)
  expected_transform = jnp.linalg.inv(b2)

  assert jnp.allclose(transform.value, expected_transform)

def test_get_coordinate_transform_rotated():
  # cs1 is standard basis
  p1 = jnp.array([0., 0.])
  b1 = jnp.eye(2)
  components1 = Jet(value=b1, gradient=None, hessian=None)
  cs1 = BasisVectors(p=p1, components=components1)

  # cs2 is rotated by 45 degrees
  p2 = jnp.array([0., 0.])
  angle = jnp.pi / 4
  b2 = jnp.array([
    [jnp.cos(angle), -jnp.sin(angle)],
    [jnp.sin(angle), jnp.cos(angle)]
  ])
  components2 = Jet(value=b2, gradient=None, hessian=None)
  cs2 = BasisVectors(p=p2, components=components2)

  # Transform from cs1 to cs2
  transform = get_basis_transform(cs1, cs2)

  # Expected transform
  # v2 = inv(b2) @ b1 @ v1
  # T = inv(b2) @ b1
  expected_transform = jnp.linalg.inv(b2) @ b1

  assert jnp.allclose(transform.value, expected_transform)

def test_basis_vectors_second_derivatives():
  p = jnp.array([1., 2.])
  basis_vectors = jnp.eye(2)
  hessian = jnp.ones((2, 2, 2))  # Example second derivatives
  components_jet = Jet(value=basis_vectors, gradient=hessian, hessian=None)
  cs = BasisVectors(p=p, components=components_jet)
  assert jnp.array_equal(cs.second_derivatives, hessian)

def test_basis_vectors_batching():
  # Batch of 3 points
  p_batch = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
  # Batch of 3 corresponding identity basis vectors
  b_batch = jnp.stack([jnp.eye(2)] * 3)
  components_jet = Jet(value=b_batch, gradient=None, hessian=None)
  cs = BasisVectors(p=p_batch, components=components_jet)

  assert cs.batch_size == 3
  assert cs.p.shape == (3, 2)
  assert cs.basis_vectors.shape == (3, 2, 2)

def test_get_coordinate_transform_skewed():
  # cs1 is standard basis
  p1 = jnp.array([0., 0.])
  b1 = jnp.eye(2)
  components1 = Jet(value=b1, gradient=None, hessian=None)
  cs1 = BasisVectors(p=p1, components=components1)

  # cs2 has skewed (non-orthogonal) basis vectors
  p2 = jnp.array([0., 0.])
  b2 = jnp.array([
    [1.0, 0.5],  # Skewed basis
    [0.0, 1.0]
  ])
  components2 = Jet(value=b2, gradient=None, hessian=None)
  cs2 = BasisVectors(p=p2, components=components2)

  # Transform from cs1 to cs2
  transform = get_basis_transform(cs1, cs2)

  # The transform should still be the inverse of the target basis matrix
  # since the source basis is identity.
  expected_transform = jnp.linalg.inv(b2)

  assert jnp.allclose(transform.value, expected_transform)

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
  components_jet = Jet(value=basis_vectors, gradient=second_derivatives, hessian=None)
  coord_basis = BasisVectors(p=p, components=components_jet)

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
  components_jet = Jet(value=frame, gradient=dframe_dx, hessian=None)
  non_coord_basis = BasisVectors(p=p, components=components_jet)

  # Apply the function
  new_basis = make_coordinate_basis(non_coord_basis)

  # Check that the new derivatives are symmetric in the frame's own basis
  dframe_dx_new = new_basis.second_derivatives
  frame_new = new_basis.basis_vectors # This is unchanged

  # d(E_j)^i / dz^k = ∑_r (d(E_j)^i / dx^r) * (E_k)^r
  dframe_dz_new = jnp.einsum('ijr,rk->ijk', dframe_dx_new, frame_new)

  # Assert that d(E_j)/dz^k is symmetric in j and k
  assert jnp.allclose(dframe_dz_new, jnp.swapaxes(dframe_dz_new, 1, 2))

import jax
import jax.numpy as jnp
import pytest
from local_coordinates.jet import Jet
from local_coordinates.basis import BasisVectors, get_mixing_function

@pytest.fixture
def sample_basis():
    """Provides a sample 2D BasisVectors object for testing."""
    p = jnp.array([1.0, 2.0])
    # Let D=2 (input dim) and N=2 (output dim)
    components_jet = Jet(
        value=jnp.array([[1., 0.5], [0.2, 1.]]),  # Basis vectors (Jacobian)
        # The effective hessian will be symmetrized by the Taylor expansion's quadratic form.
        # We provide a symmetric one here to make the test pass.
        gradient=jnp.array([[[0.1, 0.25], [0.25, 0.4]], [[0.5, 0.65], [0.65, 0.8]]]), # d(Basis)/dx (Hessian)
        hessian=None # Not needed for mixing function up to 2nd order
    )
    return BasisVectors(p=p, components=components_jet)

def test_mixing_function_at_origin(sample_basis):
    """Tests if the mixing function returns the base point p at dx=0."""
    mixing_fn = get_mixing_function(sample_basis)
    output_p = mixing_fn(jnp.zeros(2))
    assert jnp.allclose(output_p, sample_basis.p)

def test_mixing_function_first_derivative(sample_basis):
    """Tests if the Jacobian of the mixing function at dx=0 is the basis vectors."""
    mixing_fn = get_mixing_function(sample_basis)
    jacobian = jax.jacfwd(mixing_fn)(jnp.zeros(2))
    assert jnp.allclose(jacobian, sample_basis.components.value)

def test_mixing_function_second_derivative(sample_basis):
    """Tests if the Hessian of the mixing function at dx=0 is correct."""
    mixing_fn = get_mixing_function(sample_basis)
    # The Hessian of a vector-valued function is a tensor of shape (out, in, in)
    # jax.jacfwd(jax.jacrev(f)) computes this.
    hessian = jax.jacfwd(jax.jacrev(mixing_fn))(jnp.zeros(2))
    assert jnp.allclose(hessian, sample_basis.components.gradient)

def test_mixing_function_jit_compatibility(sample_basis):
    """Tests if the mixing function can be JIT-compiled."""
    mixing_fn = get_mixing_function(sample_basis)

    dx = jnp.array([0.1, -0.1])

    # Ensure both original and JIT'd versions give the same output
    expected_output = mixing_fn(dx)
    jit_output = jax.jit(mixing_fn)(dx)

    assert jnp.allclose(jit_output, expected_output)

def test_mixing_function_vmap_compatibility(sample_basis):
    """Tests if the mixing function works with vmap."""
    mixing_fn = get_mixing_function(sample_basis)

    batch_size = 5
    dx_batch = jnp.ones((batch_size, 2))

    # vmap the function over a batch of displacement vectors
    vmapped_fn = jax.vmap(mixing_fn)
    batch_output = vmapped_fn(dx_batch)

    # Manually compute the expected output for comparison
    manual_output = jnp.stack([mixing_fn(dx) for dx in dx_batch])

    assert batch_output.shape == (batch_size, 2)
    assert jnp.allclose(batch_output, manual_output)


def test_get_basis_transform_with_derivatives():
  """
  Tests that get_basis_transform correctly propagates derivatives
  when both basis vectors have value, gradient, and hessian.
  """
  # Define two basis vector fields as functions of a single coordinate 'x'
  def from_basis_func(x):
    # A simple linear function of x
    return jnp.eye(2) + x[0] * jnp.array([[1., 2.], [3., 4.]])

  def to_basis_func(x):
    # A rotation matrix dependent on x
    angle = jnp.pi / 4 * x[0]
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([[c, -s], [s, c]])

  # The point in the coordinate space to evaluate derivatives
  x0 = jnp.array([0.5])

  # Create Jet objects for each basis using function_to_jet
  from_components_jet = function_to_jet(from_basis_func, x0)
  to_components_jet = function_to_jet(to_basis_func, x0)

  # Create BasisVectors objects
  p = jnp.array([0., 0.]) # this is arbitrary for this test
  from_basis = BasisVectors(p=p, components=from_components_jet)
  to_basis = BasisVectors(p=p, components=to_components_jet)

  # Get the transformation Jet
  transform_jet = get_basis_transform(from_basis, to_basis)

  # For verification, define the transformation function directly
  def transform_func(x):
    return jnp.linalg.inv(to_basis_func(x)) @ from_basis_func(x)

  # Get the expected Jet for the transformation function
  expected_transform_jet = function_to_jet(transform_func, x0)

  # Compare value, gradient, and hessian
  assert jnp.allclose(transform_jet.value, expected_transform_jet.value)
  assert jnp.allclose(transform_jet.gradient, expected_transform_jet.gradient)
  assert jnp.allclose(transform_jet.hessian, expected_transform_jet.hessian)
