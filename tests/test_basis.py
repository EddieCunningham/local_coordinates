import jax.numpy as jnp
from local_coordinates.basis import BasisVectors, DualBasis, get_basis_transform, make_coordinate_basis, get_standard_basis, get_standard_dual_basis, get_lie_bracket_components
from local_coordinates.jet import Jet, function_to_jet, jet_decorator
import pytest
import jax
import jax.random as random

def test_basis_vectors_creation():
  p = jnp.array([1., 2.])
  basis_vectors = jnp.eye(2)
  components_jet = Jet(value=basis_vectors, gradient=None, hessian=None)
  cs = BasisVectors(p=p, components=components_jet)
  assert jnp.array_equal(cs.p, p)
  assert jnp.array_equal(cs.components.value, basis_vectors)

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
  assert jnp.array_equal(cs.components.gradient, hessian)

def test_basis_vectors_batching():
  # Batch of 3 points
  p_batch = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
  # Batch of 3 corresponding identity basis vectors
  b_batch = jnp.stack([jnp.eye(2)] * 3)
  components_jet = Jet(value=b_batch, gradient=None, hessian=None)
  cs = BasisVectors(p=p_batch, components=components_jet)

  assert cs.batch_size == 3
  assert cs.p.shape == (3, 2)
  assert cs.components.value.shape == (3, 2, 2)

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
  assert jnp.allclose(coord_basis.components.gradient, new_basis.components.gradient)
  # The point and basis vectors should also be unchanged
  assert jnp.allclose(coord_basis.p, new_basis.p)
  assert jnp.allclose(coord_basis.components.value, new_basis.components.value)


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
  dframe_dx_new = new_basis.components.gradient
  frame_new = new_basis.components.value # This is unchanged

  # d(E_j)^i / dz^k = ∑_r (d(E_j)^i / dx^r) * (E_k)^r
  dframe_dz_new = jnp.einsum('ijr,rk->ijk', dframe_dx_new, frame_new)

  # Assert that d(E_j)/dz^k is symmetric in j and k
  assert jnp.allclose(dframe_dz_new, jnp.swapaxes(dframe_dz_new, 1, 2))

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

def test_basis_transform_via_inverse():
  # Test transforming from a basis to the standard basis
  p = jnp.array([0., 0.])
  key = random.key(0)
  k1, k2, k3 = random.split(key, 3)

  basis_vectors = random.normal(k1, (2, 2))
  gradient = random.normal(k2, (2, 2, 2))
  hessian = random.normal(k3, (2, 2, 2, 2))
  components_jet = Jet(value=basis_vectors, gradient=gradient, hessian=hessian)
  basis = BasisVectors(p=p, components=components_jet)

  # Create the standard basis
  standard_basis = BasisVectors(p=p, components=Jet(value=jnp.eye(2), gradient=jnp.zeros((2, 2, 2)), hessian=jnp.zeros((2, 2, 2, 2))))

  # Get the transformation to the standard basis
  transform = get_basis_transform(basis, standard_basis)
  inverse_transform = get_basis_transform(standard_basis, basis)

  @jet_decorator
  def blah(transform_vals, inv_transform_vals):
    return transform_vals @ inv_transform_vals

  eye = blah(transform.get_value_jet(), inverse_transform.get_value_jet())

  # In these coordinates,
  assert jnp.allclose(eye.value, jnp.eye(2))
  assert jnp.allclose(eye.gradient, 0.0)
  assert jnp.allclose(eye.hessian, 0.0)


def test_get_standard_basis():
  p = jnp.array([1.0, 2.0])
  basis = get_standard_basis(p)

  # Point is preserved
  assert jnp.allclose(basis.p, p)

  # Value is identity, derivatives are zero with correct shapes
  assert jnp.allclose(basis.components.value, jnp.eye(2))
  assert basis.components.gradient.shape == (2, 2, 2)
  assert basis.components.hessian.shape == (2, 2, 2, 2)
  assert jnp.allclose(basis.components.gradient, 0.0)
  assert jnp.allclose(basis.components.hessian, 0.0)


def test_dual_basis_standard():
  p = jnp.array([1.0, 2.0])
  dual = get_standard_dual_basis(p)
  assert jnp.allclose(dual.p, p)
  assert jnp.allclose(dual.components.value, jnp.eye(2))
  assert dual.components.gradient.shape == (2, 2, 2)
  assert dual.components.hessian.shape == (2, 2, 2, 2)
  assert jnp.allclose(dual.components.gradient, 0.0)
  assert jnp.allclose(dual.components.hessian, 0.0)


def test_dual_basis_transform_matches_vector_inverse():
  p = jnp.array([0., 0.])
  B_from = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  B_to = jnp.array([[0.0, 1.0], [1.0, 0.0]])

  vec_from = BasisVectors(p=p, components=Jet(value=B_from, gradient=None, hessian=None))
  vec_to = BasisVectors(p=p, components=Jet(value=B_to, gradient=None, hessian=None))

  # Dual components are inverses of vector basis matrices
  theta_from = DualBasis(p=p, components=Jet(value=jnp.linalg.inv(B_from), gradient=None, hessian=None))
  theta_to = DualBasis(p=p, components=Jet(value=jnp.linalg.inv(B_to), gradient=None, hessian=None))

  T_vec = get_basis_transform(vec_from, vec_to).value
  T_dual = get_basis_transform(theta_from, theta_to).value

  assert jnp.allclose(T_dual, jnp.linalg.inv(T_vec))


def test_dual_basis_transform_composition():
  p = jnp.array([0., 0.])
  B1 = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  B2 = jnp.array([[0.0, 1.0], [1.0, 0.0]])
  B3 = jnp.array([[2.0, 0.0], [0.0, 0.5]])

  theta1 = DualBasis(p=p, components=Jet(value=jnp.linalg.inv(B1), gradient=None, hessian=None))
  theta2 = DualBasis(p=p, components=Jet(value=jnp.linalg.inv(B2), gradient=None, hessian=None))
  theta3 = DualBasis(p=p, components=Jet(value=jnp.linalg.inv(B3), gradient=None, hessian=None))

  T12 = get_basis_transform(theta1, theta2).value
  T23 = get_basis_transform(theta2, theta3).value
  T13 = get_basis_transform(theta1, theta3).value

  # For dual transforms, composition order is T13 = T12 @ T23
  assert jnp.allclose(T13, T12 @ T23)


def test_dual_and_vector_transforms_pairing_identity():
  p = jnp.array([0., 0.])
  B_from = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  B_to = jnp.array([[0.0, 1.0], [1.0, 0.0]])

  vec_from = BasisVectors(p=p, components=Jet(value=B_from, gradient=None, hessian=None))
  vec_to = BasisVectors(p=p, components=Jet(value=B_to, gradient=None, hessian=None))

  theta_from = DualBasis(p=p, components=Jet(value=jnp.linalg.inv(B_from), gradient=None, hessian=None))
  theta_to = DualBasis(p=p, components=Jet(value=jnp.linalg.inv(B_to), gradient=None, hessian=None))

  T_vec = get_basis_transform(vec_from, vec_to).value
  T_dual = get_basis_transform(theta_from, theta_to).value

  # Start with identity coordinates in the "from" bases
  E_coords_from = jnp.eye(2)
  Theta_coords_from = jnp.eye(2)

  # Transform coordinates into the "to" bases
  E_coords_to = T_vec @ E_coords_from
  Theta_coords_to = Theta_coords_from @ T_dual

  # Pairing θ(E) should remain identity in the target basis coordinates
  assert jnp.allclose(Theta_coords_to @ E_coords_to, jnp.eye(2))


def test_basis_dual_roundtrip():
  p = jnp.array([0., 0.])
  B = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  vec = BasisVectors(p=p, components=Jet(value=B, gradient=None, hessian=None))
  dual = vec.to_dual()
  vec_back = dual.to_primal()
  assert jnp.allclose(dual.components.value @ vec.components.value, jnp.eye(2))
  assert jnp.allclose(vec_back.components.value, B)


def test_dual_primal_pairing_invariance_under_transform():
  p = jnp.array([0., 0.])
  B1 = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  B2 = jnp.array([[0.0, 1.0], [1.0, 0.0]])
  vec1 = BasisVectors(p=p, components=Jet(value=B1, gradient=None, hessian=None))
  vec2 = BasisVectors(p=p, components=Jet(value=B2, gradient=None, hessian=None))
  dual1 = vec1.to_dual()
  dual2 = vec2.to_dual()

  T_vec = get_basis_transform(vec1, vec2).value
  T_dual = get_basis_transform(dual1, dual2).value

  # Coordinates of identity pairing transform accordingly
  E = jnp.eye(2)
  Theta = jnp.eye(2)
  E2 = T_vec @ E
  Theta2 = Theta @ T_dual
  assert jnp.allclose(Theta2 @ E2, jnp.eye(2))


def test_lie_bracket():
  # Construct a coordinate basis

  def nonlin(x):
    return jnp.log1p(jnp.abs(x))*jnp.sign(x)

  key = random.key(0)
  k1, k2 = random.split(key)
  mat1 = random.normal(k1, (2, 2))
  mat2 = random.normal(k2, (2, 2))
  b1 = random.normal(k1, (2,))
  b2 = random.normal(k2, (2,))

  def inv_chart(x):
    h = nonlin(mat1@x + b1)
    return nonlin(mat2@h + b2)

  # The point in the coordinate space to evaluate derivatives
  x0 = jnp.array([0.5, 0.5])

  # Create Jet objects for each basis using function_to_jet
  inv_coord_vector_jet = function_to_jet(inv_chart, x0) # dz/dx

  @jet_decorator
  def invert_basis(coord_grads):
    return jnp.linalg.inv(coord_grads)

  coord_vector_jet = invert_basis(inv_coord_vector_jet.get_gradient_jet()) # dx/dz

  # Create BasisVectors objects
  p = jnp.array([0., 0.]) # this is arbitrary for this test
  basis = BasisVectors(p=p, components=coord_vector_jet)

  @jet_decorator
  def lie_bracket_components(basis_vals, basis_grads):
    term1 = jnp.einsum("ai,kja->kij", basis_vals, basis_grads)
    term2 = jnp.einsum("aj,kia->kij", basis_vals, basis_grads)
    return term1 - term2

  basis_vals = basis.components.get_value_jet()
  basis_grads = basis.components.get_gradient_jet()
  lie_bracket_components = lie_bracket_components(basis_vals, basis_grads)

  assert jnp.allclose(lie_bracket_components.value, 0.0)

def test_get_lie_bracket_components_random():

  # Create BasisVectors object
  p = jnp.array([0., 0.]) # arbitrary for this test
  key = random.key(0)
  vals = random.normal(key, (2, 2))
  grads = random.normal(key, (2, 2, 2))
  hessians = random.normal(key, (2, 2, 2, 2))

  basis = BasisVectors(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians))

  # Compute Lie bracket components via library function and check they vanish
  c_jet = get_lie_bracket_components(basis)


def test_get_lie_bracket_components():
  # Construct a coordinate basis and verify the Lie bracket components vanish

  def nonlin(x):
    return jnp.log1p(jnp.abs(x))*jnp.sign(x)

  key = random.key(0)
  k1, k2 = random.split(key)
  mat1 = random.normal(k1, (2, 2))
  mat2 = random.normal(k2, (2, 2))
  b1 = random.normal(k1, (2,))
  b2 = random.normal(k2, (2,))

  def inv_chart(x):
    h = nonlin(mat1@x + b1)
    return nonlin(mat2@h + b2)

  # The point in the coordinate space to evaluate derivatives
  x0 = jnp.array([0.5, 0.5])

  # Create Jet objects for each basis using function_to_jet
  inv_coord_vector_jet = function_to_jet(inv_chart, x0) # dz/dx

  @jet_decorator
  def invert_basis(coord_grads):
    return jnp.linalg.inv(coord_grads)

  coord_vector_jet = invert_basis(inv_coord_vector_jet.get_gradient_jet()) # dx/dz

  # Create BasisVectors object
  p = jnp.array([0., 0.]) # arbitrary for this test
  basis = BasisVectors(p=p, components=coord_vector_jet)

  # Compute Lie bracket components via library function and check they vanish
  c_jet = get_lie_bracket_components(basis)
  assert jnp.allclose(c_jet.value, 0.0)


def test_get_lie_bracket_components_constant_frame_zero():
  # For a constant frame E (no spatial variation), the Lie bracket vanishes
  p = jnp.array([0.0, 0.0])
  A = jnp.array([[2.0, -1.0],
                 [1.5,  3.0]])
  dA = jnp.zeros((2, 2, 2))
  basis = BasisVectors(p=p, components=Jet(value=A, gradient=dA, hessian=None))
  c_jet = get_lie_bracket_components(basis)
  assert jnp.allclose(c_jet.value, 0.0)


def test_get_lie_bracket_components_simple_noncommuting_frame():
  # E1 = (1, 0), E2 = (x, 1) -> [E1,E2] = (1,0) = E1
  p = jnp.array([0.3, -0.2])
  E = jnp.array([[1.0, p[0]], # entry (0, 1) is p[0]
                 [0.0, 1.0]])
  dE = jnp.zeros((2, 2, 2))
  dE = dE.at[0, 1, 0].set(1.0)  # ∂_x E2^x = 1

  basis = BasisVectors(p=p, components=Jet(value=E, gradient=dE, hessian=None))
  out = get_lie_bracket_components(basis)
  c = out.value  # shape (k, i, j)

  expected = jnp.zeros_like(c)
  # c^0_{01} = +1, c^0_{10} = -1 (where k=0 corresponds to the first basis vector E_0)
  expected = expected.at[0, 0, 1].set(1.0)
  expected = expected.at[0, 1, 0].set(-1.0)
  assert jnp.allclose(c, expected)