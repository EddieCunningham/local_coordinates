import jax.numpy as jnp
from local_coordinates.basis import BasisVectors, change_basis, get_basis_transform, get_dual_basis_transform, make_coordinate_basis, get_standard_basis, get_standard_dual_basis, change_coordinates, apply_covariant_transform, apply_contravariant_transform
from local_coordinates.jet import Jet, function_to_jet, jet_decorator, get_identity_jet
import pytest
import jax
import jax.random as random
from jaxtyping import Array

def test_basis_vectors_creation():
  p = jnp.array([1., 2.])
  basis_vectors = jnp.eye(2)
  components_jet = Jet(value=basis_vectors, gradient=None, hessian=None, dim=2)
  cs = BasisVectors(p=p, components=components_jet)
  assert jnp.array_equal(cs.p, p)
  assert jnp.array_equal(cs.components.value, basis_vectors)

def test_get_coordinate_transform_simple():
  # cs1 is standard basis
  p1 = jnp.array([0., 0.])
  b1 = jnp.eye(2)
  components1 = Jet(value=b1, gradient=None, hessian=None, dim=2)
  cs1 = BasisVectors(p=p1, components=components1)

  # cs2 has swapped basis vectors
  p2 = jnp.array([0., 0.])
  b2 = jnp.array([
    [0., 1.],
    [1., 0.]
  ])
  components2 = Jet(value=b2, gradient=None, hessian=None, dim=2)
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
  components1 = Jet(value=b1, gradient=None, hessian=None, dim=2)
  cs1 = BasisVectors(p=p1, components=components1)

  # cs2 is rotated by 45 degrees
  p2 = jnp.array([0., 0.])
  angle = jnp.pi / 4
  b2 = jnp.array([
    [jnp.cos(angle), -jnp.sin(angle)],
    [jnp.sin(angle), jnp.cos(angle)]
  ])
  components2 = Jet(value=b2, gradient=None, hessian=None, dim=2)
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
  components_jet = Jet(value=b_batch, gradient=None, hessian=None, dim=2)
  cs = BasisVectors(p=p_batch, components=components_jet)

  assert cs.batch_size == 3
  assert cs.p.shape == (3, 2)
  assert cs.components.value.shape == (3, 2, 2)

def test_get_coordinate_transform_skewed():
  # cs1 is standard basis
  p1 = jnp.array([0., 0.])
  b1 = jnp.eye(2)
  components1 = Jet(value=b1, gradient=None, hessian=None, dim=2)
  cs1 = BasisVectors(p=p1, components=components1)

  # cs2 has skewed (non-orthogonal) basis vectors
  p2 = jnp.array([0., 0.])
  b2 = jnp.array([
    [1.0, 0.5],  # Skewed basis
    [0.0, 1.0]
  ])
  components2 = Jet(value=b2, gradient=None, hessian=None, dim=2)
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

  vec_from = BasisVectors(p=p, components=Jet(value=B_from, gradient=None, hessian=None, dim=2))
  vec_to = BasisVectors(p=p, components=Jet(value=B_to, gradient=None, hessian=None, dim=2))

  # Dual components are inverses of vector basis matrices
  theta_from = BasisVectors(p=p, components=Jet(value=jnp.linalg.inv(B_from), gradient=None, hessian=None, dim=2))
  theta_to = BasisVectors(p=p, components=Jet(value=jnp.linalg.inv(B_to), gradient=None, hessian=None, dim=2))

  T_vec = get_basis_transform(vec_from, vec_to).value
  T_dual = get_dual_basis_transform(theta_from, theta_to).value

  assert jnp.allclose(T_dual, jnp.linalg.inv(T_vec))


def test_dual_basis_transform_composition():
  p = jnp.array([0., 0.])
  B1 = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  B2 = jnp.array([[0.0, 1.0], [1.0, 0.0]])
  B3 = jnp.array([[2.0, 0.0], [0.0, 0.5]])

  theta1 = BasisVectors(p=p, components=Jet(value=jnp.linalg.inv(B1), gradient=None, hessian=None, dim=2))
  theta2 = BasisVectors(p=p, components=Jet(value=jnp.linalg.inv(B2), gradient=None, hessian=None, dim=2))
  theta3 = BasisVectors(p=p, components=Jet(value=jnp.linalg.inv(B3), gradient=None, hessian=None, dim=2))

  T12 = get_dual_basis_transform(theta1, theta2).value
  T23 = get_dual_basis_transform(theta2, theta3).value
  T13 = get_dual_basis_transform(theta1, theta3).value

  # For dual transforms, composition order is T13 = T12 @ T23
  assert jnp.allclose(T13, T12 @ T23)


def test_dual_and_vector_transforms_pairing_identity():
  p = jnp.array([0., 0.])
  B_from = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  B_to = jnp.array([[0.0, 1.0], [1.0, 0.0]])

  vec_from = BasisVectors(p=p, components=Jet(value=B_from, gradient=None, hessian=None, dim=2))
  vec_to = BasisVectors(p=p, components=Jet(value=B_to, gradient=None, hessian=None, dim=2))

  theta_from = BasisVectors(p=p, components=Jet(value=jnp.linalg.inv(B_from), gradient=None, hessian=None, dim=2))
  theta_to = BasisVectors(p=p, components=Jet(value=jnp.linalg.inv(B_to), gradient=None, hessian=None, dim=2))

  T_vec = get_basis_transform(vec_from, vec_to).value
  T_dual = get_dual_basis_transform(theta_from, theta_to).value

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
  vec = BasisVectors(p=p, components=Jet(value=B, gradient=None, hessian=None, dim=2))
  dual = BasisVectors(p=p, components=Jet(value=jnp.linalg.inv(B), gradient=None, hessian=None, dim=2))
  assert jnp.allclose(dual.components.value @ vec.components.value, jnp.eye(2))


def test_dual_primal_pairing_invariance_under_transform():
  p = jnp.array([0., 0.])
  B1 = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  B2 = jnp.array([[0.0, 1.0], [1.0, 0.0]])
  vec1 = BasisVectors(p=p, components=Jet(value=B1, gradient=None, hessian=None, dim=2))
  vec2 = BasisVectors(p=p, components=Jet(value=B2, gradient=None, hessian=None, dim=2))
  dual1 = BasisVectors(p=p, components=Jet(value=jnp.linalg.inv(B1), gradient=None, hessian=None, dim=2))
  dual2 = BasisVectors(p=p, components=Jet(value=jnp.linalg.inv(B2), gradient=None, hessian=None, dim=2))

  T_vec = get_basis_transform(vec1, vec2).value
  T_dual = get_dual_basis_transform(dual1, dual2).value

  # Coordinates of identity pairing transform accordingly
  E = jnp.eye(2)
  Theta = jnp.eye(2)
  E2 = T_vec @ E
  Theta2 = Theta @ T_dual
  assert jnp.allclose(Theta2 @ E2, jnp.eye(2))


def spherical_to_cartesian(q_in):
    q_in = jnp.asarray(q_in)
    N = q_in.shape[0]
    r = q_in[0]
    phis = q_in[1:]

    def prod_sin(k):
        return jnp.prod(jnp.sin(phis[:k])) if k > 0 else 1.0

    coords = []
    for i in range(N):
        base = r * prod_sin(i)
        if i < N - 1:
            coords.append(base * jnp.cos(phis[i]))
        else:
            coords.append(base)
    return jnp.stack(coords)

def cartesian_to_spherical(x_in):
  x_in = jnp.asarray(x_in)
  N = x_in.shape[0]
  r = jnp.linalg.norm(x_in)
  phis = []
  for i in range(N - 1):
    if i < N - 2:
      phi = jnp.arctan2(jnp.linalg.norm(x_in[i+1:]), x_in[i])
    else:
      # Last angle
      phi = jnp.arctan2(x_in[-1], x_in[-2])
    phis.append(phi)
  return jnp.concatenate([jnp.array([r], dtype=x_in.dtype), jnp.stack(phis)])

def test_change_coordinates():
  q = jnp.array([1.0, jnp.pi / 4, jnp.pi / 6])
  x = spherical_to_cartesian(q)
  basis = BasisVectors(p=q, components=Jet(value=jnp.eye(3), gradient=None, hessian=None, dim=3))
  out: BasisVectors = change_coordinates(basis, spherical_to_cartesian, q)

  # Expected: components transform by dxdz where z = spherical_to_cartesian(x)
  J = jax.jacrev(spherical_to_cartesian)(q)  # dz/dx at q
  expected = jnp.linalg.inv(J)               # dxdz at (q -> x)

  assert jnp.allclose(out.p, x)
  assert jnp.allclose(out.components.value, expected)

def test_change_coordinates_round_trip():
  q = jnp.array([1.0, jnp.pi / 4, jnp.pi / 6])
  x = spherical_to_cartesian(q)
  value = random.normal(random.key(0), (3, 3))
  gradient = random.normal(random.key(0), (3, 3, 3))
  hessian = None
  basis = BasisVectors(p=q, components=Jet(value=value, gradient=gradient, hessian=hessian, dim=3))
  out: BasisVectors = change_coordinates(basis, spherical_to_cartesian, q)
  out2: BasisVectors = change_coordinates(out, cartesian_to_spherical, x)
  assert jnp.allclose(out2.p, q)
  assert jnp.allclose(out2.components.value, value)
  assert jnp.allclose(out2.components.gradient, gradient)

def test_change_coordinates_matches_basis_transform():
  q = jnp.array([1.0, jnp.pi / 4, jnp.pi / 6])
  x = spherical_to_cartesian(q)

  # Create a random basis
  value = random.normal(random.key(0), (3, 3))
  gradient = random.normal(random.key(0), (3, 3, 3))
  hessian = random.normal(random.key(0), (3, 3, 3, 3))
  x_basis = BasisVectors(p=q, components=Jet(value=value, gradient=gradient, hessian=hessian, dim=3))

  # Change its coordinates
  z_basis: BasisVectors = change_coordinates(x_basis, spherical_to_cartesian, q)

  # Get the basis transform
  transform = get_basis_transform(x_basis, z_basis)
  z_basis_components_comp: Jet = apply_covariant_transform(transform, x_basis.components)

  # Check that the transformed components are consistent with the coordinate change
  assert jnp.allclose(z_basis_components_comp.value, z_basis.components.value)
  assert jnp.allclose(z_basis_components_comp.gradient, z_basis.components.gradient)


def test_apply_contravariant_transform_vector():
  q = jnp.array([1.0, jnp.pi / 4, jnp.pi / 6])

  # Random basis to produce a nontrivial transform Jet
  value = random.normal(random.key(0), (3, 3))
  gradient = random.normal(random.key(0), (3, 3, 3))
  hessian = random.normal(random.key(0), (3, 3, 3, 3))
  x_basis = BasisVectors(p=q, components=Jet(value=value, gradient=gradient, hessian=hessian, dim=3))

  # Build a second basis via coordinate change and the basis transform T
  z_basis: BasisVectors = change_coordinates(x_basis, spherical_to_cartesian, q)
  T: Jet = get_basis_transform(x_basis, z_basis)

  # A contravariant vector's coordinates transform via W = T V
  V = random.normal(random.key(1), (3,))
  V_jet = Jet(value=V, gradient=None, hessian=None, dim=3)

  # Use the helper under test
  W_comp: Jet = apply_contravariant_transform(T, V_jet)

  # Independent expected result via direct matrix-vector multiply with the same Jet transform
  @jet_decorator
  def matvec(T_val: Array, v_val: Array) -> Array:
    return jnp.einsum("ij,j->i", T_val, v_val)

  expected: Jet = matvec(T.get_value_jet(), V_jet.get_value_jet())

  # Compare value and derivatives
  assert jnp.allclose(W_comp.value, expected.value)
  assert jnp.allclose(W_comp.gradient, expected.gradient)
  assert (W_comp.hessian is None and expected.hessian is None)
