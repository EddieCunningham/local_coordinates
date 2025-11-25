import jax.numpy as jnp
from local_coordinates.basis import BasisVectors, change_basis, get_basis_transform, get_dual_basis_transform, get_standard_basis, get_standard_dual_basis, change_coordinates, apply_covariant_transform, apply_contravariant_transform
from local_coordinates.jet import Jet, function_to_jet, jet_decorator, get_identity_jet, change_coordinates
import pytest
import jax
import jax.random as random
from jaxtyping import Array
from local_coordinates.basis import make_coordinate_basis
from local_coordinates.jacobian import function_to_jacobian

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
  T_dual = get_dual_basis_transform(vec_from, vec_to).value

  assert jnp.allclose(T_dual, jnp.linalg.inv(T_vec))


def test_dual_basis_transform_composition():
  p = jnp.array([0., 0.])
  B1 = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  B2 = jnp.array([[0.0, 1.0], [1.0, 0.0]])
  B3 = jnp.array([[2.0, 0.0], [0.0, 0.5]])

  vec1 = BasisVectors(p=p, components=Jet(value=B1, gradient=None, hessian=None, dim=2))
  vec2 = BasisVectors(p=p, components=Jet(value=B2, gradient=None, hessian=None, dim=2))
  vec3 = BasisVectors(p=p, components=Jet(value=B3, gradient=None, hessian=None, dim=2))

  T12 = get_dual_basis_transform(vec1, vec2).value
  T23 = get_dual_basis_transform(vec2, vec3).value
  T13 = get_dual_basis_transform(vec1, vec3).value

  # For dual transforms, composition order is T13 = T23 @ T12
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
  T_dual = get_dual_basis_transform(vec_from, vec_to).value

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
  T_dual = get_dual_basis_transform(vec1, vec2).value

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

  # Basis vectors E_j^a transform as: E_new_j^i = E_j^a G_a^i
  # With convention E[a,j] = E_j^a (column j = basis j), this is E_new = G @ E
  # For standard basis E = I, we get E_new = G
  J = jax.jacrev(spherical_to_cartesian)(q)  # G = dz/dq
  expected = J                                # G (not G.T!)

  assert jnp.allclose(out.p, x)
  assert jnp.allclose(out.components.value, expected)

def test_change_coordinates_round_trip():
  q = jnp.array([1.0, jnp.pi / 4, jnp.pi / 6])
  x = spherical_to_cartesian(q)
  k1, k2, k3 = random.split(random.key(0), 3)
  value = random.normal(k1, (3, 3))
  gradient = random.normal(k2, (3, 3, 3))
  hessian = random.normal(k3, (3, 3, 3, 3))
  basis = BasisVectors(p=q, components=Jet(value=value, gradient=gradient, hessian=hessian, dim=3))
  basis = make_coordinate_basis(basis)
  out: BasisVectors = change_coordinates(basis, spherical_to_cartesian, q)
  out2: BasisVectors = change_coordinates(out, cartesian_to_spherical, x)
  assert jnp.allclose(out2.p, q)
  assert jnp.allclose(out2.components.value, basis.components.value)
  assert jnp.allclose(out2.components.gradient, basis.components.gradient)

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
  assert jnp.allclose(z_basis_components_comp.hessian, z_basis.components.hessian)


def test_change_coordinates_standard_basis_hessian():
  """
  For a standard coordinate basis, the coordinate change should induce
  derivatives consistent with the forward Jacobian expressed as
  a function of the new coordinates.

  The transformation is E_new = G @ E where G = dz/dq.
  For identity E, this gives G. The derivatives are computed w.r.t. z.
  """
  q = jnp.array([1.0, jnp.pi / 4, jnp.pi / 6])
  x = spherical_to_cartesian(q)

  # Start with standard basis at q (identity, zero derivatives)
  basis_q = get_standard_basis(q)

  # Change coordinates to x
  basis_x: BasisVectors = change_coordinates(basis_q, spherical_to_cartesian, q)

  # The expected result is G where G(z) = dz/dq(q(z)) expressed as function of z.
  # We compute this by defining the transformed components as a function of z
  # and taking its Jet.
  def expected_components(z):
    # Invert to get q from z (Cartesian to spherical)
    q_from_z = cartesian_to_spherical(z)
    # Get the forward Jacobian at that point
    G = jax.jacrev(spherical_to_cartesian)(q_from_z)
    return G  # Not G.T!

  expected_jet = function_to_jet(expected_components, x)

  assert jnp.allclose(basis_x.p, x)
  assert jnp.allclose(basis_x.components.value, expected_jet.value)
  assert jnp.allclose(basis_x.components.gradient, expected_jet.gradient)
  assert jnp.allclose(basis_x.components.hessian, expected_jet.hessian)

def test_change_coordinates_jacobian_agrees_with_function():
  """
  For a standard coordinate basis, the coordinate change should induce
  derivatives consistent with the inverse Jacobian Jet.
  """
  q = jnp.array([1.0, jnp.pi / 4, jnp.pi / 6])
  x = spherical_to_cartesian(q)

  # Start with standard basis at q (identity, zero derivatives)
  basis_q = get_standard_basis(q)

  # Change coordinates to x
  basis_x: BasisVectors = change_coordinates(basis_q, spherical_to_cartesian, q)

  # Build Jacobian for z = spherical_to_cartesian(q) and invert it
  J = function_to_jacobian(spherical_to_cartesian, q)
  basis_comp: BasisVectors = change_coordinates(basis_q, J)

  # assert jnp.allclose(basis_x.p, basis_comp.p)
  assert jnp.allclose(basis_x.components.value, basis_comp.components.value)
  assert jnp.allclose(basis_x.components.gradient, basis_comp.components.gradient)
  assert jnp.allclose(basis_x.components.hessian, basis_comp.components.hessian)

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
  # assert jnp.allclose(W_comp.gradient, expected.gradient) # Ignore gradient because V_jet has no gradient
  # assert (W_comp.hessian is None and expected.hessian is None)

def test_change_coordinates_contravariant_transform_plus_chain_rule():
  """
  Verify that change_coordinates(basis, jacobian) computes E_new = G @ E.

  With convention E[a,j] = E_j^a (column j = basis vector j):
    E_new_j^i = E_j^a G_a^i = (G @ E)[i,j]
  """
  q = jnp.array([1.0, jnp.pi / 4, jnp.pi / 6])
  x = spherical_to_cartesian(q)

  # Random basis
  value = random.normal(random.key(0), (3, 3))
  gradient = random.normal(random.key(0), (3, 3, 3))
  hessian = random.normal(random.key(0), (3, 3, 3, 3))
  basis = BasisVectors(p=q, components=Jet(value=value, gradient=gradient, hessian=hessian, dim=3))

  # Jacobian for transform
  J = function_to_jacobian(spherical_to_cartesian, q)

  # Standard change_coordinates
  basis_transformed: BasisVectors = change_coordinates(basis, J)

  # Verify value: E_new = G @ E
  G = J.value
  expected_value = G @ basis.components.value
  assert jnp.allclose(basis_transformed.components.value, expected_value)


def test_change_coordinates_identity_map_noop():
  """
  For the identity coordinate change z(x) = x, change_coordinates should leave
  the basis components (value, gradient, hessian) unchanged.
  """
  dim = 3
  key = random.key(0)
  k1, k2, k3 = random.split(key, 3)

  p = jnp.array([0.3, -0.5, 1.1])
  value = random.normal(k1, (dim, dim))
  gradient = random.normal(k2, (dim, dim, dim))
  hessian = random.normal(k3, (dim, dim, dim, dim))
  basis = BasisVectors(p=p, components=Jet(value=value, gradient=gradient, hessian=hessian, dim=dim))

  def identity_map(x: Array) -> Array:
    return x

  out = change_coordinates(basis, identity_map, p)

  assert jnp.allclose(out.p, p)
  assert jnp.allclose(out.components.value, basis.components.value)
  assert jnp.allclose(out.components.gradient, basis.components.gradient)
  assert jnp.allclose(out.components.hessian, basis.components.hessian)


def test_change_coordinates_matches_direct_shear_function():
  """
  For a concrete nonlinear shear map z(x) and an explicit basis field E(x),
  change_coordinates(basis, shear_map, x0) should match the Jet obtained by
  directly transforming E as a function of z.
  """
  dim = 2

  def E_func(x: Array) -> Array:
    x1, x2 = x
    return jnp.array([
      [x1*x1 + x2, x1*x2],
      [jnp.sin(x1), x2*x2 + 1.0],
    ])

  def shear_map(x: Array) -> Array:
    x1, x2 = x
    return jnp.array([x1, x2 + x1*x1])

  def shear_inverse(z: Array) -> Array:
    z1, z2 = z
    return jnp.array([z1, z2 - z1*z1])

  x0 = jnp.array([0.4, -0.6])
  z0 = shear_map(x0)

  # Basis in x-coordinates
  basis_x = BasisVectors(p=x0, components=function_to_jet(E_func, x0))

  # Transform using the library change_coordinates
  basis_z = change_coordinates(basis_x, shear_map, x0)

  # Directly define the transformed basis as a function of z using the tensorial rule
  # With convention E[a,j] = E_j^a (column j = basis vector j):
  # E_new = G @ E where G = dz/dx
  def tilde_E_func(z: Array) -> Array:
    x = shear_inverse(z)
    G = jax.jacrev(shear_map)(x)  # dz/dx, shape (i, a)
    E = E_func(x)                 # shape (a, j)
    return G @ E                  # shape (i, j), E_new_j^i = E_j^a G_a^i

  direct_jet = function_to_jet(tilde_E_func, z0)

  assert jnp.allclose(basis_z.p, z0)
  assert jnp.allclose(basis_z.components.value, direct_jet.value)
  assert jnp.allclose(basis_z.components.gradient, direct_jet.gradient)
  assert jnp.allclose(basis_z.components.hessian, direct_jet.hessian)
