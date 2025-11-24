import jax.numpy as jnp
import jax.random as random
import jax
import pytest
from local_coordinates.jet import Jet, function_to_jet, jet_decorator, get_identity_jet
from local_coordinates.basis import BasisVectors, get_standard_basis, get_basis_transform, change_coordinates
from local_coordinates.frame import Frame, change_basis, get_lie_bracket_between_frame_pairs, pushforward, frames_are_equivalent
from local_coordinates.tangent import lie_bracket as lie_bracket_vec, change_basis as change_basis_vec
from local_coordinates.basis import BasisVectors
from local_coordinates.tangent import lie_bracket, TangentVector, tangent_vectors_are_equivalent
from typing import Annotated
from local_coordinates.jacobian import function_to_jacobian

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

def test_orthogonal_coordinate_frame_inverse():
  z = jnp.array([1.7, 0.3])
  x_jet = function_to_jet(spherical_to_cartesian, z)
  dim = x_jet.gradient.shape[-1]
  dxdz_frame = Frame(
    p=x_jet.value,
    components=get_identity_jet(dim),
    basis=BasisVectors(
      p=x_jet.value,
      components=Jet(
        value=x_jet.gradient,
        gradient=x_jet.hessian,
        hessian=None
      )
    )
  )

  x = spherical_to_cartesian(z)
  z_jet = function_to_jet(cartesian_to_spherical, x)
  dzdx_frame = Frame(
    p=z_jet.value,
    components=get_identity_jet(dim),
    basis=BasisVectors(
      p=z_jet.value,
      components=Jet(
        value=z_jet.gradient,
        gradient=z_jet.hessian,
        hessian=None
      )
    )
  )

  @jet_decorator
  def matmul(A, B):
    return A @ B

  dirac_delta: Jet = matmul(dxdz_frame.components.get_value_jet(), dzdx_frame.components.get_value_jet())

  assert jnp.allclose(dirac_delta.value, jnp.eye(dim))
  assert jnp.allclose(dirac_delta.gradient, 0.0)
  assert jnp.allclose(dirac_delta.hessian, 0.0)


def test_coordinate_change_consistent_with_tangent_vector_change():
  p = jnp.array([0., 0.])
  key = random.key(0)
  vals = random.normal(key, (2, 2))
  grads = random.normal(key, (2, 2, 2))
  hessians = random.normal(key, (2, 2, 2, 2))

  basis = BasisVectors(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians))
  I_jet = get_identity_jet(2)
  frame = Frame(p=p, basis=basis, components=I_jet)
  V = frame.get_basis_vector(0)

  frame_coord_change = change_basis(frame, get_standard_basis(p))
  V_coord_change = change_basis(V, get_standard_basis(p))
  V_comp = frame_coord_change.get_basis_vector(0)

  assert tangent_vectors_are_equivalent(V, V_comp)
  assert tangent_vectors_are_equivalent(V, V_coord_change)

def test_lie_bracket_matches_tangent_vector_lie_bracket():
  p = jnp.array([0., 0.])
  key = random.key(0)
  vals = random.normal(key, (2, 2))
  grads = random.normal(key, (2, 2, 2))
  hessians = random.normal(key, (2, 2, 2, 2))
  basis = BasisVectors(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians))
  I_jet = get_identity_jet(2)
  frame = Frame(p=p, basis=basis, components=I_jet)
  V0 = frame.get_basis_vector(0)
  V1 = frame.get_basis_vector(1)
  bracket: TangentVector = lie_bracket(V0, V1)
  batched_bracket: Annotated[TangentVector, "D D"] = get_lie_bracket_between_frame_pairs(frame)
  assert tangent_vectors_are_equivalent(bracket, batched_bracket[0, 1])

def test_lie_bracket_of_basis_vectors_matches_random_frame():
  # Random 3D frame with derivatives
  p = jnp.array([0., 0., 0.])
  key = random.key(0)
  k1, k2, k3 = random.split(key, 3)

  E_vals = random.normal(k1, (3, 3))
  dE_vals = random.normal(k2, (3, 3, 3))
  H_vals = random.normal(k3, (3, 3, 3, 3))
  basis = BasisVectors(p=p, components=Jet(value=E_vals, gradient=dE_vals, hessian=H_vals))

  # Identity frame in its own basis
  c_vals = random.normal(key, (3, 3))
  c_grads = random.normal(key, (3, 3, 3))
  c_hessians = random.normal(key, (3, 3, 3, 3))
  c_jet = Jet(value=c_vals, gradient=c_grads, hessian=c_hessians)
  frame = Frame(p=p, basis=basis, components=c_jet)

  lie_bracket = get_lie_bracket_between_frame_pairs(frame)
  N = 3
  for i in range(N):
    for j in range(N):
      Ei = frame.get_basis_vector(i)
      Ej = frame.get_basis_vector(j)
      bracket_std = lie_bracket_vec(Ei, Ej)
      assert tangent_vectors_are_equivalent(bracket_std, lie_bracket[i, j])

@pytest.mark.xfail(reason="Frame pushforward Jet derivatives in y-coords not implemented yet")
def test_frame_pushforward_coordinate_frame_spherical_to_cartesian():
  # q-coordinates → x-coordinates via spherical_to_cartesian
  q = jnp.array([1.7, 0.4, -0.3])

  # Frame at q: coordinate frame (identity components) in the standard q-basis
  basis_q = get_standard_basis(q)
  dim = q.shape[0]
  frame_q = Frame(p=q, basis=basis_q, components=get_identity_jet(dim))

  pushed: Frame = pushforward(frame_q, spherical_to_cartesian)

  # Base point matches the image point
  x = spherical_to_cartesian(q)
  assert jnp.allclose(pushed.p, x)

  # For a coordinate frame, pushforward frame components should equal J^T under our storage convention
  # (rows index output component; columns index basis vector index)
  J = jax.jacrev(spherical_to_cartesian)(q)
  assert jnp.allclose(pushed.components.value, J.T)

@pytest.mark.xfail(reason="Lie bracket zero check depends on full y-coordinate Jet handling")
def test_lie_bracket_of_basis_vectors_zero_in_coordinate_frame():
  # q-coordinates → x-coordinates via spherical_to_cartesian
  q = jnp.array([1.7, 0.4, -0.3])

  # Frame at q: coordinate frame (identity components) in the standard q-basis
  basis_q = get_standard_basis(q)
  dim = q.shape[0]
  frame_q = Frame(p=q, basis=basis_q, components=get_identity_jet(dim))

  pushed: Frame = pushforward(frame_q, spherical_to_cartesian)

  lie_brackets: Annotated[TangentVector, "D D"] = get_lie_bracket_between_frame_pairs(pushed)
  assert jnp.allclose(lie_brackets.components.value, 0.0)
  assert jnp.allclose(lie_brackets.components.gradient, 0.0)
  assert jnp.allclose(lie_brackets.components.hessian, 0.0)

def test_frame_change_coordinates_round_trip():
  """
  Test that changing coordinates forward and backward preserves the Frame.
  x -> z -> x
  """
  q = jnp.array([2.5, jnp.pi / 3, jnp.pi / 4])
  x = spherical_to_cartesian(q)

  dim = 3
  key = random.key(1)

  basis_val = random.normal(key, (dim, dim))
  basis_grad = random.normal(key, (dim, dim, dim))
  basis_hess = random.normal(key, (dim, dim, dim, dim))
  basis = BasisVectors(p=q, components=Jet(value=basis_val, gradient=basis_grad, hessian=basis_hess, dim=dim))

  comp_val = random.normal(key, (dim, dim))
  comp_grad = random.normal(key, (dim, dim, dim))
  comp_hess = random.normal(key, (dim, dim, dim, dim))
  frame_comp = Jet(value=comp_val, gradient=comp_grad, hessian=comp_hess, dim=dim)

  frame_q = Frame(p=q, basis=basis, components=frame_comp)

  J_zq = function_to_jacobian(spherical_to_cartesian, q)
  frame_x = change_coordinates(frame_q, J_zq)

  J_xz = function_to_jacobian(cartesian_to_spherical, x)
  frame_q_restored = change_coordinates(frame_x, J_xz)

  assert jnp.allclose(frame_q_restored.basis.components.value, frame_q.basis.components.value, atol=1e-5)
  assert jnp.allclose(frame_q_restored.basis.components.gradient, frame_q.basis.components.gradient, atol=1e-5)
  assert jnp.allclose(frame_q_restored.basis.components.hessian, frame_q.basis.components.hessian, atol=1e-5)

  assert jnp.allclose(frame_q_restored.components.value, frame_q.components.value, atol=1e-5)
  assert jnp.allclose(frame_q_restored.components.gradient, frame_q.components.gradient, atol=1e-5)
  assert jnp.allclose(frame_q_restored.components.hessian, frame_q.components.hessian, atol=1e-5)

def test_frame_change_coordinates_vector_consistency():
  """
  Check that evaluating a frame vector gives the same physical vector
  before and after coordinate change.
  """
  q = jnp.array([2.0, jnp.pi/4, jnp.pi/4])
  x = spherical_to_cartesian(q)
  dim = 3

  basis_q = get_standard_basis(q)

  key = random.key(2)
  comp_val = random.normal(key, (dim, dim))
  frame_comp = Jet(value=comp_val, gradient=None, hessian=None, dim=dim)

  frame_q = Frame(p=q, basis=basis_q, components=frame_comp)

  J_zq = function_to_jacobian(spherical_to_cartesian, q)
  frame_x = change_coordinates(frame_q, J_zq)

  v_idx = 0
  v_q_obj = frame_q.get_basis_vector(0) # TangentVector
  v_x_obj = frame_x.get_basis_vector(0)

  J_val = J_zq.value # dx/dq
  J_inv = jnp.linalg.inv(J_val) # dq/dx

  v_q_std = v_q_obj.components.value @ basis_q.components.value
  v_x_std = v_x_obj.components.value @ frame_x.basis.components.value

  # Check geometric consistency
  assert jnp.allclose(v_x_std, v_q_std @ J_inv, atol=1e-5)

def test_frame_change_coordinates_scalar_components():
  """
  Verify that Frame components transform as scalars (invariants)
  but their derivatives change via chain rule.
  """
  def shift_map(q):
    return q + jnp.ones_like(q)

  q = jnp.array([1.0, 2.0])

  def comp_func(q):
    return jnp.einsum('i,j->ij', q**2, jnp.ones_like(q)) # Make it matrix

  comp_jet = function_to_jet(comp_func, q)
  dim = 2
  basis = get_standard_basis(q)

  frame_q = Frame(p=q, basis=basis, components=comp_jet)

  J_zq = function_to_jacobian(shift_map, q)
  frame_x = change_coordinates(frame_q, J_zq)

  assert jnp.allclose(frame_x.components.value, comp_jet.value)
  assert jnp.allclose(frame_x.components.gradient, comp_jet.gradient)
  assert jnp.allclose(frame_x.components.hessian, comp_jet.hessian)
