import jax.numpy as jnp
import jax.random as random
from local_coordinates.jet import Jet, function_to_jet, jet_decorator, get_identity_jet
from local_coordinates.basis import BasisVectors, get_standard_basis, get_basis_transform
from local_coordinates.frame import Frame, DualFrame, change_basis, get_lie_bracket_components
from local_coordinates.basis import BasisVectors


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

def test_frame_identity_change_is_noop():
  p = jnp.array([0.0, 0.0])

  # Start with an arbitrary basis
  B = jnp.array([[2.0, -1.0], [0.5, 3.0]])
  basis = BasisVectors(p=p, components=Jet(value=B, gradient=None, hessian=None, dim=2))

  # Frame components: identity (represents the basis itself)
  I = jnp.eye(2)
  frame = Frame(p=p, basis=basis, components=Jet(value=I, gradient=None, hessian=None, dim=2))

  # Changing to the same basis should not change components
  new_frame = change_basis(frame, basis)

  assert jnp.allclose(new_frame.components.value, I)
  assert jnp.allclose(new_frame.p, p)
  assert jnp.allclose(new_frame.basis.components.value, B)


def test_frame_change_matches_basis_transform():
  p = jnp.array([0.0, 0.0])

  # Source and target bases
  B_from = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  B_to = jnp.array([[0.0, 1.0], [1.0, 0.0]])
  basis_from = BasisVectors(p=p, components=Jet(value=B_from, gradient=None, hessian=None, dim=2))
  basis_to = BasisVectors(p=p, components=Jet(value=B_to, gradient=None, hessian=None, dim=2))

  # Arbitrary frame components (columns are frame vectors in 'from' basis coordinates)
  F = jnp.array([[1.0, 2.0], [3.0, 4.0]])
  frame = Frame(p=p, basis=basis_from, components=Jet(value=F, gradient=None, hessian=None, dim=2))

  # Apply change of coordinates
  new_frame = change_basis(frame, basis_to)

  # Expected: left-multiply by T = inv(B_to) @ B_from
  T = jnp.linalg.inv(B_to) @ B_from
  expected = T @ F

  assert jnp.allclose(new_frame.components.value, expected)


def test_frame_derivative_propagation_constant_transform():
  p = jnp.array([0.0, 0.0])

  # Constant transform (no derivatives), but frame has derivatives
  B_from = jnp.eye(2)
  B_to = jnp.array([[0.0, 1.0], [1.0, 0.0]])  # swap
  basis_from = BasisVectors(p=p, components=Jet(value=B_from, gradient=None, hessian=None, dim=2))
  basis_to = BasisVectors(p=p, components=Jet(value=B_to, gradient=None, hessian=None, dim=2))

  # Frame with nontrivial derivatives
  F = jnp.array([[1.0, 2.0], [3.0, 4.0]])
  dF = jnp.ones((2, 2, 2))  # shape (i, j, r)
  frame = Frame(p=p, basis=basis_from, components=Jet(value=F, gradient=dF, hessian=None))

  new_frame = change_basis(frame, basis_to)

  # Since T is constant, derivatives should transform by left-multiplication as well
  T = jnp.linalg.inv(B_to) @ B_from
  expected_grad = jnp.einsum("ij,jkr->ikr", T, dF)

  assert jnp.allclose(new_frame.components.value, T @ F)
  assert new_frame.components.gradient is not None
  assert jnp.allclose(new_frame.components.gradient, expected_grad)



def test_dualframe_identity_change_is_noop():
  p = jnp.array([0.0, 0.0])

  B = jnp.array([[2.0, -1.0], [0.5, 3.0]])
  basis = BasisVectors(p=p, components=Jet(value=B, gradient=None, hessian=None, dim=2))

  I = jnp.eye(2)
  dual_basis = BasisVectors(p=p, components=Jet(value=B, gradient=None, hessian=None, dim=2))
  dual = DualFrame(p=p, basis=dual_basis, components=Jet(value=I, gradient=None, hessian=None, dim=2))

  new_dual = change_basis(dual, dual_basis)

  assert jnp.allclose(new_dual.components.value, I)
  assert jnp.allclose(new_dual.p, p)
  assert jnp.allclose(new_dual.basis.components.value, B)


def test_dualframe_change_matches_basis_transform_inverse():
  p = jnp.array([0.0, 0.0])

  B_from = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  B_to = jnp.array([[0.0, 1.0], [1.0, 0.0]])
  basis_from = BasisVectors(p=p, components=Jet(value=B_from, gradient=None, hessian=None, dim=2))
  basis_to = BasisVectors(p=p, components=Jet(value=B_to, gradient=None, hessian=None, dim=2))
  dual_basis_from = BasisVectors(p=p, components=Jet(value=B_from, gradient=None, hessian=None, dim=2))
  dual_basis_to = BasisVectors(p=p, components=Jet(value=B_to, gradient=None, hessian=None, dim=2))

  C = jnp.array([[1.0, 2.0], [3.0, 4.0]])
  dual = DualFrame(p=p, basis=dual_basis_from, components=Jet(value=C, gradient=None, hessian=None, dim=2))

  new_dual = change_basis(dual, dual_basis_to)

  T = jnp.linalg.inv(B_to) @ B_from
  Tinv = jnp.linalg.inv(T)
  expected = C @ Tinv

  assert jnp.allclose(new_dual.components.value, expected)


def test_dualframe_derivative_propagation_constant_transform():
  p = jnp.array([0.0, 0.0])

  B_from = jnp.eye(2)
  B_to = jnp.array([[0.0, 1.0], [1.0, 0.0]])
  basis_from = BasisVectors(p=p, components=Jet(value=B_from, gradient=None, hessian=None, dim=2))
  basis_to = BasisVectors(p=p, components=Jet(value=B_to, gradient=None, hessian=None, dim=2))
  dual_basis_from = BasisVectors(p=p, components=Jet(value=B_from, gradient=None, hessian=None, dim=2))
  dual_basis_to = BasisVectors(p=p, components=Jet(value=B_to, gradient=None, hessian=None, dim=2))

  C = jnp.array([[1.0, 2.0], [3.0, 4.0]])
  dC = jnp.ones((2, 2, 2))
  dual = DualFrame(p=p, basis=dual_basis_from, components=Jet(value=C, gradient=dC, hessian=None))

  new_dual = change_basis(dual, dual_basis_to)

  T = jnp.linalg.inv(B_to) @ B_from
  Tinv = jnp.linalg.inv(T)
  expected_grad = jnp.einsum("imr,mj->ijr", dC, Tinv)

  assert jnp.allclose(new_dual.components.value, C @ Tinv)
  assert new_dual.components.gradient is not None
  assert jnp.allclose(new_dual.components.gradient, expected_grad)


def test_frame_dual_roundtrip():
  p = jnp.array([0.0, 0.0])
  B = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  basis = BasisVectors(p=p, components=Jet(value=B, gradient=None, hessian=None, dim=2))

  F = jnp.array([[2.0, -1.0], [0.0, 3.0]])
  frame = Frame(p=p, basis=basis, components=Jet(value=F, gradient=None, hessian=None, dim=2))

  dual = DualFrame(p=p, basis=basis, components=Jet(value=jnp.linalg.inv(F), gradient=None, hessian=None, dim=2))
  frame_back = Frame(p=p, basis=basis, components=Jet(value=F, gradient=None, hessian=None, dim=2))

  assert jnp.allclose(frame_back.components.value, F)
  assert jnp.allclose(dual.components.value @ frame.components.value, jnp.eye(2))


def test_dual_primal_invariance_under_change_of_coordinates():
  p = jnp.array([0.0, 0.0])
  B_from = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  B_to = jnp.array([[0.0, 1.0], [1.0, 0.0]])
  basis_from = BasisVectors(p=p, components=Jet(value=B_from, gradient=None, hessian=None, dim=2))
  basis_to = BasisVectors(p=p, components=Jet(value=B_to, gradient=None, hessian=None, dim=2))

  F = jnp.array([[1.0, 2.0], [3.0, 4.0]])
  frame = Frame(p=p, basis=basis_from, components=Jet(value=F, gradient=None, hessian=None, dim=2))
  dual = DualFrame(p=p, basis=basis_from, components=Jet(value=jnp.linalg.inv(F), gradient=None, hessian=None, dim=2))

  frame_new = change_basis(frame, basis_to)
  dual_new = change_basis(dual, basis_to)

  # Pairing should remain identity: θ(E) = I in any basis
  pairing_old = dual.components.value @ frame.components.value
  pairing_new = dual_new.components.value @ frame_new.components.value
  assert jnp.allclose(pairing_old, jnp.eye(2))
  assert jnp.allclose(pairing_new, jnp.eye(2))


def test_dualframe_primal_roundtrip():
  p = jnp.array([0.0, 0.0])
  B = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  basis = BasisVectors(p=p, components=Jet(value=B, gradient=None, hessian=None, dim=2))

  Theta = jnp.array([[0.5, 0.0], [0.2, 0.3]])  # invertible
  dual = DualFrame(p=p, basis=basis, components=Jet(value=Theta, gradient=None, hessian=None, dim=2))

  frame = dual.to_primal()
  dual_back = frame.to_dual()

  assert jnp.allclose(dual_back.components.value, Theta)
  # Pairing identity holds with the recovered primal
  assert jnp.allclose(dual_back.components.value @ frame.components.value, jnp.eye(2))


def test_change_coordinates_propagates_grad_and_hess_constant_T():
  p = jnp.array([0.0, 0.0])

  # Constant transform (no derivatives)
  B_from = jnp.eye(2)
  angle = 0.3
  c, s = jnp.cos(angle), jnp.sin(angle)
  B_to = jnp.array([[c, -s], [s, c]])

  basis_from = BasisVectors(p=p, components=Jet(value=B_from, gradient=None, hessian=None, dim=2))
  basis_to = BasisVectors(p=p, components=Jet(value=B_to, gradient=None, hessian=None, dim=2))

  # Frame with nontrivial derivatives
  F = jnp.array([[1.0, 2.0], [3.0, 4.0]])
  dF = 1.0*jnp.arange(2*2*2).reshape(2, 2, 2)  # (i,j,r)
  d2F = 1.0*jnp.arange(2*2*2*2).reshape(2, 2, 2, 2)  # (i,j,r,s)
  frame = Frame(p=p, basis=basis_from, components=Jet(value=F, gradient=dF, hessian=d2F))

  new_frame = change_basis(frame, basis_to)

  T = jnp.linalg.inv(B_to) @ B_from
  expected_val = T @ F
  expected_grad = jnp.einsum("im,mjr->ijr", T, dF)
  expected_hess = jnp.einsum("im,mjrs->ijrs", T, d2F)

  assert jnp.allclose(new_frame.components.value, expected_val)
  assert new_frame.components.gradient is not None
  assert jnp.allclose(new_frame.components.gradient, expected_grad)
  assert new_frame.components.hessian is not None
  assert jnp.allclose(new_frame.components.hessian, expected_hess)


def test_change_coordinates_full_chain_rule_variable_T():
  # Mathematically complete chain rule when the basis transform T depends on coordinates
  p = jnp.array([0.0, 0.0])

  def from_basis_func(x):  # 2x2, varies with both x[0] and x[1]
    A0 = jnp.array([[1.0, 0.5], [0.3, -0.7]])
    A1 = jnp.array([[0.2, -0.4], [0.6, 0.1]])
    return jnp.eye(2) + x[0] * A0 + x[1] * A1

  def to_basis_func(x):  # rotation-shear depending on x
    angle = 0.25 * x[0]
    c, s = jnp.cos(angle), jnp.sin(angle)
    R = jnp.array([[c, -s], [s, c]])
    S = jnp.array([[1.0 + 0.1 * x[1], 0.0], [0.0, 1.0 - 0.05 * x[1]]])
    return R @ S

  x0 = jnp.array([0.8, -0.4])
  from_components_jet = function_to_jet(from_basis_func, x0)
  to_components_jet = function_to_jet(to_basis_func, x0)

  basis_from = BasisVectors(p=p, components=from_components_jet)
  basis_to = BasisVectors(p=p, components=to_components_jet)

  # Frame with derivatives (with respect to same coordinates as bases)
  F = jnp.array([[1.2, -0.7], [0.4, 2.3]])
  dF = jnp.array([
    [[0.1, 0.2], [0.3, 0.4]],  # ∂/∂x0
    [[-0.5, 0.6], [0.7, -0.8]],  # ∂/∂x1
  ])
  d2F = jnp.array([
    [[[0.01, -0.02], [0.03, -0.04]], [[0.05, -0.06], [0.07, -0.08]]],  # ∂^2/∂x0∂x{0,1}
    [[[-0.09, 0.1], [0.11, -0.12]], [[0.13, -0.14], [0.15, -0.16]]],   # ∂^2/∂x1∂x{0,1}
  ])
  frame = Frame(p=p, basis=basis_from, components=Jet(value=F, gradient=dF, hessian=d2F))

  new_frame = change_basis(frame, basis_to)

  # Full chain rule for value, gradient, hessian
  T = get_basis_transform(basis_from, basis_to)

  expected_val = T.value @ F
  # ∂_r(TF) = (∂_r T) F + T (∂_r F)
  expected_grad = jnp.einsum("imr,mj->ijr", T.gradient, F) \
                  + jnp.einsum("im,mjr->ijr", T.value, dF)
  # ∂_{rs}(TF) = (∂_{rs}T)F + (∂_r T)(∂_s F) + (∂_s T)(∂_r F) + T(∂_{rs} F)
  expected_hess = jnp.einsum("imrs,mj->ijrs", T.hessian, F) \
                  + jnp.einsum("imr,mjs->ijrs", T.gradient, dF) \
                  + jnp.einsum("ims,mjr->ijrs", T.gradient, dF) \
                  + jnp.einsum("im,mjrs->ijrs", T.value, d2F)

  assert jnp.allclose(new_frame.components.value, expected_val)
  assert new_frame.components.gradient is not None
  assert jnp.allclose(new_frame.components.gradient, expected_grad)
  assert new_frame.components.hessian is not None
  assert jnp.allclose(new_frame.components.hessian, expected_hess)

def test_lie_bracket():
  # Construct a coordinate basis, then verify bracket vanishes via Frame API

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

  x0 = jnp.array([0.5, 0.5])

  inv_coord_vector_jet = function_to_jet(inv_chart, x0)  # dz/dx

  @jet_decorator
  def invert_basis(coord_grads):
    return jnp.linalg.inv(coord_grads)

  coord_vector_jet = invert_basis(inv_coord_vector_jet.get_gradient_jet())  # dx/dz

  p = jnp.array([0., 0.])
  basis = BasisVectors(p=p, components=coord_vector_jet)

  I = jnp.eye(2)
  frame = Frame(p=p, basis=basis, components=Jet(value=I, gradient=None, hessian=None, dim=2))

  c_jet = get_lie_bracket_components(frame)
  assert jnp.allclose(c_jet.value, 0.0)


def test_get_lie_bracket_components_zero_for_constant_frame():
  p = jnp.array([0.0, 0.0])
  A = jnp.array([[2.0, -1.0],
                 [1.5,  3.0]])
  dA = jnp.zeros((2, 2, 2))
  basis = BasisVectors(p=p, components=Jet(value=A, gradient=dA, hessian=None))

  I = jnp.eye(2)
  frame = Frame(p=p, basis=basis, components=Jet(value=I, gradient=None, hessian=None, dim=2))

  c_jet = get_lie_bracket_components(frame)
  assert jnp.allclose(c_jet.value, 0.0)


def test_get_lie_bracket_components_simple_noncommuting_frame():
  # E1 = (1, 0), E2 = (x, 1) -> [E1,E2] = (1,0) = E1
  p = jnp.array([0.3, -0.2])
  E = jnp.array([[1.0, p[0]],
                 [0.0, 1.0]])
  dE = jnp.zeros((2, 2, 2))
  dE = dE.at[0, 1, 0].set(1.0)  # ∂_x E2^x = 1

  basis = BasisVectors(p=p, components=Jet(value=E, gradient=dE, hessian=None))

  I = jnp.eye(2)
  frame = Frame(p=p, basis=basis, components=Jet(value=I, gradient=None, hessian=None, dim=2))

  out = get_lie_bracket_components(frame)
  c = out.value

  expected = jnp.zeros_like(c)
  expected = expected.at[0, 0, 1].set(1.0)
  expected = expected.at[0, 1, 0].set(-1.0)
  assert jnp.allclose(c, expected)

def test_change_basis_propagates_grad_and_hess():
  p = jnp.array([0., 0.])
  key = random.key(0)
  vals = random.normal(key, (2, 2))
  grads = random.normal(key, (2, 2, 2))
  hessians = random.normal(key, (2, 2, 2, 2))

  basis = BasisVectors(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians))
  I_jet = get_identity_jet(2)
  frame = Frame(p=p, basis=basis, components=I_jet)

  # Go to the standard basis
  standard_basis = get_standard_basis(frame.p)
  frame_standard: Frame = change_basis(frame, standard_basis)
  standard_components: Jet = frame_standard.components

  assert jnp.any(standard_components.value != 0), "standard_components.value is all zero"
  assert standard_components.gradient is not None and jnp.any(standard_components.gradient != 0), "standard_components.gradient is all zero or None"
  assert standard_components.hessian is not None and jnp.any(standard_components.hessian != 0), "standard_components.hessian is all zero or None"

def test_get_lie_bracket_components_random():
  p = jnp.array([0., 0.])
  key = random.key(0)
  vals = random.normal(key, (2, 2))
  grads = random.normal(key, (2, 2, 2))
  hessians = random.normal(key, (2, 2, 2, 2))

  basis = BasisVectors(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians))

  I_jet = get_identity_jet(2)
  frame = Frame(p=p, basis=basis, components=I_jet)

  c_jet = get_lie_bracket_components(frame)

  @jet_decorator
  def reconstruct(c_kij_vals, Ek_vals):
    return jnp.einsum("kij,ak->aij", c_kij_vals, Ek_vals)

  out: Jet = reconstruct(c_jet.value, basis.components)

  standard_basis = get_standard_basis(p)

  @jet_decorator
  def change_basis2(components_euclidean_vals, T_val):
    return jnp.einsum("ka,aij->kij", T_val, components_euclidean_vals)

  out2: Jet = change_basis2(out.get_value_jet(), standard_basis.components.get_value_jet())
