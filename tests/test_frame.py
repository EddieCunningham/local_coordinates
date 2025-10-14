import jax.numpy as jnp
import jax.random as random
from local_coordinates.jet import Jet, function_to_jet, jet_decorator
from local_coordinates.basis import BasisVectors, DualBasis, get_standard_basis, get_basis_transform
from local_coordinates.frame import Frame, DualFrame, change_coordinates
from local_coordinates.basis import BasisVectors


def test_frame_identity_change_is_noop():
  p = jnp.array([0.0, 0.0])

  # Start with an arbitrary basis
  B = jnp.array([[2.0, -1.0], [0.5, 3.0]])
  basis = BasisVectors(p=p, components=Jet(value=B, gradient=None, hessian=None))

  # Frame components: identity (represents the basis itself)
  I = jnp.eye(2)
  frame = Frame(p=p, basis=basis, components=Jet(value=I, gradient=None, hessian=None))

  # Changing to the same basis should not change components
  new_frame = change_coordinates(frame, basis)

  assert jnp.allclose(new_frame.components.value, I)
  assert jnp.allclose(new_frame.p, p)
  assert jnp.allclose(new_frame.basis.components.value, B)


def test_frame_change_matches_basis_transform():
  p = jnp.array([0.0, 0.0])

  # Source and target bases
  B_from = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  B_to = jnp.array([[0.0, 1.0], [1.0, 0.0]])
  basis_from = BasisVectors(p=p, components=Jet(value=B_from, gradient=None, hessian=None))
  basis_to = BasisVectors(p=p, components=Jet(value=B_to, gradient=None, hessian=None))

  # Arbitrary frame components (columns are frame vectors in 'from' basis coordinates)
  F = jnp.array([[1.0, 2.0], [3.0, 4.0]])
  frame = Frame(p=p, basis=basis_from, components=Jet(value=F, gradient=None, hessian=None))

  # Apply change of coordinates
  new_frame = change_coordinates(frame, basis_to)

  # Expected: left-multiply by T = inv(B_to) @ B_from
  T = jnp.linalg.inv(B_to) @ B_from
  expected = T @ F

  assert jnp.allclose(new_frame.components.value, expected)


def test_frame_derivative_propagation_constant_transform():
  p = jnp.array([0.0, 0.0])

  # Constant transform (no derivatives), but frame has derivatives
  B_from = jnp.eye(2)
  B_to = jnp.array([[0.0, 1.0], [1.0, 0.0]])  # swap
  basis_from = BasisVectors(p=p, components=Jet(value=B_from, gradient=None, hessian=None))
  basis_to = BasisVectors(p=p, components=Jet(value=B_to, gradient=None, hessian=None))

  # Frame with nontrivial derivatives
  F = jnp.array([[1.0, 2.0], [3.0, 4.0]])
  dF = jnp.ones((2, 2, 2))  # shape (i, j, r)
  frame = Frame(p=p, basis=basis_from, components=Jet(value=F, gradient=dF, hessian=None))

  new_frame = change_coordinates(frame, basis_to)

  # Since T is constant, derivatives should transform by left-multiplication as well
  T = jnp.linalg.inv(B_to) @ B_from
  expected_grad = jnp.einsum("ij,jkr->ikr", T, dF)

  assert jnp.allclose(new_frame.components.value, T @ F)
  assert new_frame.components.gradient is not None
  assert jnp.allclose(new_frame.components.gradient, expected_grad)



def test_dualframe_identity_change_is_noop():
  p = jnp.array([0.0, 0.0])

  B = jnp.array([[2.0, -1.0], [0.5, 3.0]])
  basis = BasisVectors(p=p, components=Jet(value=B, gradient=None, hessian=None))

  I = jnp.eye(2)
  dual_basis = DualBasis(p=p, components=Jet(value=jnp.linalg.inv(B), gradient=None, hessian=None))
  dual = DualFrame(p=p, basis=dual_basis, components=Jet(value=I, gradient=None, hessian=None))

  new_dual = change_coordinates(dual, dual_basis)

  assert jnp.allclose(new_dual.components.value, I)
  assert jnp.allclose(new_dual.p, p)
  assert jnp.allclose(new_dual.basis.components.value, jnp.linalg.inv(B))


def test_dualframe_change_matches_basis_transform_inverse():
  p = jnp.array([0.0, 0.0])

  B_from = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  B_to = jnp.array([[0.0, 1.0], [1.0, 0.0]])
  basis_from = BasisVectors(p=p, components=Jet(value=B_from, gradient=None, hessian=None))
  basis_to = BasisVectors(p=p, components=Jet(value=B_to, gradient=None, hessian=None))
  dual_basis_from = DualBasis(p=p, components=Jet(value=jnp.linalg.inv(B_from), gradient=None, hessian=None))
  dual_basis_to = DualBasis(p=p, components=Jet(value=jnp.linalg.inv(B_to), gradient=None, hessian=None))

  C = jnp.array([[1.0, 2.0], [3.0, 4.0]])
  dual = DualFrame(p=p, basis=dual_basis_from, components=Jet(value=C, gradient=None, hessian=None))

  new_dual = change_coordinates(dual, dual_basis_to)

  T = jnp.linalg.inv(B_to) @ B_from
  Tinv = jnp.linalg.inv(T)
  expected = C @ Tinv

  assert jnp.allclose(new_dual.components.value, expected)


def test_dualframe_derivative_propagation_constant_transform():
  p = jnp.array([0.0, 0.0])

  B_from = jnp.eye(2)
  B_to = jnp.array([[0.0, 1.0], [1.0, 0.0]])
  basis_from = BasisVectors(p=p, components=Jet(value=B_from, gradient=None, hessian=None))
  basis_to = BasisVectors(p=p, components=Jet(value=B_to, gradient=None, hessian=None))
  dual_basis_from = DualBasis(p=p, components=Jet(value=jnp.linalg.inv(B_from), gradient=None, hessian=None))
  dual_basis_to = DualBasis(p=p, components=Jet(value=jnp.linalg.inv(B_to), gradient=None, hessian=None))

  C = jnp.array([[1.0, 2.0], [3.0, 4.0]])
  dC = jnp.ones((2, 2, 2))
  dual = DualFrame(p=p, basis=dual_basis_from, components=Jet(value=C, gradient=dC, hessian=None))

  new_dual = change_coordinates(dual, dual_basis_to)

  T = jnp.linalg.inv(B_to) @ B_from
  Tinv = jnp.linalg.inv(T)
  expected_grad = jnp.einsum("imr,mj->ijr", dC, Tinv)

  assert jnp.allclose(new_dual.components.value, C @ Tinv)
  assert new_dual.components.gradient is not None
  assert jnp.allclose(new_dual.components.gradient, expected_grad)


def test_frame_dual_roundtrip():
  p = jnp.array([0.0, 0.0])
  B = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  basis = BasisVectors(p=p, components=Jet(value=B, gradient=None, hessian=None))

  F = jnp.array([[2.0, -1.0], [0.0, 3.0]])
  frame = Frame(p=p, basis=basis, components=Jet(value=F, gradient=None, hessian=None))

  dual = frame.to_dual()
  frame_back = dual.to_primal()

  assert jnp.allclose(frame_back.components.value, F)
  assert jnp.allclose(dual.components.value @ frame.components.value, jnp.eye(2))


def test_dual_primal_invariance_under_change_of_coordinates():
  p = jnp.array([0.0, 0.0])
  B_from = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  B_to = jnp.array([[0.0, 1.0], [1.0, 0.0]])
  basis_from = BasisVectors(p=p, components=Jet(value=B_from, gradient=None, hessian=None))
  basis_to = BasisVectors(p=p, components=Jet(value=B_to, gradient=None, hessian=None))

  F = jnp.array([[1.0, 2.0], [3.0, 4.0]])
  frame = Frame(p=p, basis=basis_from, components=Jet(value=F, gradient=None, hessian=None))
  dual = frame.to_dual()

  frame_new = change_coordinates(frame, basis_to)
  dual_basis_to = DualBasis(p=p, components=Jet(value=jnp.linalg.inv(B_to), gradient=None, hessian=None))
  dual_new = change_coordinates(dual, dual_basis_to)

  # Pairing should remain identity: θ(E) = I in any basis
  pairing_old = dual.components.value @ frame.components.value
  pairing_new = dual_new.components.value @ frame_new.components.value
  assert jnp.allclose(pairing_old, jnp.eye(2))
  assert jnp.allclose(pairing_new, jnp.eye(2))


def test_dualframe_primal_roundtrip():
  p = jnp.array([0.0, 0.0])
  B = jnp.array([[1.0, 0.5], [0.0, 1.0]])
  basis = BasisVectors(p=p, components=Jet(value=B, gradient=None, hessian=None))

  Theta = jnp.array([[0.5, 0.0], [0.2, 0.3]])  # invertible
  dual = DualFrame(p=p, basis=basis, components=Jet(value=Theta, gradient=None, hessian=None))

  frame = dual.to_primal()
  dual_back = frame.to_dual()

  assert jnp.allclose(dual_back.components.value, Theta)
  # Pairing identity holds with the recovered primal
  assert jnp.allclose(dual_back.components.value @ frame.components.value, jnp.eye(2))

