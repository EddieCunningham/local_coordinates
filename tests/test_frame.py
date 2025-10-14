import jax.numpy as jnp
import jax.random as random
from local_coordinates.jet import Jet, function_to_jet, jet_decorator
from local_coordinates.basis import BasisVectors, get_standard_basis, get_basis_transform
from local_coordinates.frame import Frame, change_coordinates
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


