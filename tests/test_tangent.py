import jax.numpy as jnp
import jax
import jax.random as random
import pytest

from local_coordinates.jet import Jet, jet_decorator, function_to_jet, get_identity_jet
from local_coordinates.basis import BasisVectors, get_basis_transform, get_standard_basis, apply_contravariant_transform
from local_coordinates.tangent import TangentVector, change_basis, lie_bracket, tangent_vectors_are_equivalent, pushforward


def test_tangent_vector_creation_fields():
  p = jnp.array([1.0, 2.0])
  V = jnp.array([3.0, -4.0])
  components_jet = Jet(value=V, gradient=None, hessian=None, dim=2)
  basis = get_standard_basis(p)

  tv = TangentVector(p=p, components=components_jet, basis=basis)

  assert jnp.allclose(tv.p, p)
  assert jnp.allclose(tv.components.value, V)
  assert jnp.allclose(tv.basis.components.value, basis.components.value)


def test_tangent_vector_invalid_components_dim_raises():
  p = jnp.array([1.0, 2.0])
  # Components with ndim=2 while p.ndim=1 should raise
  bad_components = Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2)
  basis = get_standard_basis(p)

  with pytest.raises(ValueError):
    TangentVector(p=p, components=bad_components, basis=basis)


def test_tangent_vector_batch_size_delegates_to_basis():
  # Batch of 3 points in R^2
  p_batch = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  # Batched identity bases
  b_batch = jnp.stack([jnp.eye(2)] * 3)
  basis = BasisVectors(p=p_batch, components=Jet(value=b_batch, gradient=None, hessian=None, dim=2))

  V_batch = random.normal(random.key(0), (3, 2))
  components = Jet(value=V_batch, gradient=None, hessian=None, dim=2)

  tv = TangentVector(p=p_batch, components=components, basis=basis)
  assert tv.batch_size == 3


def test_change_basis_contravariant_transform_matches_apply():
  # Build two (potentially nontrivial) bases in R^3 with derivative data
  N = 3
  p = jnp.zeros((N,))

  k = random.key(1)
  k1, k2, k3, k4, k5, k6, k7 = random.split(k, 7)

  from_value = random.normal(k1, (N, N))
  from_grad = random.normal(k2, (N, N, N))
  from_hess = random.normal(k3, (N, N, N, N))
  from_basis = BasisVectors(p=p, components=Jet(value=from_value, gradient=from_grad, hessian=from_hess))

  to_value = random.normal(k4, (N, N))
  to_grad = random.normal(k5, (N, N, N))
  to_hess = random.normal(k6, (N, N, N, N))
  to_basis = BasisVectors(p=p, components=Jet(value=to_value, gradient=to_grad, hessian=to_hess))

  T = get_basis_transform(from_basis, to_basis)

  V = random.normal(k7, (N,))
  V_jet = Jet(value=V, gradient=None, hessian=None, dim=N)
  tv = TangentVector(p=p, components=V_jet, basis=from_basis)

  tv_new = change_basis(tv, to_basis)
  expected = apply_contravariant_transform(T, V_jet)

  assert jnp.allclose(tv_new.components.value, expected.value)
  assert jnp.allclose(tv_new.basis.components.value, to_basis.components.value)


def test_change_basis_round_trip_restores_components():
  N = 2
  p = jnp.array([0.0, 0.0])
  key = random.key(0)
  k1, k2, k3 = random.split(key, 3)

  # Two random bases (full rank with high probability)
  b1 = BasisVectors(
    p=p,
    components=Jet(
      value=random.normal(k1, (N, N)),
      gradient=random.normal(k2, (N, N, N)),
      hessian=None,
    ),
  )
  b2 = BasisVectors(
    p=p,
    components=Jet(
      value=random.normal(k3, (N, N)),
      gradient=None,
      hessian=None,
      dim=N,
    ),
  )

  V = jnp.array([1.0, -2.0])
  tv = TangentVector(p=p, components=Jet(value=V, gradient=None, hessian=None, dim=N), basis=b1)
  tv_b2 = change_basis(tv, b2)
  tv_back = change_basis(tv_b2, b1)

  assert jnp.allclose(tv_back.components.value, V)


def test_call_invariant_under_basis_change():
  # Dimension and point
  N = 3
  p = jnp.array([0.2, -0.3, 0.5])

  # Construct two invertible bases at p
  key = random.key(42)
  k1, k2, k3 = random.split(key, 3)
  vals = random.normal(k1, (N, N))
  grads = random.normal(k1, (N, N, N))
  hessians = random.normal(k1, (N, N, N, N))
  from_basis = BasisVectors(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians))

  vals = random.normal(k2, (N, N))
  grads = random.normal(k2, (N, N, N))
  hessians = random.normal(k2, (N, N, N, N))
  to_basis = BasisVectors(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians))

  # Same geometric vector represented in the two bases
  tv_from = TangentVector(p=p, components=get_identity_jet(N)[0], basis=from_basis)
  tv_to: TangentVector = change_basis(tv_from, to_basis)

  # Define a smooth function and build its Jet at p
  def f(x):
    return jnp.array([
      x[0]**2 + 3.0 * x[1] - 0.7 * x[2],
      jnp.sin(x[0]) + x[1]**3 + x[2],
    ])

  f_jet = function_to_jet(f, p)

  # Apply the tangent vector (derivation) to the function's Jet
  out_from = tv_from(f_jet)
  out_to = tv_to(f_jet)

  # Results must be basis-independent
  assert jnp.allclose(out_from.value, out_to.value)
  assert jnp.allclose(out_from.gradient, out_to.gradient)
  assert jnp.allclose(out_from.hessian, out_to.hessian)


def test_lie_bracket_zero_for_constant_vectors():
  p = jnp.array([0.0, 0.0])
  basis = get_standard_basis(p)

  X = TangentVector(
    p=p,
    basis=basis,
    components=Jet(value=jnp.array([1.0, 0.0]), gradient=jnp.zeros((2, 2)), hessian=None),
  )
  Y = TangentVector(
    p=p,
    basis=basis,
    components=Jet(value=jnp.array([0.0, 1.0]), gradient=jnp.zeros((2, 2)), hessian=None),
  )

  bracket = lie_bracket(X, Y)

  assert jnp.allclose(bracket.components.value, jnp.zeros(2))
  assert jnp.allclose(bracket.components.gradient, 0.0)


def test_lie_bracket_simple_noncommuting_vectors():
  # In R^2 with coordinates (x, y), take X = (1, 0), Y = (x, 1).
  # Expected [X, Y] = (1, 0) in standard convention.
  p = jnp.array([0.3, -0.2])
  basis = get_standard_basis(p)

  X = TangentVector(
    p=p,
    basis=basis,
    components=Jet(value=jnp.array([1.0, 0.0]), gradient=jnp.zeros((2, 2)), hessian=None),
  )

  Y_val = jnp.array([p[0], 1.0])
  # dY/dx = (1, 0), dY/dy = (0, 0); last axis indexes coordinates (x, y)
  dYdx = jnp.array([1.0, 0.0])
  dYdy = jnp.array([0.0, 0.0])
  Y_grad = jnp.stack([dYdx, dYdy], axis=-1)
  Y = TangentVector(
    p=p,
    basis=basis,
    components=Jet(value=Y_val, gradient=Y_grad, hessian=None),
  )

  bracket = lie_bracket(X, Y)

  expected = jnp.array([1.0, 0.0])
  # Value should match the standard definition of the Lie bracket
  assert jnp.allclose(bracket.components.value, expected)
  # Gradient is zero since expected is constant w.r.t. coordinates in this setup
  assert jnp.allclose(bracket.components.gradient, 0.0)


def test_lie_bracket_change_of_basis_invariance():
  # Build two bases at the same point; compute bracket in either basis and compare.
  N = 3
  p = jnp.array([0.2, 0.1, -0.4])

  key = random.key(7)
  k1, k2, k3, kx, ky = random.split(key, 5)

  # Basis 1: standard for simplicity
  basis1 = get_standard_basis(p)

  # Basis 2: random with derivatives
  vals = random.normal(k1, (N, N))
  grads = random.normal(k2, (N, N, N))
  hessians = random.normal(k3, (N, N, N, N))
  basis2 = BasisVectors(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians))

  # Vector fields X, Y with nontrivial values and gradients
  X = TangentVector(
    p=p,
    basis=basis1,
    components=Jet(
      value=random.normal(kx, (N,)),
      gradient=random.normal(kx, (N, N)),
      hessian=random.normal(kx, (N, N, N)),
    ),
  )
  Y = TangentVector(
    p=p,
    basis=basis1,
    components=Jet(
      value=random.normal(ky, (N,)),
      gradient=random.normal(ky, (N, N)),
      hessian=random.normal(ky, (N, N, N)),
    ),
  )

  # Bracket in basis 1, then move to basis 2
  bracket1 = lie_bracket(X, Y)
  bracket1_in_basis2 = change_basis(bracket1, basis2)

  # Transform X, Y to basis 2 first, then bracket
  X2 = change_basis(X, basis2)
  Y2 = change_basis(Y, basis2)
  bracket2 = lie_bracket(X2, Y2)

  assert tangent_vectors_are_equivalent(bracket1_in_basis2, bracket2)

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

@pytest.mark.xfail(reason="Pushforward jet w.r.t. y-coordinates not possible in general")
def test_tangent_pushforward_definition():
  """
  dF(X)(f) = X(f\circ F)
  """
  # Choose a non-degenerate spherical point
  q = jnp.array([1.7, 0.4, -0.3])
  F = spherical_to_cartesian
  p = F(q)

  def f(x):
    return jnp.array([
      x[0]**2 + 3.0 * x[1] - 0.7 * x[2],
      jnp.sin(x[0]) + x[1]**3 + x[2],
    ])

  f_jet = function_to_jet(f, p)

  def f_circ_F(q):
    return f(F(q))

  f_circ_F_jet = function_to_jet(f_circ_F, q)

  # Build a tangent vector at q in the standard (coordinate) basis
  key = random.key(0)
  k1, k2, k3 = random.split(key, 3)
  V_val = random.normal(k1, (3,))
  V_grad = random.normal(k2, (3, 3))
  V_hess = random.normal(k3, (3, 3, 3))
  V_jet = Jet(value=V_val, gradient=V_grad, hessian=V_hess)
  basis_q = get_standard_basis(q)
  X = TangentVector(p=q, components=V_jet, basis=basis_q)

  dF_X: TangentVector = pushforward(X, F)

  out1 = dF_X(f_jet)
  out2 = X(f_circ_F_jet)

  assert jnp.allclose(out1.value, out2.value)
  assert jnp.allclose(out1.gradient, out2.gradient)
  assert jnp.allclose(out1.hessian, out2.hessian)

@pytest.mark.xfail(reason="Pushforward jet w.r.t. y-coordinates not possible in general")
def test_pushforward_round_trip_restores_components():
  # Choose a non-degenerate spherical point
  q = jnp.array([1.7, 0.4, -0.3])
  p = spherical_to_cartesian(q)

  # Build a tangent vector at q in the standard (coordinate) basis
  V = jnp.array([0.9, -1.2, 0.7])
  key = random.key(0)
  k1, k2 = random.split(key)
  V_grad = random.normal(k1, (3, 3))
  V_hess = random.normal(k2, (3, 3, 3))
  V_jet = Jet(value=V, gradient=V_grad, hessian=V_hess)
  basis_q = get_standard_basis(q)
  X = TangentVector(p=q, components=V_jet, basis=basis_q)

  f_jet: Jet = function_to_jet(spherical_to_cartesian, q) # f, df/dx, d²f/dx²
  f_inv_jet: Jet = function_to_jet(cartesian_to_spherical, p) # f, df/dx, d²f/dx²


  blah1 = apply_contravariant_transform(f_jet.get_gradient_jet(), X.components)
  blah2 = apply_contravariant_transform(f_inv_jet.get_gradient_jet(), blah1)

  Y = pushforward(X, spherical_to_cartesian)
  X2 = pushforward(Y, cartesian_to_spherical)
  assert tangent_vectors_are_equivalent(X, X2)
  assert jnp.allclose(X2.components.value, X.components.value)
  assert jnp.allclose(X2.components.gradient, X.components.gradient)
  assert jnp.allclose(X2.components.hessian, X.components.hessian)

@pytest.mark.xfail(reason="Pushforward jet w.r.t. y-coordinates not possible in general")
def test_tangent_pushforward_does_not_affect_lie_bracket():
  # Choose a non-degenerate spherical point
  q = jnp.array([1.7, 0.4, -0.3])
  basis_q = get_standard_basis(q)
  x = spherical_to_cartesian(q)

  # Build a tangent vector at q in the standard (coordinate) basis
  key = random.key(0)
  k1, k2, k3 = random.split(key, 3)
  X_val, Y_val = random.normal(k1, (2, 3))
  X_grad, Y_grad = random.normal(k2, (2, 3, 3))
  X_hess, Y_hess = random.normal(k3, (2, 3, 3, 3))
  X_jet = Jet(value=X_val, gradient=X_grad, hessian=X_hess)
  Y_jet = Jet(value=Y_val, gradient=Y_grad, hessian=Y_hess)
  X = TangentVector(p=q, components=X_jet, basis=basis_q)
  Y = TangentVector(p=q, components=Y_jet, basis=basis_q)

  # [F_*X, F_*Y]
  F_X = pushforward(X, spherical_to_cartesian)
  F_Y = pushforward(Y, spherical_to_cartesian)
  F_bracket = lie_bracket(F_X, F_Y)

  # F_*[X, Y]
  bracket = lie_bracket(X, Y)
  F_bracket2 = pushforward(bracket, spherical_to_cartesian)

  F_bracket_standard = change_basis(F_bracket, get_standard_basis(x))
  F_bracket2_standard = change_basis(F_bracket2, get_standard_basis(x))
  assert jnp.allclose(F_bracket_standard.components.value, F_bracket2_standard.components.value)
  assert jnp.allclose(F_bracket_standard.components.gradient, F_bracket2_standard.components.gradient)
  assert jnp.allclose(F_bracket_standard.components.hessian, F_bracket2_standard.components.hessian)


  assert tangent_vectors_are_equivalent(F_bracket, F_bracket2)
