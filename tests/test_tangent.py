import jax.numpy as jnp
import jax
import jax.random as random
import pytest
import numpy as np
import sympy
from sympy import symbols

from local_coordinates.jet import Jet, jet_decorator, function_to_jet, get_identity_jet
from local_coordinates.basis import BasisVectors, get_basis_transform, get_standard_basis, apply_contravariant_transform, change_coordinates
from local_coordinates.tangent import TangentVector, change_basis, lie_bracket, tangent_vectors_are_equivalent, pushforward, change_coordinates
from local_coordinates.jacobian import function_to_jacobian

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
  # assert jnp.allclose(bracket.components.gradient, 0.0) # Don't check this because X and Y have no hessian


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
  # assert jnp.allclose(bracket.components.gradient, 0.0) # Don't check gradient because X and Y have no gradient



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


def test_lie_bracket_matches_sympy_value_grad_hess_coordinate_basis():
  # Coordinate basis in R^2; define polynomial vector fields with nontrivial derivatives.
  x, y = symbols("x, y")
  X_sym = [x**2 + x*y + 1, x*y + y**2 + 2]
  Y_sym = [x + y**2, x**2 - y + 3]

  # [X,Y]^k = X^i ∂_i Y^k − Y^i ∂_i X^k
  def bracket_sym(Xv, Yv):
    out = [0, 0]
    for k in range(2):
      term = 0
      for i, si in enumerate((x, y)):
        term += Xv[i]*sympy.diff(Yv[k], si) - Yv[i]*sympy.diff(Xv[k], si)
      out[k] = sympy.simplify(term)
    return out

  B = bracket_sym(X_sym, Y_sym)
  B_grad = [[sympy.diff(B[k], a) for a in (x, y)] for k in range(2)]
  B_hess = [[[sympy.diff(B[k], a, b) for b in (x, y)] for a in (x, y)] for k in range(2)]

  # Evaluation point
  xv, yv = 0.3, -0.4
  def eval_expr(e):
    f = sympy.lambdify((x, y), e, "numpy")
    return float(f(xv, yv))

  B_val_np = np.array([eval_expr(Bk) for Bk in B])
  B_grad_np = np.array([[eval_expr(B_grad[k][a]) for a in range(2)] for k in range(2)])
  B_hess_np = np.array([[[eval_expr(B_hess[k][a][b]) for b in range(2)] for a in range(2)] for k in range(2)])

  # Now build Jets via function_to_jet and compare
  p = jnp.array([xv, yv])
  basis = get_standard_basis(p)

  def X_func(pt):
    xj, yj = pt
    return jnp.array([xj**2 + xj*yj + 1.0, xj*yj + yj**2 + 2.0])

  def Y_func(pt):
    xj, yj = pt
    return jnp.array([xj + yj**2, xj**2 - yj + 3.0])

  X_tv = TangentVector(p=p, basis=basis, components=function_to_jet(X_func, p))
  Y_tv = TangentVector(p=p, basis=basis, components=function_to_jet(Y_func, p))

  B_tv = lie_bracket(X_tv, Y_tv)

  np.testing.assert_allclose(B_tv.components.value, B_val_np, rtol=1e-8, atol=1e-8)
  np.testing.assert_allclose(B_tv.components.gradient, B_grad_np, rtol=1e-8, atol=1e-8)
  # np.testing.assert_allclose(B_tv.components.hessian, B_hess_np, rtol=1e-8, atol=1e-8) # Can't check hessian because X and Y have no third derivatives


def test_lie_bracket_matches_sympy_in_nonholonomic_basis():
  # Non-holonomic frame in R^2: E1 = ∂_x, E2 = ∂_y + x ∂_x
  # Define polynomial vector fields X, Y in coordinate components, then express
  # them in the frame and verify bracket jets (after mapping back to standard)
  # match SymPy ground truth.
  x, y = symbols("x, y")
  X_sym = [x**2 + x*y + 1, x*y + y**2 + 2]
  Y_sym = [x + y**2, x**2 - y + 3]

  def bracket_sym(Xv, Yv):
    out = [0, 0]
    for k in range(2):
      term = 0
      for i, si in enumerate((x, y)):
        term += Xv[i]*sympy.diff(Yv[k], si) - Yv[i]*sympy.diff(Xv[k], si)
      out[k] = sympy.simplify(term)
    return out

  B = bracket_sym(X_sym, Y_sym)
  B_grad = [[sympy.diff(B[k], a) for a in (x, y)] for k in range(2)]
  B_hess = [[[sympy.diff(B[k], a, b) for b in (x, y)] for a in (x, y)] for k in range(2)]

  # Evaluation point
  xv, yv = 0.1, -0.7
  def eval_expr(e):
    f = sympy.lambdify((x, y), e, "numpy")
    return float(f(xv, yv))

  B_val_np = np.array([eval_expr(Bk) for Bk in B])
  B_grad_np = np.array([[eval_expr(B_grad[k][a]) for a in range(2)] for k in range(2)])
  B_hess_np = np.array([[[eval_expr(B_hess[k][a][b]) for b in range(2)] for a in range(2)] for k in range(2)])

  # Build non-holonomic basis A(x) = [[1, x],[0,1]]
  p = jnp.array([xv, yv])
  def A_func(pt):
    xj, yj = pt
    return jnp.array([[1.0, xj],[0.0, 1.0]])

  basis_frame = BasisVectors(p=p, components=function_to_jet(A_func, p))
  basis_std = get_standard_basis(p)

  # Vector fields as Jets in coordinate basis
  def X_func(pt):
    xj, yj = pt
    return jnp.array([xj**2 + xj*yj + 1.0, xj*yj + yj**2 + 2.0])

  def Y_func(pt):
    xj, yj = pt
    return jnp.array([xj + yj**2, xj**2 - yj + 3.0])

  X_std = TangentVector(p=p, basis=basis_std, components=function_to_jet(X_func, p))
  Y_std = TangentVector(p=p, basis=basis_std, components=function_to_jet(Y_func, p))

  # Express X, Y in the non-holonomic frame
  X_fr = change_basis(X_std, basis_frame)
  Y_fr = change_basis(Y_std, basis_frame)

  # Compute bracket in our implementation (result in frame basis), then map back to standard
  B_fr = lie_bracket(X_fr, Y_fr)
  B_std = change_basis(B_fr, basis_std)

  np.testing.assert_allclose(B_std.components.value, B_val_np, rtol=1e-8, atol=1e-8)
  np.testing.assert_allclose(B_std.components.gradient, B_grad_np, rtol=1e-8, atol=1e-8)
  # np.testing.assert_allclose(B_std.components.hessian, B_hess_np, rtol=1e-8, atol=1e-8) # Can't check hessian because X and Y have no third derivatives

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
  dF(X)(f) = X(f\\circ F)
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


def create_random_basis(key: random.PRNGKey, dim: int) -> BasisVectors:
  p_key, vals_key, grads_key, hessians_key = random.split(key, 4)
  p = jnp.zeros(dim)
  vals = random.normal(vals_key, (dim, dim))
  grads = random.normal(grads_key, (dim, dim, dim))
  hessians = random.normal(hessians_key, (dim, dim, dim, dim))
  return BasisVectors(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians))


def create_random_vector_field(key: random.PRNGKey, dim: int) -> TangentVector:
  p_key, basis_key, vals_key, grads_key, hessians_key = random.split(key, 5)
  p = random.normal(p_key, (dim,))
  random_basis = create_random_basis(basis_key, dim)
  vals = random.normal(vals_key, (dim,))
  grads = random.normal(grads_key, (dim, dim))
  hessians = random.normal(hessians_key, (dim, dim, dim))
  return TangentVector(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians), basis=random_basis)

def test_derivation():
  """
  X(fg) = X(f)g + fX(g)
  """
  key = random.key(0)
  k1, k2, k3 = random.split(key, 3)
  dim = 2
  X = create_random_vector_field(k1, dim)
  X = change_basis(X, get_standard_basis(X.p))

  def f(x):
    return x.sum()

  def g(x):
    return (x**2).sum()

  def fg(x):
    return (x.sum())*(x**2).sum()

  f_jet = function_to_jet(f, X.p)
  g_jet = function_to_jet(g, X.p)

  @jet_decorator
  def multiply_functions(f_val, g_val):
    return f_val*g_val

  fg_jet = multiply_functions(f_jet, g_jet)
  fg_jet2 = function_to_jet(fg, X.p)

  assert jnp.allclose(fg_jet.value, fg_jet2.value)
  assert jnp.allclose(fg_jet.gradient, fg_jet2.gradient)
  assert jnp.allclose(fg_jet.hessian, fg_jet2.hessian)

  Xfg_jet = X(fg_jet)
  Xf: Jet = X(f_jet)
  Xg: Jet = X(g_jet)
  Xfg = multiply_functions(Xf, g_jet)
  fXg = multiply_functions(f_jet, Xg)

  lhs = Xfg_jet
  rhs = Xfg + fXg
  assert jnp.allclose(lhs.value, rhs.value)
  assert jnp.allclose(lhs.gradient, rhs.gradient)
  # assert jnp.allclose(lhs.hessian, rhs.hessian) # Can't check hessian because X(f) has no hessian

def test_lie_bracket_definition():
  """
  [X, Y](f) = X(Y(f)) - Y(X(f))
  """
  key = random.key(0)
  k1, k2, k3 = random.split(key, 3)
  dim = 2
  X = create_random_vector_field(k1, dim)
  Y = create_random_vector_field(k2, dim)
  Y = change_basis(Y, X.basis)

  lb_XY = lie_bracket(X, Y)

  def f(x):
    x = x + 1.0
    return jnp.array([
      x[0]**2 + 3.0 * x[1],
      jnp.sin(x[0]) + x[1]**3,
    ]).sum()
  f_jet = function_to_jet(f, X.p)

  lhs: Jet = lb_XY(f_jet)
  rhs1 = X(Y(f_jet))
  rhs2 = Y(X(f_jet))
  rhs = rhs1 - rhs2

  assert jnp.allclose(lhs.value, rhs.value)
  # assert jnp.allclose(lhs.gradient, rhs.gradient) # Don't have enough information to check gradient
  # assert jnp.allclose(lhs.hessian, rhs.hessian) # Don't have enough information to check hessian

def test_lie_bracket_identities():
  """
  [fX, gY] = fg[X, Y] + fX(g)Y - gY(f)X
  """
  key = random.key(0)
  k1, k2, k3 = random.split(key, 3)
  dim = 2
  X = create_random_vector_field(k1, dim)
  Y = create_random_vector_field(k2, dim)
  Y = change_basis(Y, X.basis)

  def f(x):
    x = x + 1.0
    return jnp.array([
      x[0]**2 + 3.0 * x[1],
      jnp.sin(x[0]) + x[1]**3,
    ]).sum()

  def g(x):
    return f(x)**2 + x.sum()**2

  f_jet = function_to_jet(f, X.p)
  g_jet = function_to_jet(g, X.p)

  lb_XY = lie_bracket(X, Y)

  @jet_decorator
  def multiply_function_and_vector(f_val, X_val):
    return f_val*X_val

  fX_components: Jet = multiply_function_and_vector(f_jet, X.components)
  fX = TangentVector(p=X.p, components=fX_components, basis=X.basis)

  gY_components: Jet = multiply_function_and_vector(g_jet, Y.components)
  gY = TangentVector(p=Y.p, components=gY_components, basis=Y.basis)

  Xg: Jet = X(g_jet)
  Yf: Jet = Y(f_jet)

  @jet_decorator
  def multiply_multiple_functions_and_vectors(f_val, g_val, X_val):
    return f_val*g_val*X_val

  fg_lb_XY_components: Jet = multiply_multiple_functions_and_vectors(f_jet, g_jet, lb_XY.components)
  fg_lb_XY = TangentVector(p=X.p, components=fg_lb_XY_components, basis=X.basis)

  fXg_Y_components: Jet = multiply_multiple_functions_and_vectors(f_jet, Xg, Y.components)
  fXg_Y = TangentVector(p=X.p, components=fXg_Y_components, basis=X.basis)

  gYf_X_components: Jet = multiply_multiple_functions_and_vectors(g_jet, Yf, X.components)
  gYf_X = TangentVector(p=X.p, components=gYf_X_components, basis=X.basis)


  lhs = lie_bracket(fX, gY)
  rhs = fg_lb_XY + fXg_Y - gYf_X
  assert jnp.allclose(lhs.components.value, rhs.components.value)
  assert jnp.allclose(lhs.components.gradient, rhs.components.gradient)
  # assert jnp.allclose(lhs.components.hessian, rhs.components.hessian) # Don't have enough information to check hessian

def test_tangent_vector_change_coordinates_round_trip():
  """
  Test that changing coordinates forward and backward preserves the TangentVector.
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

  comp_val = random.normal(key, (dim,))
  comp_grad = random.normal(key, (dim, dim))
  comp_hess = random.normal(key, (dim, dim, dim))
  vec_comp = Jet(value=comp_val, gradient=comp_grad, hessian=comp_hess, dim=dim)

  vec_q = TangentVector(p=q, basis=basis, components=vec_comp)

  J_zq = function_to_jacobian(spherical_to_cartesian, q)
  vec_x = change_coordinates(vec_q, J_zq)

  J_xz = function_to_jacobian(cartesian_to_spherical, x)
  vec_q_restored = change_coordinates(vec_x, J_xz)

  assert jnp.allclose(vec_q_restored.basis.components.value, vec_q.basis.components.value, atol=1e-5)
  assert jnp.allclose(vec_q_restored.basis.components.gradient, vec_q.basis.components.gradient, atol=1e-5)
  assert jnp.allclose(vec_q_restored.basis.components.hessian, vec_q.basis.components.hessian, atol=1e-5)

  assert jnp.allclose(vec_q_restored.components.value, vec_q.components.value, atol=1e-5)
  assert jnp.allclose(vec_q_restored.components.gradient, vec_q.components.gradient, atol=1e-5)
  assert jnp.allclose(vec_q_restored.components.hessian, vec_q.components.hessian, atol=1e-5)

def test_tangent_vector_change_coordinates_vector_consistency():
  """
  Check that evaluating a tangent vector gives the same physical vector
  before and after coordinate change.

  With convention E[a,j] = E_j^a (column j = basis vector j):
    Physical vector = E @ V where V is the component vector.

  After coordinate change, the basis transforms as E_new = G @ E,
  and components stay the same (treated as scalars in the current implementation).
  So the new physical vector is E_new @ V = G @ E @ V = G @ (original physical).
  """
  q = jnp.array([2.0, jnp.pi/4, jnp.pi/4])
  x = spherical_to_cartesian(q)
  dim = 3

  basis_q = get_standard_basis(q)

  key = random.key(2)
  comp_val = random.normal(key, (dim,))
  vec_comp = Jet(value=comp_val, gradient=None, hessian=None, dim=dim)

  vec_q = TangentVector(p=q, basis=basis_q, components=vec_comp)

  J_zq = function_to_jacobian(spherical_to_cartesian, q)
  vec_x = change_coordinates(vec_q, J_zq)

  J_val = J_zq.value # G = dx/dq

  # Physical vector = basis @ components (columns are basis vectors)
  v_q_phys = basis_q.components.value @ vec_q.components.value
  v_x_phys = vec_x.basis.components.value @ vec_x.components.value

  # Check geometric consistency: v_x_phys = G @ v_q_phys
  assert jnp.allclose(v_x_phys, J_val @ v_q_phys, atol=1e-5)

def test_tangent_vector_change_coordinates_scalar_components():
  """
  Verify that TangentVector components transform as scalars (invariants)
  but their derivatives change via chain rule.
  """
  def shift_map(q):
    return q + jnp.ones_like(q)

  q = jnp.array([1.0, 2.0])

  def comp_func(q):
    return q**2

  comp_jet = function_to_jet(comp_func, q)
  dim = 2
  basis = get_standard_basis(q)

  vec_q = TangentVector(p=q, basis=basis, components=comp_jet)

  J_zq = function_to_jacobian(shift_map, q)
  vec_x = change_coordinates(vec_q, J_zq)

  assert jnp.allclose(vec_x.components.value, comp_jet.value)
  assert jnp.allclose(vec_x.components.gradient, comp_jet.gradient)
  assert jnp.allclose(vec_x.components.hessian, comp_jet.hessian)


def test_tangent_vector_call_coordinate_invariance():
  """
  Check that applying a tangent vector to a scalar function (Jet) yields
  the same geometric directional derivative, regardless of coordinate chart.
  """
  dim = 3
  key = random.key(0)

  # Use a fixed spherical point (avoid issues with random points near singularities)
  q = jnp.array([1.5, 0.9, 1.2])
  x = spherical_to_cartesian(q)

  # Create random basis and vector AT THE SAME POINT
  k1, k2, k3, k4, k5, k6 = random.split(key, 6)
  basis_vals = random.normal(k1, (dim, dim))
  basis_grads = random.normal(k2, (dim, dim, dim))
  basis_hess = random.normal(k3, (dim, dim, dim, dim))
  random_basis = BasisVectors(p=q, components=Jet(value=basis_vals, gradient=basis_grads, hessian=basis_hess))

  comp_vals = random.normal(k4, (dim,))
  comp_grads = random.normal(k5, (dim, dim))
  comp_hess = random.normal(k6, (dim, dim, dim))
  X_q = TangentVector(p=q, components=Jet(value=comp_vals, gradient=comp_grads, hessian=comp_hess), basis=random_basis)

  # Scalar function f and its Jet at q
  def f(x_):
    return x_[0]**2 + 3.0 * x_[1] - 0.7 * x_[2]

  f_jet_q = function_to_jet(f, q)

  # Evaluate X_q(f) in q-coordinates
  out_q: Jet = X_q(f_jet_q)

  # Coordinate change q -> x
  J_zq = function_to_jacobian(spherical_to_cartesian, q)
  X_x_temp: TangentVector = change_coordinates(X_q, J_zq)

  # Manually update point to x-coordinates
  X_x = TangentVector(
    p=x,
    basis=BasisVectors(p=x, components=X_x_temp.basis.components),
    components=X_x_temp.components
  )

  # Transform f to x-coordinates
  f_jet_x: Jet = change_coordinates(f_jet_q, J_zq)

  # Evaluate X_x(f) in x-coordinates
  out_x: Jet = X_x(f_jet_x)

  # Transform result back to q-coordinates and compare full Jets
  J_xz = function_to_jacobian(cartesian_to_spherical, x)
  out_x_q: Jet = change_coordinates(out_x, J_xz)

  assert jnp.allclose(out_x_q.value, out_q.value)
  assert jnp.allclose(out_x_q.gradient, out_q.gradient)
  # Note: Hessian comparison is skipped because computing d^2(X(f))/dx^2
  # requires third derivatives of f, which function_to_jet doesn't provide.
  # The inf-fill for unknown hessians propagates through and produces NaN.
