import jax.numpy as jnp
from jax import random
import numpy as np
from local_coordinates.basis import BasisVectors, get_standard_basis, change_basis
from local_coordinates.connection import Connection, get_levi_civita_connection
from local_coordinates.jet import Jet
from local_coordinates.metric import RiemannianMetric
from local_coordinates.tensor import Tensor, TensorType
from local_coordinates.tangent import TangentVector
from local_coordinates.frame import Frame


def create_random_basis(key: random.PRNGKey, dim: int) -> BasisVectors:
  p_key, vals_key, grads_key, hessians_key = random.split(key, 4)
  p = random.normal(p_key, (dim,))
  vals = random.normal(vals_key, (dim, dim))
  grads = random.normal(grads_key, (dim, dim, dim))
  hessians = random.normal(hessians_key, (dim, dim, dim, dim))
  return BasisVectors(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians))


def test_covariant_derivative_uses_frame_directional_derivative_in_nonholonomic_basis():
  # Construct a simple non-holonomic frame in R^2: E_1 = ∂_x, E_2 = ∂_y + x ∂_x
  # Then for a vector field Y with components Y^k(x,y), E_2(Y^k) = ∂_y Y^k + x ∂_x Y^k.
  p = jnp.array([0.3, -0.5])
  d = 2
  # Basis components E^a_i stacked as (a,i)
  E_val = jnp.array([[1.0, 1.0 * p[0]],   # a=0 (x): E_1^x = 1, E_2^x = x
                     [0.0, 1.0]])         # a=1 (y): E_1^y = 0, E_2^y = 1
  E_grad = jnp.zeros((d, d, d))
  # ∂_x E_2^x = 1, all others 0
  E_grad = E_grad.at[0, 1, 0].set(1.0)
  basis = BasisVectors(p=p, components=Jet(value=E_val, gradient=E_grad, hessian=jnp.zeros((d, d, d, d))))

  # Zero connection
  Gamma = jnp.zeros((d, d, d))
  conn = Connection(basis=basis, christoffel_symbols=Jet(value=Gamma, gradient=None, hessian=None, dim=d))

  # Choose X = E_2 (components (0,1) in the frame)
  X = TangentVector(p=p, basis=basis, components=Jet(value=jnp.array([0.0, 1.0]), gradient=None, hessian=None, dim=d))

  # Define a vector field Y with prescribed partials at p
  # Y^x(x,y) = ax*x + ay*y + c  ⇒ ∂_x Y^x = ax, ∂_y Y^x = ay
  # Y^y(x,y) = bx*x + by*y + d  ⇒ ∂_x Y^y = bx, ∂_y Y^y = by
  ax, ay = 0.7, -0.2
  bx, by = 0.4, 0.5
  Y_val = jnp.array([1.2, -0.8])
  Y_grad = jnp.array([[ax, ay],
                      [bx, by]])
  Y = TangentVector(p=p, basis=basis, components=Jet(value=Y_val, gradient=Y_grad, hessian=None))

  out = conn.covariant_derivative(X, Y)

  # Expected: (∇_{E2} Y)^k = E_2(Y^k) since Γ=0 and basis derivatives handled by E^a_i ∂_a
  # E_2(Y^k) = ∂_y Y^k + x ∂_x Y^k evaluated at p[0] = x
  x = p[0]
  expected = jnp.array([
    ay + x * ax,
    by + x * bx,
  ])
  assert jnp.allclose(out.components.value, expected)


def test_connection_basic_construction():
  p = jnp.array([0.0, 0.0])
  basis = get_standard_basis(p)

  Gamma = jnp.zeros((2, 2, 2))
  Gamma_jet = Jet(value=Gamma, gradient=None, hessian=None, dim=2)

  conn = Connection(basis=basis, christoffel_symbols=Gamma_jet)

  assert conn.basis is basis
  assert conn.christoffel_symbols.shape == (2, 2, 2)
  assert jnp.allclose(conn.christoffel_symbols.value, 0.0)


def test_get_levi_civita_connection_diagonal_metric_depends_on_u():
  # Metric G = diag(a(u), b(u)) with a,b > 0 and only u-dependence
  # Expected (see [Khudian, HW6 Sols](https://khudian.net/Teaching/Geometry/GeomRim18/solutions6a.pdf)):
  # Γ^u_{uu} = a_u/(2a), Γ^u_{vv} = - b_u/(2a), Γ^v_{uv} = Γ^v_{vu} = b_u/(2b);
  # all other symbols vanish when a_v = b_v = 0.
  u = 0.3
  v = -0.7
  p = jnp.array([u, v])
  basis = get_standard_basis(p)

  def a(u):
    return jnp.exp(u)  # a(u) > 0

  def b(u):
    return 1.0 + u*u   # b(u) > 0

  a_val = a(u)
  b_val = b(u)
  a_u = jnp.exp(u)
  b_u = 2.0*u

  # Metric value and gradient g_{ij,k}
  g_val = jnp.array([[a_val, 0.0], [0.0, b_val]])
  g_grad = jnp.zeros((2, 2, 2))
  g_grad = g_grad.at[0, 0, 0].set(a_u)  # ∂_u g_{uu}
  g_grad = g_grad.at[1, 1, 0].set(b_u)  # ∂_u g_{vv}

  metric_components = Jet(value=g_val, gradient=g_grad, hessian=jnp.zeros((2, 2, 2, 2)))
  metric = RiemannianMetric(basis=basis, components=metric_components)

  connection = get_levi_civita_connection(metric)
  gamma = connection.christoffel_symbols.value

  expected = jnp.zeros((2, 2, 2))
  expected = expected.at[0, 0, 0].set(a_u / (2.0 * a_val))           # Γ^u_{uu}
  expected = expected.at[0, 1, 1].set(- b_u / (2.0 * a_val))         # Γ^u_{vv}
  expected = expected.at[1, 0, 1].set(b_u / (2.0 * b_val))           # Γ^v_{uv}
  expected = expected.at[1, 1, 0].set(b_u / (2.0 * b_val))           # Γ^v_{vu}

  assert jnp.allclose(gamma, expected, atol=1e-6)


def test_get_levi_civita_connection_polar_plane():
  # Euclidean plane in polar coords: G = diag(1, r^2)
  # Expected nonzero symbols (see [Khudian, HW6 Sols](https://khudian.net/Teaching/Geometry/GeomRim18/solutions6a.pdf)):
  # Γ^r_{φφ} = -r, Γ^φ_{rφ} = Γ^φ_{φr} = 1/r.
  r = 2.5
  phi = 1.2
  p = jnp.array([r, phi])
  basis = get_standard_basis(p)

  g_val = jnp.array([[1.0, 0.0], [0.0, r*r]])
  g_grad = jnp.zeros((2, 2, 2))
  g_grad = g_grad.at[1, 1, 0].set(2.0 * r)  # ∂_r g_{φφ} = 2r

  metric_components = Jet(value=g_val, gradient=g_grad, hessian=jnp.zeros((2, 2, 2, 2)))
  metric = RiemannianMetric(basis=basis, components=metric_components)

  connection = get_levi_civita_connection(metric)
  gamma = connection.christoffel_symbols.value

  expected = jnp.zeros((2, 2, 2))
  expected = expected.at[0, 1, 1].set(-r)     # Γ^r_{φφ}
  expected = expected.at[1, 0, 1].set(1.0/r)  # Γ^φ_{rφ}
  expected = expected.at[1, 1, 0].set(1.0/r)  # Γ^φ_{φr}

  assert jnp.allclose(gamma, expected, atol=1e-6)


def test_get_levi_civita_connection_diagonal_metric_uv():
  # Metric G = diag(a(u,v), b(u,v))
  # Expected (see Khudian HW6 solutions):
  # Γ^u_{uu} = a_u/(2a), Γ^u_{uv} = Γ^u_{vu} = a_v/(2a), Γ^u_{vv} = - b_u/(2a)
  # Γ^v_{uu} = - a_v/(2b), Γ^v_{uv} = Γ^v_{vu} = b_u/(2b), Γ^v_{vv} = b_v/(2b)
  u = 0.2
  v = -0.4
  p = jnp.array([u, v])
  basis = get_standard_basis(p)

  def a(u, v):
    return jnp.exp(u + 2.0*v)

  def b(u, v):
    return 1.0 + u*u + v

  a_val = a(u, v)
  b_val = b(u, v)
  a_u = jnp.exp(u + 2.0*v)
  a_v = 2.0 * jnp.exp(u + 2.0*v)
  b_u = 2.0 * u
  b_v = 1.0

  g_val = jnp.array([[a_val, 0.0], [0.0, b_val]])
  g_grad = jnp.zeros((2, 2, 2))
  # ∂_u terms (k=0)
  g_grad = g_grad.at[0, 0, 0].set(a_u)
  g_grad = g_grad.at[1, 1, 0].set(b_u)
  # ∂_v terms (k=1)
  g_grad = g_grad.at[0, 0, 1].set(a_v)
  g_grad = g_grad.at[1, 1, 1].set(b_v)

  metric_components = Jet(value=g_val, gradient=g_grad, hessian=jnp.zeros((2, 2, 2, 2)))
  metric = RiemannianMetric(basis=basis, components=metric_components)

  connection = get_levi_civita_connection(metric)
  gamma = connection.christoffel_symbols.value

  expected = jnp.zeros((2, 2, 2))
  expected = expected.at[0, 0, 0].set(a_u / (2.0 * a_val))      # Γ^u_{uu}
  expected = expected.at[0, 0, 1].set(a_v / (2.0 * a_val))      # Γ^u_{uv}
  expected = expected.at[0, 1, 0].set(a_v / (2.0 * a_val))      # Γ^u_{vu}
  expected = expected.at[0, 1, 1].set(- b_u / (2.0 * a_val))    # Γ^u_{vv}
  expected = expected.at[1, 0, 0].set(- a_v / (2.0 * b_val))    # Γ^v_{uu}
  expected = expected.at[1, 0, 1].set(b_u / (2.0 * b_val))      # Γ^v_{uv}
  expected = expected.at[1, 1, 0].set(b_u / (2.0 * b_val))      # Γ^v_{vu}
  expected = expected.at[1, 1, 1].set(b_v / (2.0 * b_val))      # Γ^v_{vv}

  assert jnp.allclose(gamma, expected, atol=1e-6)


def test_get_levi_civita_connection_hyperbolic_half_plane():
  # Lobachevsky (upper half-plane) metric: G = (dx^2 + dy^2) / y^2
  # Coordinates (x, y). Expected symbols:
  # Γ^x_{xy} = Γ^x_{yx} = -1/y, Γ^y_{xx} = 1/y, Γ^y_{yy} = -1/y
  x = 0.7
  y = 1.3
  p = jnp.array([x, y])
  basis = get_standard_basis(p)

  g_val = jnp.array([[1.0 / (y*y), 0.0], [0.0, 1.0 / (y*y)]])
  g_grad = jnp.zeros((2, 2, 2))
  # ∂_x g = 0
  # ∂_y g_{xx} = ∂_y g_{yy} = -2 / y^3
  dgy = -2.0 / (y*y*y)
  g_grad = g_grad.at[0, 0, 1].set(dgy)
  g_grad = g_grad.at[1, 1, 1].set(dgy)

  metric_components = Jet(value=g_val, gradient=g_grad, hessian=jnp.zeros((2, 2, 2, 2)))
  metric = RiemannianMetric(basis=basis, components=metric_components)

  connection = get_levi_civita_connection(metric)
  gamma = connection.christoffel_symbols.value

  expected = jnp.zeros((2, 2, 2))
  expected = expected.at[0, 0, 1].set(-1.0 / y)   # Γ^x_{x y}
  expected = expected.at[0, 1, 0].set(-1.0 / y)   # Γ^x_{y x}
  expected = expected.at[1, 0, 0].set(1.0 / y)    # Γ^y_{x x}
  expected = expected.at[1, 1, 1].set(-1.0 / y)   # Γ^y_{y y}

  assert jnp.allclose(gamma, expected, atol=1e-6)


def test_covariant_derivative_zero_connection_matches_directional_derivative():
  # Basis: standard Cartesian; Γ = 0
  p = jnp.array([0.0, 0.0])
  basis = get_standard_basis(p)

  Gamma = jnp.zeros((2, 2, 2))
  conn = Connection(basis=basis, christoffel_symbols=Jet(value=Gamma, gradient=None, hessian=None, dim=2))

  # X components (vector)
  X_val = jnp.array([1.2, -0.7])
  X = Jet(value=X_val, gradient=None, hessian=None, dim=2)
  X_tensor = TangentVector(p=p, basis=basis, components=X)

  # Y components and gradient (∂_i Y^k)
  Y_val = jnp.array([2.0, -3.0])
  Y_grad = jnp.array([[0.5, 1.0],   # ∂_x Y^x, ∂_y Y^x
                      [2.0, 0.25]]) # ∂_x Y^y, ∂_y Y^y
  Y = Jet(value=Y_val, gradient=Y_grad, hessian=None)
  Y_tensor = TangentVector(p=p, basis=basis, components=Y)

  out = Connection(basis=basis, christoffel_symbols=Jet(value=Gamma, gradient=None, hessian=None, dim=2)).covariant_derivative(X_tensor, Y_tensor)
  expected = jnp.einsum("i,ki->k", X_val, Y_grad)
  assert jnp.allclose(out.components.value, expected)


def test_covariant_derivative_with_connection_matches_formula():
  # Basis: standard; simple nonzero Γ
  p = jnp.array([0.0, 0.0])
  basis = get_standard_basis(p)

  Gamma = jnp.zeros((2, 2, 2))
  Gamma = Gamma.at[0, 0, 0].set(2.0)  # Γ^x_{xx} = 2
  Gamma = Gamma.at[1, 0, 1].set(-1.0) # Γ^y_{x y} = -1
  conn = Connection(basis=basis, christoffel_symbols=Jet(value=Gamma, gradient=None, hessian=None, dim=2))

  X_val = jnp.array([0.3, -1.1])
  X_tensor = TangentVector(p=p, basis=basis, components=Jet(value=X_val, gradient=None, hessian=None, dim=2))

  Y_val = jnp.array([1.4, 0.6])
  Y_grad = jnp.array([[0.2, 0.1],
                      [0.0, -0.4]])
  Y_tensor = TangentVector(p=p, basis=basis, components=Jet(value=Y_val, gradient=Y_grad, hessian=None))

  out = Connection(basis=basis, christoffel_symbols=Jet(value=Gamma, gradient=None, hessian=None, dim=2)).covariant_derivative(X_tensor, Y_tensor)
  term1 = jnp.einsum("i,ki->k", X_val, Y_grad)
  term2 = jnp.einsum("kij,i,j->k", Gamma, X_val, Y_val)
  expected = term1 + term2
  assert jnp.allclose(out.components.value, expected)


def test_covariant_derivative_polar_plane_lc_connection():
  # Polar plane G = diag(1, r^2)
  r = 2.0
  phi = 0.3
  p = jnp.array([r, phi])
  basis = get_standard_basis(p)

  g_val = jnp.array([[1.0, 0.0], [0.0, r*r]])
  g_grad = jnp.zeros((2, 2, 2))
  g_grad = g_grad.at[1, 1, 0].set(2.0 * r)
  metric = RiemannianMetric(basis=basis, components=Jet(value=g_val, gradient=g_grad, hessian=jnp.zeros((2,2,2,2))))

  # Ground-truth Christoffel symbols for polar plane (r, φ):
  Gamma = jnp.zeros((2, 2, 2))
  Gamma = Gamma.at[0, 1, 1].set(-r)     # Γ^r_{φφ} = -r
  Gamma = Gamma.at[1, 0, 1].set(1.0/r)  # Γ^φ_{rφ} = 1/r
  Gamma = Gamma.at[1, 1, 0].set(1.0/r)  # Γ^φ_{φr} = 1/r

  # Choose X, Y
  X_val = jnp.array([1.0, 0.5])
  Y_val = jnp.array([0.7, -1.2])
  # Supply Y gradient (∂_i Y^k); arbitrary but fixed
  Y_grad = jnp.array([[0.3, -0.2],
                      [0.4,  0.1]])
  X_tensor = TangentVector(p=p, basis=basis, components=Jet(value=X_val, gradient=None, hessian=None, dim=2))
  Y_tensor = TangentVector(p=p, basis=basis, components=Jet(value=Y_val, gradient=Y_grad, hessian=None))

  out = Connection(basis=basis, christoffel_symbols=Jet(value=Gamma, gradient=None, hessian=None, dim=2)).covariant_derivative(X_tensor, Y_tensor)
  expected = jnp.einsum("i,ki->k", X_val, Y_grad) + jnp.einsum("kij,i,j->k", Gamma, X_val, Y_val)
  assert jnp.allclose(out.components.value, expected)


def test_covariant_derivative_hyperbolic_half_plane_ground_truth():
  # Upper half-plane metric: G = (dx^2 + dy^2)/y^2 with coords (x,y)
  x = 0.4
  y = 1.7
  p = jnp.array([x, y])
  basis = get_standard_basis(p)

  # Build a Levi-Civita connection for the metric (used only to evaluate ∇, not for Γ ground truth)
  g_val = jnp.array([[1.0/(y*y), 0.0], [0.0, 1.0/(y*y)]])
  g_grad = jnp.zeros((2, 2, 2))
  dgy = -2.0/(y*y*y)
  g_grad = g_grad.at[0, 0, 1].set(dgy)
  g_grad = g_grad.at[1, 1, 1].set(dgy)
  metric = RiemannianMetric(basis=basis, components=Jet(value=g_val, gradient=g_grad, hessian=jnp.zeros((2,2,2,2))))
  # Analytic Christoffels: Γ^x_{xy}=Γ^x_{yx}=-1/y; Γ^y_{xx}=1/y; Γ^y_{yy}=-1/y
  Gamma = jnp.zeros((2, 2, 2))
  Gamma = Gamma.at[0, 0, 1].set(-1.0/y)
  Gamma = Gamma.at[0, 1, 0].set(-1.0/y)
  Gamma = Gamma.at[1, 0, 0].set(1.0/y)
  Gamma = Gamma.at[1, 1, 1].set(-1.0/y)

  # Choose X, Y and Y_grad
  X_val = jnp.array([0.8, -0.3])
  Y_val = jnp.array([1.1, -0.5])
  Y_grad = jnp.array([[0.2, 0.0],
                      [0.1, -0.15]])
  X_tensor = TangentVector(p=p, basis=basis, components=Jet(value=X_val, gradient=None, hessian=None, dim=2))
  Y_tensor = TangentVector(p=p, basis=basis, components=Jet(value=Y_val, gradient=Y_grad, hessian=None))

  out = Connection(basis=basis, christoffel_symbols=Jet(value=Gamma, gradient=None, hessian=None, dim=2)).covariant_derivative(X_tensor, Y_tensor)
  expected = jnp.einsum("i,ki->k", X_val, Y_grad) + jnp.einsum("kij,i,j->k", Gamma, X_val, Y_val)
  assert jnp.allclose(out.components.value, expected)


def test_get_levi_civita_connection_euclidean_cartesian_zero():
  p = jnp.array([0.3, -1.2])
  basis = get_standard_basis(p)

  # Euclidean metric in Cartesian coordinates: identity with zero derivatives
  metric_components = Jet(
      value=jnp.eye(2),
      gradient=jnp.zeros((2, 2, 2)),
    hessian=jnp.zeros((2, 2, 2, 2)),
  )
  metric = RiemannianMetric(basis=basis, components=metric_components)

  conn = get_levi_civita_connection(metric)

  # Compare to known zero connection in this chart
  assert conn.christoffel_symbols.shape == (2, 2, 2)
  assert jnp.allclose(conn.christoffel_symbols.value, 0.0)


def test_connection_change_basis_round_trip():
    """
    Tests that changing basis to a new basis and back to the original
    recovers the original Christoffel symbols. This will fail if the
    transformation rule is implemented incorrectly.
    """
    key = random.PRNGKey(123)
    dim = 3

    # 1. Create a connection with random Christoffel symbols in a random basis
    basis_key, gamma_key = random.split(key)
    basis_a = create_random_basis(basis_key, dim)

    gamma_val_key, gamma_grad_key, gamma_hess_key = random.split(gamma_key, 3)
    gamma_val = random.normal(gamma_val_key, (dim, dim, dim))
    gamma_grad = random.normal(gamma_grad_key, (dim, dim, dim, dim))
    gamma_hess = random.normal(gamma_hess_key, (dim, dim, dim, dim, dim))
    gamma_jet_a = Jet(value=gamma_val, gradient=gamma_grad, hessian=gamma_hess)

    conn_a = Connection(basis=basis_a, christoffel_symbols=gamma_jet_a)

    # 2. Get the standard basis
    basis_std = get_standard_basis(basis_a.p)

    # 3. Perform the round trip transformation
    conn_std = change_basis(conn_a, basis_std)
    conn_a_round_trip = change_basis(conn_std, basis_a)

    # 4. Check if the original and round-trip symbols are the same
    np.testing.assert_allclose(
        conn_a.christoffel_symbols.value,
        conn_a_round_trip.christoffel_symbols.value,
        rtol=1e-5
    )