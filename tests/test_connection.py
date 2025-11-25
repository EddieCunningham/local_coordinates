import jax.numpy as jnp
from jax import random
import numpy as np
import pytest
from jaxtyping import Array
from local_coordinates.basis import BasisVectors, get_standard_basis, change_basis
from local_coordinates.connection import Connection, get_levi_civita_connection, change_coordinates
from local_coordinates.jet import Jet, function_to_jet, jet_decorator
from local_coordinates.metric import RiemannianMetric
from local_coordinates.tensor import Tensor, TensorType
from local_coordinates.tangent import TangentVector, lie_bracket
from local_coordinates.frame import Frame
from local_coordinates.jacobian import function_to_jacobian, Jacobian, get_inverse


def create_random_basis(key: random.PRNGKey, dim: int) -> BasisVectors:
  p_key, vals_key, grads_key, hessians_key = random.split(key, 4)
  p = jnp.zeros(dim)
  vals = random.normal(vals_key, (dim, dim))
  grads = random.normal(grads_key, (dim, dim, dim))
  hessians = random.normal(hessians_key, (dim, dim, dim, dim))
  return BasisVectors(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians))

def test_connection_basic_construction():
  p = jnp.array([0.0, 0.0])
  basis = get_standard_basis(p)

  Gamma = jnp.zeros((2, 2, 2))
  Gamma_jet = Jet(value=Gamma, gradient=None, hessian=None, dim=2)

  conn = Connection(basis=basis, christoffel_symbols=Gamma_jet)

  assert conn.basis is basis
  assert conn.christoffel_symbols.shape == (2, 2, 2)
  assert jnp.allclose(conn.christoffel_symbols.value, 0.0)
  # assert jnp.allclose(conn.christoffel_symbols.gradient, 0.0) # Don't have enough information to check gradient
  # assert jnp.allclose(conn.christoffel_symbols.hessian, 0.0) # Don't have enough information to check hessian


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
  expected = expected.at[0, 0, 0].set(a_u / (2.0 * a_val))           # Γ^u_{uu} -> (i=u,j=u,k=u)
  expected = expected.at[1, 1, 0].set(- b_u / (2.0 * a_val))         # Γ^u_{vv} -> (i=v,j=v,k=u)
  expected = expected.at[0, 1, 1].set(b_u / (2.0 * b_val))           # Γ^v_{uv} -> (i=u,j=v,k=v)
  expected = expected.at[1, 0, 1].set(b_u / (2.0 * b_val))           # Γ^v_{vu} -> (i=v,j=u,k=v)

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
  expected = expected.at[1, 1, 0].set(-r)     # Γ^r_{φφ} -> (i=φ,j=φ,k=r)
  expected = expected.at[0, 1, 1].set(1.0/r)  # Γ^φ_{rφ} -> (i=r,j=φ,k=φ)
  expected = expected.at[1, 0, 1].set(1.0/r)  # Γ^φ_{φr} -> (i=φ,j=r,k=φ)

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
  expected = expected.at[0, 0, 0].set(a_u / (2.0 * a_val))      # Γ^u_{uu} -> (u,u,u)
  expected = expected.at[0, 1, 0].set(a_v / (2.0 * a_val))      # Γ^u_{uv} -> (u,v,u)
  expected = expected.at[1, 0, 0].set(a_v / (2.0 * a_val))      # Γ^u_{vu} -> (v,u,u)
  expected = expected.at[1, 1, 0].set(- b_u / (2.0 * a_val))    # Γ^u_{vv} -> (v,v,u)
  expected = expected.at[0, 0, 1].set(- a_v / (2.0 * b_val))    # Γ^v_{uu} -> (u,u,v)
  expected = expected.at[0, 1, 1].set(b_u / (2.0 * b_val))      # Γ^v_{uv} -> (u,v,v)
  expected = expected.at[1, 0, 1].set(b_u / (2.0 * b_val))      # Γ^v_{vu} -> (v,u,v)
  expected = expected.at[1, 1, 1].set(b_v / (2.0 * b_val))      # Γ^v_{vv} -> (v,v,v)

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
  expected = expected.at[0, 1, 0].set(-1.0 / y)   # Γ^x_{x y} -> (i=x,j=y,k=x)
  expected = expected.at[1, 0, 0].set(-1.0 / y)   # Γ^x_{y x} -> (i=y,j=x,k=x)
  expected = expected.at[0, 0, 1].set(1.0 / y)    # Γ^y_{x x} -> (i=x,j=x,k=y)
  expected = expected.at[1, 1, 1].set(-1.0 / y)   # Γ^y_{y y} -> (i=y,j=y,k=y)

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
  # assert jnp.allclose(out.components.gradient, 0.0) # Don't have enough information to check gradient


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
  term2 = jnp.einsum("ijk,i,j->k", Gamma, X_val, Y_val)
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
  expected = jnp.einsum("i,ki->k", X_val, Y_grad) + jnp.einsum("ijk,i,j->k", Gamma, X_val, Y_val)
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
  expected = jnp.einsum("i,ki->k", X_val, Y_grad) + jnp.einsum("ijk,i,j->k", Gamma, X_val, Y_val)
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
  assert jnp.allclose(conn.christoffel_symbols.gradient, 0.0)
  # assert jnp.allclose(conn.christoffel_symbols.hessian, 0.0) # Don't have enough information to check hessian

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
    np.testing.assert_allclose(
        conn_a.christoffel_symbols.gradient,
        conn_a_round_trip.christoffel_symbols.gradient,
        rtol=1e-5
    )
    # np.testing.assert_allclose(
    #     conn_a.christoffel_symbols.hessian,
    #     conn_a_round_trip.christoffel_symbols.hessian,
    #     rtol=1e-5
    # ) # Don't have enough information to check hessian

def create_random_metric(key: random.PRNGKey, dim: int) -> RiemannianMetric:
  p_key, W_key, basis_key = random.split(key, 3)
  p = jnp.zeros(dim)
  W = random.normal(W_key, (dim, dim, dim))
  def random_metric_func(point):
    g = jnp.einsum('ijk,j,k->i', W, point, point)
    g_matrix = jnp.outer(g, g) + jnp.eye(dim)  # Add identity to ensure invertibility
    return g_matrix
  metric_jet = function_to_jet(random_metric_func, p)
  random_basis = create_random_basis(basis_key, dim)
  return RiemannianMetric(basis=random_basis, components=metric_jet)

def create_random_vector_field(key: random.PRNGKey, dim: int) -> TangentVector:
  p_key, basis_key, vals_key, grads_key, hessians_key = random.split(key, 5)
  p = jnp.zeros(dim)
  random_basis = create_random_basis(basis_key, dim)
  vals = random.normal(vals_key, (dim,))
  grads = random.normal(grads_key, (dim, dim))
  hessians = random.normal(hessians_key, (dim, dim, dim))
  return TangentVector(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians), basis=random_basis)

def test_lc_gamma_antisym_zero_in_cartesian():
  # In a holonomic (coordinate) basis, c_{ij}^k = 0 and torsion-free implies Γ^k_{ij} = Γ^k_{ji}.
  # This should hold for value, gradient, and hessian.
  dim = 3
  p = jnp.zeros(dim)
  basis = get_standard_basis(p)

  # Smooth metric with nontrivial derivatives
  def g_func(x):
    # Positive-definite: A(x) = I + outer(ax, ax)
    a = jnp.array([1.0, -0.5, 0.7])
    ax = jnp.einsum('i,i->', a, x)
    v = a * ax
    return jnp.eye(dim) + jnp.outer(v, v)

  metric = RiemannianMetric(basis=basis, components=function_to_jet(g_func, p))
  conn = get_levi_civita_connection(metric)

  Gamma = conn.christoffel_symbols
  # Antisymmetric part across (i,j)
  A_val = Gamma.value - jnp.transpose(Gamma.value, (1, 0, 2))
  A_grad = Gamma.gradient - jnp.transpose(Gamma.gradient, (1, 0, 2, 3))
  A_hess = Gamma.hessian - jnp.transpose(Gamma.hessian, (1, 0, 2, 3, 4))

  assert jnp.allclose(A_val, 0.0)
  assert jnp.allclose(A_grad, 0.0)
  # assert jnp.allclose(A_hess, 0.0) # Don't have enough information to check hessian


def test_levi_civita_connection_is_torsion_free():
    """
    Tests that the Levi-Civita connection is torsion free.
    """
    key = random.PRNGKey(0)
    dim = 5
    metric = create_random_metric(key, dim)
    connection = get_levi_civita_connection(metric)

    k1, k2, k3 = random.split(key, 3)
    X = create_random_vector_field(k1, dim)
    Y = create_random_vector_field(k2, dim)
    X = change_basis(X, metric.basis)
    Y = change_basis(Y, metric.basis)

    # Test that the connection is torsion free
    nablaX_Y = connection.covariant_derivative(X, Y)
    nablaY_X = connection.covariant_derivative(Y, X)
    lb_XY = lie_bracket(X, Y)
    torsion = nablaX_Y - nablaY_X - lb_XY
    assert jnp.allclose(torsion.components.value, 0.0)
    assert jnp.allclose(torsion.components.gradient, 0.0)
    # assert jnp.allclose(torsion.components.hessian, 0.0) # Don't have enough information to check hessian

def test_levi_civita_connection_is_metric_compatible():
    """
    Tests that the Levi-Civita connection is metric compatible.
    """
    key = random.PRNGKey(0)
    dim = 5
    metric = create_random_metric(key, dim)
    connection = get_levi_civita_connection(metric)

    k1, k2, k3 = random.split(key, 3)
    X = create_random_vector_field(k1, dim)
    Y = create_random_vector_field(k2, dim)
    Z = create_random_vector_field(k3, dim)
    X = change_basis(X, metric.basis)
    Y = change_basis(Y, metric.basis)
    Z = change_basis(Z, metric.basis)

    @jet_decorator
    def metric_inner_product(g_vals, U_vals, V_vals) -> Array:
      return jnp.einsum("ij,i,j->", g_vals, U_vals, V_vals)

    nablaX_Y: TangentVector = connection.covariant_derivative(X, Y)
    nablaX_Z: TangentVector = connection.covariant_derivative(X, Z)

    gYZ: Jet = metric_inner_product(metric.components, Y.components, Z.components)
    XgYZ: Jet = X(gYZ)

    gnablaXY_Z: Jet = metric_inner_product(metric.components, nablaX_Y.components, Z.components)
    gnablaXZ_Y: Jet = metric_inner_product(metric.components, Y.components, nablaX_Z.components)

    comp = gnablaXY_Z + gnablaXZ_Y - XgYZ
    assert jnp.allclose(comp.value, 0.0)
    assert jnp.allclose(comp.gradient, 0.0)
    # assert jnp.allclose(comp.hessian, 0.0) # Don't have enough information to check hessian

def test_covariant_hessian():
    key = random.PRNGKey(0)
    dim = 5
    metric = create_random_metric(key, dim)
    connection = get_levi_civita_connection(metric)

    k1, k2, k3 = random.split(key, 3)
    X = create_random_vector_field(k1, dim)
    Y = create_random_vector_field(k2, dim)
    Z = create_random_vector_field(k3, dim)
    X = change_basis(X, metric.basis)
    Y = change_basis(Y, metric.basis)
    Z = change_basis(Z, metric.basis)

    def covariant_hessian(X: TangentVector, Y: TangentVector, Z: TangentVector) -> TangentVector:
      nablaY_Z = connection.covariant_derivative(Y, Z)
      nablaX_Y = connection.covariant_derivative(X, Y)
      term1 = connection.covariant_derivative(X, nablaY_Z)
      term2 = connection.covariant_derivative(nablaX_Y, Z)
      return term1 - term2

    # LHS
    nabla2XY_Z = covariant_hessian(X, Y, Z)
    nabla2YX_Z = covariant_hessian(Y, X, Z)
    R_XYZ = nabla2XY_Z - nabla2YX_Z

    # RHS
    nablaY_Z = connection.covariant_derivative(Y, Z)
    nablaX_Z = connection.covariant_derivative(X, Z)
    nablaX_nablaY_Z = connection.covariant_derivative(X, nablaY_Z)
    nablaY_nablaX_Z = connection.covariant_derivative(Y, nablaX_Z)

    bracket_XY = lie_bracket(X, Y)
    nabla_bracket_XY_Z = connection.covariant_derivative(bracket_XY, Z)

    R_XYZ2 = nablaX_nablaY_Z - nablaY_nablaX_Z - nabla_bracket_XY_Z

    assert jnp.allclose(R_XYZ.components.value, R_XYZ2.components.value)

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

def test_connection_change_coordinates_round_trip():
  """
  Test x -> z -> x round trip for connection.
  """
  q = jnp.array([2.0, 1.0, 0.5]) # Spherical
  x = spherical_to_cartesian(q)
  dim = 3
  key = random.PRNGKey(99)

  # Random connection in q-coords
  basis_q = create_random_basis(key, dim)
  gamma_val = random.normal(key, (dim, dim, dim))
  gamma_grad = random.normal(key, (dim, dim, dim, dim))
  # gamma_hess = random.normal(key, (dim, dim, dim, dim, dim))
  gamma_jet = Jet(value=gamma_val, gradient=gamma_grad, hessian=None, dim=dim)

  conn_q = Connection(basis=basis_q, christoffel_symbols=gamma_jet)

  # Change to x
  J_zq = function_to_jacobian(spherical_to_cartesian, q) # q -> x
  conn_x = change_coordinates(conn_q, J_zq)

  # Change back to q
  J_xz = function_to_jacobian(cartesian_to_spherical, x) # x -> q
  conn_q_restored = change_coordinates(conn_x, J_xz)

  assert jnp.allclose(conn_q_restored.christoffel_symbols.value, conn_q.christoffel_symbols.value, atol=1e-5)
  assert jnp.allclose(conn_q_restored.christoffel_symbols.gradient, conn_q.christoffel_symbols.gradient, atol=1e-5)

def test_connection_change_coordinates_preserves_christoffel():
  """
  Test that change_coordinates preserves the connection coefficients.

  When we change coordinates, the connection coefficients with respect to
  the same basis should remain unchanged (only derivatives are adjusted).
  """
  # Cartesian point
  r = 2.5
  phi = 1.2
  x_val = r * jnp.cos(phi)
  y_val = r * jnp.sin(phi)
  x = jnp.array([x_val, y_val])

  # Connection with non-zero Christoffel symbols and a random basis
  key = random.PRNGKey(0)
  basis_x = create_random_basis(key, 2)
  gamma_original = random.normal(random.split(key)[0], (2, 2, 2))
  gamma_jet = Jet(value=gamma_original, gradient=jnp.zeros((2,2,2,2)), hessian=None, dim=2)
  conn_x = Connection(basis=basis_x, christoffel_symbols=gamma_jet)

  # Helper for cartesian -> polar map (2D)
  def cart_to_pol_2d(pt):
    xx, yy = pt
    rr = jnp.sqrt(xx**2 + yy**2)
    pp = jnp.arctan2(yy, xx)
    return jnp.array([rr, pp])

  J_xz = function_to_jacobian(cart_to_pol_2d, x)

  conn_q = change_coordinates(conn_x, J_xz)

  # The Christoffel values should stay the same (they're for the same basis)
  assert jnp.allclose(conn_q.christoffel_symbols.value, gamma_original, atol=1e-5)

def test_connection_covariant_derivative_change_coordinates():
  """
  Test covariant derivative of a connection does not depend on the coordinate system.
  """
  dim = 3
  key = random.PRNGKey(99)
  k1, k2, k3, k4, k5, k6, k7, k8, k9 = random.split(key, 9)

  # Use a non-degenerate point (spherical coords are singular at origin!)
  p = jnp.array([1.5, 0.9, 1.2])

  # Create random basis at the correct point
  basis_vals = random.normal(k1, (dim, dim))
  basis_grads = random.normal(k2, (dim, dim, dim))
  basis_hess = random.normal(k3, (dim, dim, dim, dim))
  random_basis = BasisVectors(p=p, components=Jet(value=basis_vals, gradient=basis_grads, hessian=basis_hess))

  # Create a smooth metric at the point
  W = random.normal(k4, (dim, dim, dim))
  def metric_func(point):
    g = jnp.einsum('ijk,j,k->i', W, point, point)
    g_matrix = jnp.outer(g, g) + jnp.eye(dim)
    return g_matrix
  metric_jet = function_to_jet(metric_func, p)
  metric = RiemannianMetric(basis=random_basis, components=metric_jet)

  # Get connection
  connection_x = get_levi_civita_connection(metric)

  # Create random vector fields at the same point
  X_vals = random.normal(k5, (dim,))
  X_grads = random.normal(k6, (dim, dim))
  X_hess = random.normal(k7, (dim, dim, dim))
  X_x = TangentVector(p=p, components=Jet(value=X_vals, gradient=X_grads, hessian=X_hess), basis=random_basis)

  Y_vals = random.normal(k8, (dim,))
  Y_grads = random.normal(k9, (dim, dim))
  Y_hess = random.normal(random.split(k9)[0], (dim, dim, dim))
  Y_x = TangentVector(p=p, components=Jet(value=Y_vals, gradient=Y_grads, hessian=Y_hess), basis=random_basis)

  # Transform to spherical coordinates
  jac: Jacobian = function_to_jacobian(spherical_to_cartesian, p)
  connection_q = change_coordinates(connection_x, jac)
  X_q = change_coordinates(X_x, jac)
  Y_q = change_coordinates(Y_x, jac)

  # Compute covariant derivative in both coordinate systems
  nablaX_Y_x = connection_x.covariant_derivative(X_x, Y_x)
  nablaX_Y_q = connection_q.covariant_derivative(X_q, Y_q)

  # Transform result back and compare
  jac_inv: Jacobian = get_inverse(jac)
  nablaX_Y_x_from_q = change_coordinates(nablaX_Y_q, jac_inv)

  assert jnp.allclose(nablaX_Y_x_from_q.components.value, nablaX_Y_x.components.value, atol=1e-5)
  # Note: gradient/hessian may have numerical issues due to third derivatives
  # assert jnp.allclose(nablaX_Y_x_from_q.components.gradient, nablaX_Y_x.components.gradient, atol=1e-5)