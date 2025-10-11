import jax.numpy as jnp
from local_coordinates.basis import BasisVectors
from local_coordinates.tensor import Tensor, TensorType
from local_coordinates.connection import Connection
from local_coordinates.jet import Jet, function_to_jet, _expand_jet
import jax.random as random
import jax
from local_coordinates.connection import change_coordinates
from local_coordinates.jet import jet_decorator
from local_coordinates.tensor import function_multiply_tensor

def make_identity_basis(dim: int = 2) -> BasisVectors:
  p = jnp.zeros((dim,))
  frame = jnp.eye(dim)
  # Gradient must have one more axis than value; pick R = dim
  dframe = jnp.zeros((dim, dim, dim))
  d2frame = jnp.zeros((dim, dim, dim, dim))
  return BasisVectors(p=p, components=Jet(value=frame, gradient=dframe, hessian=d2frame))

def make_random_basis(dim: int = 2) -> BasisVectors:
  key = random.PRNGKey(0)
  p = jnp.zeros((dim,))
  frame = random.normal(key, (dim, dim))
  # Gradient must have one more axis than value; pick R = dim
  dframe = random.normal(key, (dim, dim, dim))
  d2frame = random.normal(key, (dim, dim, dim, dim))
  return BasisVectors(p=p, components=Jet(value=frame, gradient=dframe, hessian=d2frame))


def make_vector(
    basis: BasisVectors,
    comps: jnp.ndarray,
    grad: jnp.ndarray | None,
    hess: jnp.ndarray | None = None
) -> Tensor:
  ttype = TensorType(1, 0)
  jet = Jet(value=comps, gradient=grad, hessian=hess)
  return Tensor(tensor_type=ttype, basis=basis, components=jet)


def test_connection_creation():
  basis = make_identity_basis(2)
  Gamma = jnp.zeros((2, 2, 2))
  Gamma_jet = Jet(value=Gamma, gradient=None, hessian=None)
  conn = Connection(basis=basis, christoffel_symbols=Gamma_jet)
  assert conn.basis is basis
  assert jnp.allclose(conn.christoffel_symbols.value, Gamma)


def test_covariant_derivative_zero_connection():
  basis = make_identity_basis(2)

  # Zero Christoffel symbols
  Gamma = jnp.zeros((2, 2, 2))
  conn = Connection(basis=basis, christoffel_symbols=Jet(value=Gamma, gradient=None, hessian=None))

  # X = [1, 0]
  X = make_vector(basis, comps=jnp.array([1.0, 0.0]), grad=None)

  # Y with gradient dY^a/dx^k = G[a, k]
  Y_val = jnp.array([3.0, -2.0])
  Y_grad = jnp.array([[10.0, 20.0],   # dY^0/dx^0, dY^0/dx^1
                      [30.0, 40.0]])  # dY^1/dx^0, dY^1/dx^1
  Y = make_vector(basis, comps=Y_val, grad=Y_grad)

  # Expected: X^i ∂_i Y^k (Γ=0)
  expected = jnp.einsum("i,ki->k", X.components.value, Y.components.gradient)

  out = conn.covariant_derivative(X, Y)
  assert jnp.allclose(out.components.value, expected)


def test_covariant_derivative_with_connection():
  basis = make_identity_basis(2)

  # Non-zero Christoffel symbols
  # Γ^k_{ij}: choose a few simple entries
  Gamma = jnp.zeros((2, 2, 2))
  Gamma = Gamma.at[0, 0, 0].set(2.0)  # Γ^0_{00} = 2
  Gamma = Gamma.at[1, 0, 1].set(-1.0) # Γ^1_{01} = -1
  conn = Connection(basis=basis, christoffel_symbols=Jet(value=Gamma, gradient=None, hessian=None))

  X = make_vector(basis, comps=jnp.array([1.0, 2.0]), grad=None)
  Y_val = jnp.array([3.0, 4.0])
  Y_grad = jnp.array([[1.0, 5.0], [2.0, 6.0]])
  Y = make_vector(basis, comps=Y_val, grad=Y_grad)

  # term1_k = X^i ∂_i Y^k
  term1 = jnp.einsum("i,ki->k", X.components.value, Y.components.gradient)
  # term2_k = Γ^k_{ij} x^i y^j
  term2 = jnp.einsum("kij,i,j->k", Gamma, X.components.value, Y.components.value)
  expected = term1 + term2

  out = conn.covariant_derivative(X, Y)
  assert jnp.allclose(out.components.value, expected)


def test_covariant_derivative_catches_index_swap_bug():
  """
  Construct data so that the incorrect contraction X^a ∂_k Y^a would differ
  from the correct X^i ∂_i Y^k. This should fail if indices are swapped.
  """
  basis = make_identity_basis(2)
  Gamma = jnp.zeros((2, 2, 2))
  conn = Connection(basis=basis, christoffel_symbols=Jet(value=Gamma, gradient=None, hessian=None))

  X = make_vector(basis, comps=jnp.array([2.0, 3.0]), grad=None)  # X^0=2, X^1=3
  # Choose Y such that rows/cols are distinct
  Y_val = jnp.array([0.0, 0.0])
  Y_grad = jnp.array([[1.0, 10.0],   # k=0 row
                      [100.0, 1000.0]])  # k=1 row
  Y = make_vector(basis, comps=Y_val, grad=Y_grad)

  # Correct: [ sum_i X^i ∂_i Y^0, sum_i X^i ∂_i Y^1 ]
  expected = jnp.einsum("i,ki->k", X.components.value, Y.components.gradient)
  out = conn.covariant_derivative(X, Y)
  assert jnp.allclose(out.components.value, expected)


def test_covariant_derivative_change_of_basis():
  basis = make_identity_basis(2)
  new_basis = make_random_basis(2)

  # Non-zero Christoffel symbols
  key = random.PRNGKey(0)
  Gamma = random.normal(key, (2, 2, 2))*0
  conn = Connection(basis=basis, christoffel_symbols=Jet(value=Gamma, gradient=None, hessian=None))

  X = make_vector(basis, comps=jnp.array([1.0, 2.0]), grad=None)
  Y_val = jnp.array([3.0, 4.0])
  Y_grad = jnp.array([[1.0, 5.0], [2.0, 6.0]])
  Y = make_vector(basis, comps=Y_val, grad=Y_grad)

  out1: Tensor = conn.covariant_derivative(X, Y)
  true_out2: Tensor = change_coordinates(out1, new_basis)

  conn2 = change_coordinates(conn, new_basis)
  comp_out2: Tensor = conn2.covariant_derivative(X, Y)

  assert jnp.allclose(true_out2.components.value, comp_out2.components.value)
  assert true_out2.components.gradient == comp_out2.components.gradient
  assert true_out2.components.hessian == comp_out2.components.hessian


def test_covariant_derivative_linearity_fX():
    """
    Tests the linearity property: ∇_{fX}Y = f ∇_X Y
    """
    basis = make_identity_basis(2)
    conn = Connection(basis=basis, christoffel_symbols=Jet(value=jnp.zeros((2,2,2)), gradient=None, hessian=None))

    p = jnp.array([0.5, 1.5])
    f_jet = function_to_jet(lambda x: x[0]**2, p)

    X = make_vector(basis, comps=jnp.array([1., 2.]), grad=jnp.ones((2,2)))
    Y = make_vector(basis, comps=jnp.array([3., 4.]), grad=jnp.ones((2,2)))

    # LHS: ∇_{fX}Y
    fX = function_multiply_tensor(X, f_jet)
    lhs = conn.covariant_derivative(fX, Y)

    # RHS: f ∇_X Y
    nabla_X_Y = conn.covariant_derivative(X, Y)
    rhs = function_multiply_tensor(nabla_X_Y, f_jet)

    assert jnp.allclose(lhs.components.value, rhs.components.value)
    if lhs.components.gradient is not None and rhs.components.gradient is not None:
        assert jnp.allclose(lhs.components.gradient, rhs.components.gradient)
    else:
        assert lhs.components.gradient is None and rhs.components.gradient is None


def test_covariant_derivative_linearity_fY():
    """
    Tests the product rule property: ∇_X(fY) = X(f)Y + f(∇_X Y)
    """
    basis = make_identity_basis(2)
    conn = Connection(basis=basis, christoffel_symbols=Jet(value=jnp.zeros((2,2,2)), gradient=None, hessian=None))

    p = jnp.array([0.5, 1.5])
    f_jet = function_to_jet(lambda x: x[0]**2, p)

    X = make_vector(basis, comps=jnp.array([1., 2.]), grad=jnp.ones((2,2))*1.0)
    Y = make_vector(basis, comps=jnp.array([3., 4.]), grad=jnp.ones((2,2))*1.0)

    # LHS: ∇_X(fY)
    fY = function_multiply_tensor(Y, f_jet)
    lhs = conn.covariant_derivative(X, fY)

    # RHS: X(f)Y + f(∇_X Y)
    @jet_decorator
    def X_f_components(X_components_val, f_grad_val):
      return jnp.einsum('i,i', f_grad_val, X_components_val)

    _, f_grad_jet, _ = _expand_jet(f_jet)
    X_f = X_f_components(X.components, f_grad_jet)
    X_f_Y = function_multiply_tensor(Y, X_f)

    nabla_X_Y = conn.covariant_derivative(X, Y)
    f_nabla_X_Y = function_multiply_tensor(nabla_X_Y, f_jet)

    rhs = X_f_Y + f_nabla_X_Y

    assert jnp.allclose(lhs.components.value, rhs.components.value)
    if lhs.components.gradient is not None and rhs.components.gradient is not None:
        assert jnp.allclose(lhs.components.gradient, rhs.components.gradient)
    else:
        assert lhs.components.gradient is None and rhs.components.gradient is None


def test_covariant_derivative_linearity_fX_with_hessian():
    """
    Tests the linearity property: ∇_{fX}Y = f ∇_X Y with non-zero hessians.
    """
    basis = make_identity_basis(2)
    conn = Connection(basis=basis, christoffel_symbols=Jet(value=jnp.zeros((2,2,2)), gradient=None, hessian=None))

    p = jnp.array([0.5, 1.5])
    f_jet = function_to_jet(lambda x: x[0]**2, p)

    X = make_vector(
        basis,
        comps=jnp.array([1., 2.]),
        grad=jnp.ones((2,2)),
        hess=jnp.ones((2,2,2)) * 0.1
    )
    Y = make_vector(
        basis,
        comps=jnp.array([3., 4.]),
        grad=jnp.ones((2,2)),
        hess=jnp.ones((2,2,2)) * 0.2
    )

    # LHS: ∇_{fX}Y
    fX = function_multiply_tensor(X, f_jet)
    lhs = conn.covariant_derivative(fX, Y)

    # RHS: f ∇_X Y
    nabla_X_Y = conn.covariant_derivative(X, Y)
    rhs = function_multiply_tensor(nabla_X_Y, f_jet)

    assert jnp.allclose(lhs.components.value, rhs.components.value)
    assert jnp.allclose(lhs.components.gradient, rhs.components.gradient)
    if lhs.components.hessian is not None and rhs.components.hessian is not None:
        assert jnp.allclose(lhs.components.hessian, rhs.components.hessian)
    else:
        assert lhs.components.hessian is None and rhs.components.hessian is None


def test_covariant_derivative_linearity_fY_with_hessian():
    """
    Tests the product rule property: ∇_X(fY) = X(f)Y + f(∇_X Y) with non-zero hessians.
    """
    basis = make_identity_basis(2)
    conn = Connection(basis=basis, christoffel_symbols=Jet(value=jnp.zeros((2,2,2)), gradient=None, hessian=None))

    p = jnp.array([0.5, 1.5])
    f_jet = function_to_jet(lambda x: x[0]**2, p)

    X = make_vector(
        basis,
        comps=jnp.array([1., 2.]),
        grad=jnp.ones((2,2))*1.0,
        hess=jnp.ones((2,2,2)) * 0.1
    )
    Y = make_vector(
        basis,
        comps=jnp.array([3., 4.]),
        grad=jnp.ones((2,2))*1.0,
        hess=jnp.ones((2,2,2)) * 0.2
    )

    # LHS: ∇_X(fY)
    fY = function_multiply_tensor(Y, f_jet)
    lhs = conn.covariant_derivative(X, fY)

    # RHS: X(f)Y + f(∇_X Y)
    @jet_decorator
    def X_f_components(X_comps, X_grad, f_grad, f_hess):
        # Directional derivative of f along X.
        # X(f) = <grad(f), X>
        val = jnp.einsum('i,i', f_grad, X_comps)
        return val

    X_val_jet, X_grad_jet, _ = _expand_jet(X.components)
    f_val_jet, f_grad_jet, f_hess_jet = _expand_jet(f_jet)
    X_f = X_f_components(X_val_jet, X_grad_jet, f_grad_jet, f_hess_jet)
    X_f_Y = function_multiply_tensor(Y, X_f)

    nabla_X_Y = conn.covariant_derivative(X, Y)
    f_nabla_X_Y = function_multiply_tensor(nabla_X_Y, f_jet)

    rhs = X_f_Y + f_nabla_X_Y

    assert jnp.allclose(lhs.components.value, rhs.components.value)
    assert jnp.allclose(lhs.components.gradient, rhs.components.gradient)
    if lhs.components.hessian is not None and rhs.components.hessian is not None:
        assert jnp.allclose(lhs.components.hessian, rhs.components.hessian)
    else:
        assert lhs.components.hessian is None and rhs.components.hessian is None