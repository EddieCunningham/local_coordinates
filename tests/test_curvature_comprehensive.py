import jax.numpy as jnp
from jax import random
import numpy as np

from local_coordinates.metric import RiemannianMetric
from local_coordinates.basis import get_standard_basis
from local_coordinates.jet import function_to_jet
from local_coordinates.connection import get_levi_civita_connection
from local_coordinates.riemann import get_riemann_curvature_tensor
from local_coordinates.basis import BasisVectors
from local_coordinates.frame import get_lie_bracket_between_frame_pairs, basis_to_frame
from local_coordinates.jet import Jet, jet_decorator, get_identity_jet
from local_coordinates.tensor import change_basis
from local_coordinates.tangent import TangentVector, lie_bracket
from jaxtyping import Array
from typing import Annotated
from local_coordinates.connection import Connection

def create_random_basis(key: random.PRNGKey, dim: int) -> BasisVectors:
  p_key, vals_key, grads_key, hessians_key = random.split(key, 4)
  p = jnp.zeros(dim)
  vals = random.normal(vals_key, (dim, dim))*0.1
  grads = random.normal(grads_key, (dim, dim, dim))*0.1
  hessians = random.normal(hessians_key, (dim, dim, dim, dim))*0.1
  return BasisVectors(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians))

def create_random_metric(key: random.PRNGKey, dim: int) -> RiemannianMetric:
  random_basis = create_random_basis(key, dim)
  return RiemannianMetric(basis=random_basis, components=get_identity_jet(dim))

def create_random_vector_field(key: random.PRNGKey, dim: int) -> TangentVector:
  p_key, basis_key, vals_key, grads_key, hessians_key = random.split(key, 5)
  p = jnp.zeros(dim)
  random_basis = create_random_basis(basis_key, dim)
  vals = random.normal(vals_key, (dim,))
  grads = random.normal(grads_key, (dim, dim))
  hessians = random.normal(hessians_key, (dim, dim, dim))
  return TangentVector(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians), basis=random_basis)

def get_double_covariant_derivative_terms(connection: Connection, X: TangentVector, Y: TangentVector, Z: TangentVector):
  basis = connection.basis
  Gamma = connection.christoffel_symbols

  #####################################################
  # ∇_X ∇_Y Z = [
  # Term 1:         X(Y^a) E_a(Z^l) + Y^a X^b E_b(E_a(Z^l))
  # Term 2:       + X(Gamma^l_{ij}) Y^i Z^j + Gamma^l_{ij} X(Y^i) Z^j + Gamma^l_{ij} Y^i X(Z^j)
  # Term 3:       + Y(Z^k) Gamma^l_{ik} X^i
  # Term 4:       + Gamma^k_{ij} Y^i Z^j Gamma^l_{mk} X^m
  #                                                         ] E_l
  #####################################################

  ###################
  # Term 1:
  ###################

  # X(Y^a) E_a(Z^l) = X^i E_i(Y^a) E_a(Z^l)
  @jet_decorator
  def term1a_computation(X_val, Y_grad, E_val, Z_grad):
    return jnp.einsum("i,bi,ab,ca,lc->l", X_val, E_val, Y_grad, E_val, Z_grad)
  term1a = term1a_computation(
    X.components,
    Y.components.get_gradient_jet(),
    basis.components,
    Z.components.get_gradient_jet()
  )

  # Y^a X^b E_b(E_a(Z^l))
  @jet_decorator
  def EaZl_computation(E_val, Z_grad):
    return jnp.einsum("ba,lb->al", E_val, Z_grad)
  EaZl = EaZl_computation(basis.components, Z.components.get_gradient_jet())

  @jet_decorator
  def term1b_computation(Y_val, X_val, E_val, EaZl_grad):
    return jnp.einsum("a,b,cb,alc->l", Y_val, X_val, E_val, EaZl_grad)
  term1b = term1b_computation(
    Y.components,
    X.components,
    basis.components,
    EaZl.get_gradient_jet()
  )

  # Checkpoint 1
  term1_truth = X(Y(Z.components))
  term1 = term1a + term1b
  assert jnp.allclose(term1.value, term1_truth.value)


  ###################
  # Term 2:
  ###################

  # X(Gamma^l_{ij}) Y^i Z^j = X^a E_a(Gamma^l_{ij}) Y^i Z^j
  @jet_decorator
  def term2a_computation(X_val, E_val, Gamma_grad, Y_val, Z_val):
    return jnp.einsum("a,ba,ijlb,i,j->l", X_val, E_val, Gamma_grad, Y_val, Z_val)
  term2a = term2a_computation(
    X.components,
    basis.components,
    Gamma.get_gradient_jet(),
    Y.components,
    Z.components
  )

  # Gamma^l_{ij} X(Y^i) Z^j = Gamma^l_{ij} X^a E_a(Y^i) Z^j
  @jet_decorator
  def term2b_computation(Gamma_val, X_val, E_val, Y_grad, Z_val):
    return jnp.einsum("ijl,a,ba,ib,j->l", Gamma_val, X_val, E_val, Y_grad, Z_val)
  term2b = term2b_computation(
    Gamma,
    X.components,
    basis.components,
    Y.components.get_gradient_jet(),
    Z.components
  )

  # Gamma^l_{ij} Y^i X(Z^j) = Gamma^l_{ij} Y^i X^a E_a(Z^j)
  @jet_decorator
  def term2c_computation(Gamma_val, Y_val, X_val, E_val, Z_grad):
    return jnp.einsum("ijl,i,a,ba,jb->l", Gamma_val, Y_val, X_val, E_val, Z_grad)
  term2c = term2c_computation(
    Gamma,
    Y.components,
    X.components,
    basis.components,
    Z.components.get_gradient_jet()
  )

  # Checkpoint 2
  @jet_decorator
  def Gamma_YZ_computation(Gamma_val, Y_val, Z_val):
    return jnp.einsum("ijl,i,j->l", Gamma_val, Y_val, Z_val)
  Gamma_YZ = Gamma_YZ_computation(
    Gamma,
    Y.components,
    Z.components
  )
  term2_truth = X(Gamma_YZ)
  term2 = term2a + term2b + term2c
  assert jnp.allclose(term2.value, term2_truth.value)


  ###################
  # Term 3:
  ###################

  # Y(Z^k) Gamma^l_{ik} X^i = Y^a E_a(Z^k) Gamma^l_{ik} X^i
  @jet_decorator
  def term3_computation(Y_val, E_val, Z_grad, Gamma_val, X_val):
    return jnp.einsum("a,ba,kb,ikl,i->l", Y_val, E_val, Z_grad, Gamma_val, X_val)
  term3 = term3_computation(
    Y.components,
    basis.components,
    Z.components.get_gradient_jet(),
    Gamma,
    X.components
  )


  ###################
  # Term 4:
  ###################

  # Gamma^k_{ij} Y^i Z^j Gamma^l_{mk} X^m
  @jet_decorator
  def term4_computation(Gamma_val, Y_val, Z_val, X_val):
    return jnp.einsum("ijk,i,j,mkl,m->l", Gamma_val, Y_val, Z_val, Gamma_val, X_val)
  term4 = term4_computation(
    Gamma,
    Y.components,
    Z.components,
    X.components
  )
  return term1a, term1b, term2a, term2b, term2c, term3, term4

def get_bracket_covariant_derivative_terms(connection: Connection, X: TangentVector, Y: TangentVector, Z: TangentVector):
  basis = connection.basis

  frame = basis_to_frame(basis)
  lie_bracket_pairs: Annotated[TangentVector, "i j"] = get_lie_bracket_between_frame_pairs(frame)
  c = lie_bracket_pairs.components
  Gamma = connection.christoffel_symbols

  #####################################################
  # ∇_{[X,Y]} Z = [
  # Term 1:        c^k_{ij} X^i Y^j E_k(Z^l)
  # Term 2:      + c^k_{ij} X^i Y^j Gamma^l_{km} Z^m ( was c^k_{ij} X^i Y^j Gamma^l_{km} Z^m )
  # Term 3:      + X(Y^j) E_j(Z^l)
  # Term 4:      - Y(X^i) E_i(Z^l)
  # Term 5:      + X(Y^j) Gamma^l_{jm} Z^m
  # Term 6:      - Y(X^i) Gamma^l_{im} Z^m
  #                                                         ] E_l
  #####################################################

  ###################
  # Term 1:
  ###################

  # c^k_{ij} X^i Y^j E_k(Z^l)
  @jet_decorator
  def term1_computation(c_val, X_val, Y_val, E_val, Z_grad):
    return jnp.einsum("ijk,i,j,ak,la->l", c_val, X_val, Y_val, E_val, Z_grad)
  term1 = term1_computation(
    c,
    X.components,
    Y.components,
    basis.components,
    Z.components.get_gradient_jet(),
  )

  ###################
  # Term 2:
  ###################

  # c^k_{ij} X^i Y^j Gamma^l_{mk} Z^m  # SHOULD BE: c^k_{ij} X^i Y^j Gamma^l_{km} Z^m
  @jet_decorator
  def term2_computation(c_val, X_val, Y_val, Gamma_val, Z_val):
    return jnp.einsum("ijk,i,j,kml,m->l", c_val, X_val, Y_val, Gamma_val, Z_val)
    # return jnp.einsum("ijk,i,j,mkl,m->l", c_val, X_val, Y_val, Gamma_val, Z_val) # THIS WAS THE BUG!  THE INDEX ORDER FOR GAMMA WAS WRONG!
  term2 = term2_computation(
    c,
    X.components,
    Y.components,
    Gamma,
    Z.components,
  )

  ###################
  # Term 3:
  ###################

  # + X(Y^j) E_j(Z^l) = X^a E_a(Y^j) E_j(Z^l)
  @jet_decorator
  def term3_computation(X_val, E_val1, Y_grad, E_val2, Z_grad):
    return jnp.einsum("a,ba,jb,cj,lc->l", X_val, E_val1, Y_grad, E_val2, Z_grad)
  term3 = term3_computation(
    X.components,
    basis.components,
    Y.components.get_gradient_jet(),
    basis.components,
    Z.components.get_gradient_jet(),
  )

  ###################
  # Term 4:
  ###################

  # - Y(X^i) E_i(Z^l) = - Y^a E_a(X^i) E_i(Z^l)
  @jet_decorator
  def term4_computation(Y_val, E_val, X_grad, Z_grad):
    return -jnp.einsum("a,ba,ib,ci,lc->l", Y_val, E_val, X_grad, E_val, Z_grad)
  term4 = term4_computation(
    Y.components,
    basis.components,
    X.components.get_gradient_jet(),
    Z.components.get_gradient_jet(),
  )

  ###################
  # Term 5:
  ###################

  # + X(Y^j) Gamma^l_{jm} Z^m = X^a E_a(Y^j) Gamma^l_{jm} Z^m
  @jet_decorator
  def term5_computation(X_val, E_val, Y_grad, Gamma_val, Z_val):
    return jnp.einsum("a,ba,jb,jml,m->l", X_val, E_val, Y_grad, Gamma_val, Z_val)
  term5 = term5_computation(
    X.components,
    basis.components,
    Y.components.get_gradient_jet(),
    Gamma,
    Z.components,
  )

  ###################
  # Term 6:
  ###################

  # - Y(X^i) Gamma^l_{im} Z^m = - Y^a E_a(X^i) Gamma^l_{im} Z^m
  @jet_decorator
  def term6_computation(Y_val, E_val, X_grad, Gamma_val, Z_val):
    return -jnp.einsum("a,ba,ib,iml,m->l", Y_val, E_val, X_grad, Gamma_val, Z_val)
  term6 = term6_computation(
    Y.components,
    basis.components,
    X.components.get_gradient_jet(),
    Gamma,
    Z.components,
  )

  return term1, term2, term3, term4, term5, term6

def test_curvature_comprehensive():
  """
  Unfortuntately, its come to this...... This is a brute force test
  to check each term involved in the computation of R(X,Y)Z.  We will
  ensure that all terms are computed as expected, that the expected
  cancellation occurs, and that we recover the correct curvature tensor.
  """
  key = random.PRNGKey(42)
  dim = 2
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)
  basis = connection.basis

  k1, k2, k3 = random.split(key, 3)
  X = change_basis(create_random_vector_field(k1, dim), connection.basis)
  Y = change_basis(create_random_vector_field(k2, dim), connection.basis)
  Z = change_basis(create_random_vector_field(k3, dim), connection.basis)

  # Ground truth for comparison
  nablaY_Z = connection.covariant_derivative(Y, Z)
  nablaX_Z = connection.covariant_derivative(X, Z)
  bracket_XY = lie_bracket(X, Y)
  nablaX_nablaY_Z_truth = connection.covariant_derivative(X, nablaY_Z)
  nablaY_nablaX_Z_truth = connection.covariant_derivative(Y, nablaX_Z)
  nabla_bracket_XY_Z_truth = connection.covariant_derivative(bracket_XY, Z)
  R_XYZ_truth = nablaX_nablaY_Z_truth - nablaY_nablaX_Z_truth - nabla_bracket_XY_Z_truth

  ###################
  ###################
  ###################
  # Construct the Riemann curvature tensor
  riemann_tensor = get_riemann_curvature_tensor(connection)

  @jet_decorator
  def apply_riemann_tensor(R_val: Array, X_val: Array, Y_val: Array, Z_val: Array) -> Array:
    # einsum is for R_{ijk}^l X^i Y^j Z^k
    return jnp.einsum("ijkl,i,j,k->l", R_val, X_val, Y_val, Z_val)

  R_val = riemann_tensor.components.get_value_jet()
  X_val = X.components.get_value_jet()
  Y_val = Y.components.get_value_jet()
  Z_val = Z.components.get_value_jet()
  out = apply_riemann_tensor(R_val, X_val, Y_val, Z_val)
  ###################
  ###################
  ###################


  ###################
  # ∇_X ∇_Y Z
  ###################
  A_term1a, A_term1b, A_term2a, A_term2b, A_term2c, A_term3, A_term4 = get_double_covariant_derivative_terms(connection, X, Y, Z)
  nablaX_nablaY_Z = A_term1a + A_term1b + A_term2a + A_term2b + A_term2c + A_term3 + A_term4
  assert jnp.allclose(nablaX_nablaY_Z.value, nablaX_nablaY_Z_truth.components.value)

  ###################
  # ∇_Y ∇_X Z
  ###################
  B_term1a, B_term1b, B_term2a, B_term2b, B_term2c, B_term3, B_term4 = get_double_covariant_derivative_terms(connection, Y, X, Z)
  nablaY_nablaX_Z = B_term1a + B_term1b + B_term2a + B_term2b + B_term2c + B_term3 + B_term4
  assert jnp.allclose(nablaY_nablaX_Z.value, nablaY_nablaX_Z_truth.components.value)

  ###################
  # ∇_{[X,Y]} Z
  ###################
  C_term1, C_term2, C_term3, C_term4, C_term5, C_term6 = get_bracket_covariant_derivative_terms(connection, X, Y, Z)
  nabla_bracket_XY_Z = C_term1 + C_term2 + C_term3 + C_term4 + C_term5 + C_term6
  assert jnp.allclose(nabla_bracket_XY_Z.value, nabla_bracket_XY_Z_truth.components.value) # Fails!

  ###################
  # R(X,Y)Z = ∇_X ∇_Y Z - ∇_Y ∇_X Z - ∇_{[X,Y]} Z
  ###################
  R_XYZ = nablaX_nablaY_Z - nablaY_nablaX_Z - nabla_bracket_XY_Z
  assert jnp.allclose(R_XYZ.value, R_XYZ_truth.components.value) # Fails!

  ###################
  ###################
  # Cancellations in the Riemann tensor
  ###################
  ###################

  ###################
  # A_term1a = X(Y^a) E_a(Z^l)
  # C_term3 = X(Y^j) E_j(Z^l)
  ###################
  assert jnp.allclose(A_term1a.value, C_term3.value)

  ###################
  # -B_term1a = -Y(X^a) E_a(Z^l)
  # C_term4 = -Y(X^i) E_i(Z^l)
  ###################
  assert jnp.allclose(-B_term1a.value, C_term4.value)

  ###################
  # A_term1b - B_term1b = Y^a X^b E_b(E_a(Z^l)) - X^a Y^b E_b(E_a(Z^l))
  #                     = X^a Y^b (E_b(E_a(Z^l)) - E_a(E_b(Z^l)))
  #                     = X^a Y^b c^l_{ab} E_l(Z^l)
  # C_term1 = c^k_{ij} X^i Y^j E_k(Z^l)
  ###################
  assert jnp.allclose(A_term1b.value - B_term1b.value, C_term1.value)

  ###################
  # A_term2b = Gamma^l_{ij} X(Y^i) Z^j
  # C_term5 = X(Y^j) Gamma^l_{jm} Z^m
  ###################
  assert jnp.allclose(A_term2b.value, C_term5.value)

  ###################
  # -B_term2b = -Gamma^l_{ij} Y(X^i) Z^j
  # C_term6 = -Y(X^i) Gamma^l_{im} Z^m
  ###################
  assert jnp.allclose(-B_term2b.value, C_term6.value)

  ###################
  # A_term2c = Gamma^l_{ij} Y^i X(Z^j)
  # B_term3 = X(Z^k) Gamma^l_{ik} Y^i
  ###################
  assert jnp.allclose(A_term2c.value, B_term3.value)

  ###################
  # A_term3 = Y(Z^k) Gamma^l_{ik} X^i
  # B_term2c = Gamma^l_{ij} Y^i X(Z^j)
  ###################
  assert jnp.allclose(A_term3.value, B_term2c.value)

  ###################
  # Remaining terms
  # -, -, A_term2a, -, -, -, A_term4
  # -, -, B_term2a, -, -, -, B_term4
  # -, C_term2, -, -, -, -
  ###################
  ###################
  R_XYZ_comp = A_term2a - B_term2a + A_term4 - B_term4 - C_term2
  assert jnp.allclose(R_XYZ_comp.value, R_XYZ.value)

def test_curvature_isolated_issue():
  key = random.PRNGKey(42)
  dim = 2
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)
  basis = connection.basis
  frame = basis_to_frame(basis)
  lie_bracket_pairs: Annotated[TangentVector, "i j"] = get_lie_bracket_between_frame_pairs(frame)
  c = lie_bracket_pairs.components
  Gamma = connection.christoffel_symbols

  k1, k2, k3 = random.split(key, 3)
  X = change_basis(create_random_vector_field(k1, dim), connection.basis)
  Y = change_basis(create_random_vector_field(k2, dim), connection.basis)
  Z = change_basis(create_random_vector_field(k3, dim), connection.basis)

  # Y^a X^b E_b(E_a(Z^l))
  @jet_decorator
  def EaZl_computation(E_val, Z_grad):
    return jnp.einsum("ba,lb->al", E_val, Z_grad)
  EaZl = EaZl_computation(basis.components, Z.components.get_gradient_jet())

  @jet_decorator
  def term1b_computation(Y_val, X_val, E_val, EaZl_grad):
    return jnp.einsum("a,b,cb,alc->l", Y_val, X_val, E_val, EaZl_grad)
  A_term1b = term1b_computation(
    Y.components,
    X.components,
    basis.components,
    EaZl.get_gradient_jet()
  )

  B_term1b = term1b_computation(
    X.components,
    Y.components,
    basis.components,
    EaZl.get_gradient_jet()
  )

  # c^k_{ij} X^i Y^j E_k(Z^l)
  @jet_decorator
  def term1_computation(c_val, X_val, Y_val, E_val, Z_grad):
    return jnp.einsum("ijk,i,j,ak,la->l", c_val, X_val, Y_val, E_val, Z_grad)
  C_term1 = term1_computation(
    c,
    X.components,
    Y.components,
    basis.components,
    Z.components.get_gradient_jet(),
  )

  assert jnp.allclose(A_term1b.value - B_term1b.value, C_term1.value)


def test_curvature_isolated_issue2():
  key = random.PRNGKey(42)
  dim = 2
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)
  basis = connection.basis
  frame = basis_to_frame(basis)
  lie_bracket_pairs: Annotated[TangentVector, "i j"] = get_lie_bracket_between_frame_pairs(frame)
  c = lie_bracket_pairs.components
  Gamma = connection.christoffel_symbols
  E1 = frame.get_basis_vector(0)
  E2 = frame.get_basis_vector(1)
  bracket_E1E2 = lie_bracket(E1, E2)

  k1, k2, k3 = random.split(key, 3)
  X = change_basis(create_random_vector_field(k1, dim), connection.basis)
  Y = change_basis(create_random_vector_field(k2, dim), connection.basis)
  Z = change_basis(create_random_vector_field(k3, dim), connection.basis)

  # E_b(E_a(Z^l)) - E_a(E_b(Z^l)) = [E_a,E_b](Z^l)
  @jet_decorator
  def EaZl_computation(E_val, Z_grad):
    return jnp.einsum("ba,lb->al", E_val, Z_grad)
  EaZl = EaZl_computation(basis.components, Z.components.get_gradient_jet())

  @jet_decorator
  def A_term1_computation(E_val, EaZl_grad):
    out = jnp.einsum("cb,alc->abl", E_val, EaZl_grad)
    return out, jnp.einsum("abl->bal", out)
  A_term1b, B_term1b = A_term1_computation(
    basis.components,
    EaZl.get_gradient_jet()
  )

  # c^k_{ij} E_k(Z^l) = [E_i,E_j](Z^l)
  @jet_decorator
  def C_term1_computation(c_val, E_val, Z_grad):
    return jnp.einsum("ijk,ak,la->ijl", c_val, E_val, Z_grad)
  C_term1 = C_term1_computation(
    c,
    basis.components,
    Z.components.get_gradient_jet(),
  )
  assert jnp.allclose(B_term1b.value - A_term1b.value, C_term1.value)


def test_curvature_isolated_issue3():
  key = random.PRNGKey(42)
  dim = 2
  p = jnp.array([0.0, 0.0])
  basis = get_standard_basis(p)
  frame = basis_to_frame(basis)

  E1 = frame.get_basis_vector(0)
  E2 = frame.get_basis_vector(1)
  bracket_E1E2 = lie_bracket(E1, E2)

  Z = change_basis(create_random_vector_field(key, dim), basis)

  # E_b(E_a(Z^l)) - E_a(E_b(Z^l)) = [E_a,E_b](Z^l)
  lhs = E1(E2(Z.components)) - E2(E1(Z.components))

  # c^k_{ij} E_k(Z^l) = [E_i,E_j](Z^l)
  rhs = bracket_E1E2(Z.components)
  assert jnp.allclose(lhs.value, rhs.value)


def test_bracket_covariant_derivative():
  key = random.PRNGKey(42)
  dim = 2
  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)
  basis = connection.basis

  frame = basis_to_frame(basis)
  lie_bracket_pairs: Annotated[TangentVector, "i j"] = get_lie_bracket_between_frame_pairs(frame)
  c = lie_bracket_pairs.components
  Gamma = connection.christoffel_symbols

  k1, k2, k3 = random.split(key, 3)
  X = change_basis(create_random_vector_field(k1, dim), connection.basis)
  Y = change_basis(create_random_vector_field(k2, dim), connection.basis)
  Z = change_basis(create_random_vector_field(k3, dim), connection.basis)

  # Ground truth for comparison
  bracket_XY = lie_bracket(X, Y)
  nabla_bracket_XY_Z_truth = connection.covariant_derivative(bracket_XY, Z)

  ###################
  # ∇_{[X,Y]} Z
  ###################
  C_term1, C_term2, C_term3, C_term4, C_term5, C_term6 = get_bracket_covariant_derivative_terms(connection, X, Y, Z)
  nabla_bracket_XY_Z = C_term1 + C_term2 + C_term3 + C_term4 + C_term5 + C_term6
  # assert jnp.allclose(nabla_bracket_XY_Z.value, nabla_bracket_XY_Z_truth.components.value) # Fails!

  #####################################################
  # [X,Y] =
  # Term 1:        c^k_{ij} X^i Y^j E_k
  # Term 2:      + X(Y^j) E_j
  # Term 3:      - Y(X^i) E_i
  #####################################################
  @jet_decorator
  def term1_computation(c_val, X_val, Y_val):
    return jnp.einsum("ijk,i,j->k", c_val, X_val, Y_val)
  term1 = term1_computation(
    c,
    X.components,
    Y.components,
  )
  term2 = X(Y.components)
  term3 = Y(X.components)

  lhs = term1 + term2 - term3
  assert jnp.allclose(lhs.value, bracket_XY.components.value)
  assert jnp.allclose(lhs.gradient, bracket_XY.components.gradient)

  vec1 = TangentVector(p=basis.p, components=term1, basis=basis)
  vec2 = TangentVector(p=basis.p, components=term2, basis=basis)
  vec3 = TangentVector(p=basis.p, components=term3, basis=basis)

  #####################################################
  # ∇_{c^k_{ij} X^i Y^j E_k} Z = [
  # Term 1:        c^k_{ij} X^i Y^j E_k(Z^l)
  # Term 2:      + c^k_{ij} X^i Y^j Gamma^l_{mk} Z^m
  #                                                         ] E_l
  #####################################################
  nabla1Z = connection.covariant_derivative(vec1, Z)

  rhs = C_term1 + C_term2
  assert jnp.allclose(nabla1Z.components.value, rhs.value)
  assert jnp.allclose(nabla1Z.components.gradient, rhs.gradient)
  # THIS EXPOSED THE BUG!  THE BUG WAS THE INDEX ORDER FOR GAMMA IN C_term2 WAS swapped!
