"""
Sanity check script for compute-curvature skill.

Verifies key invariants of curvature computations:
1. Riemann tensor definition: R(X,Y)Z = nabla_X nabla_Y Z - nabla_Y nabla_X Z - nabla_[X,Y] Z
2. Riemann tensor symmetries (skew, interchange, Bianchi)
3. Flat metric has zero curvature

Run with: uv run python .cursor/skills/compute-curvature/scripts/sanity_check.py
"""

import jax
import jax.numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)

from local_coordinates.metric import RiemannianMetric, lower_index
from local_coordinates.basis import get_standard_basis, BasisVectors
from local_coordinates.jet import function_to_jet, Jet, jet_decorator, get_identity_jet
from local_coordinates.connection import get_levi_civita_connection
from local_coordinates.riemann import get_riemann_curvature_tensor, get_ricci_tensor
from local_coordinates.tangent import TangentVector, lie_bracket
from local_coordinates.tensor import change_basis
from jaxtyping import Array


def create_random_basis(key, dim):
  """Create a random basis for testing."""
  vals_key, grads_key, hessians_key = random.split(key, 3)
  p = jnp.zeros(dim)
  vals = jnp.eye(dim) + random.normal(vals_key, (dim, dim)) * 0.1
  grads = random.normal(grads_key, (dim, dim, dim)) * 0.1
  hessians = random.normal(hessians_key, (dim, dim, dim, dim)) * 0.1
  return BasisVectors(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians))


def create_random_metric(key, dim):
  """Create a random metric for testing."""
  random_basis = create_random_basis(key, dim)
  return RiemannianMetric(basis=random_basis, components=get_identity_jet(dim))


def create_random_vector_field(key, dim, basis):
  """Create a random tangent vector in given basis."""
  vals_key, grads_key, hessians_key = random.split(key, 3)
  p = jnp.zeros(dim)
  vals = random.normal(vals_key, (dim,))
  grads = random.normal(grads_key, (dim, dim))
  hessians = random.normal(hessians_key, (dim, dim, dim))
  return TangentVector(p=p, components=Jet(value=vals, gradient=grads, hessian=hessians), basis=basis)


def check_riemann_definition():
  """Verify Riemann tensor satisfies its definition."""
  print("Checking Riemann tensor definition... ", end="")

  key = random.PRNGKey(42)
  dim = 3

  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)

  k1, k2, k3 = random.split(key, 3)
  X = create_random_vector_field(k1, dim, connection.basis)
  Y = create_random_vector_field(k2, dim, connection.basis)
  Z = create_random_vector_field(k3, dim, connection.basis)

  # Compute R(X,Y)Z via definition
  nablaY_Z = connection.covariant_derivative(Y, Z)
  nablaX_Z = connection.covariant_derivative(X, Z)
  bracket_XY = lie_bracket(X, Y)
  nablaX_nablaY_Z = connection.covariant_derivative(X, nablaY_Z)
  nablaY_nablaX_Z = connection.covariant_derivative(Y, nablaX_Z)
  nabla_bracket_XY_Z = connection.covariant_derivative(bracket_XY, Z)

  R_XYZ_def = nablaX_nablaY_Z.components.value - nablaY_nablaX_Z.components.value - nabla_bracket_XY_Z.components.value

  # Compute via Riemann tensor
  riemann = get_riemann_curvature_tensor(connection)

  @jet_decorator
  def apply_riemann(R_val: Array, X_val: Array, Y_val: Array, Z_val: Array) -> Array:
    return jnp.einsum("ijkl,i,j,k->l", R_val, X_val, Y_val, Z_val)

  R_XYZ_tensor = apply_riemann(
    riemann.components.get_value_jet(),
    X.components.get_value_jet(),
    Y.components.get_value_jet(),
    Z.components.get_value_jet()
  )

  assert jnp.allclose(R_XYZ_def, R_XYZ_tensor.value, atol=1e-10), \
    f"Definition mismatch: {jnp.max(jnp.abs(R_XYZ_def - R_XYZ_tensor.value))}"

  print("PASSED")


def check_riemann_symmetries():
  """Verify Riemann tensor symmetries."""
  print("Checking Riemann tensor symmetries... ", end="")

  key = random.PRNGKey(42)
  dim = 4

  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)
  riemann = get_riemann_curvature_tensor(connection)

  # Lower the last index to get R_{ijkl}
  R_lower = lower_index(riemann, metric, 4)
  R = R_lower.components.value

  # Skew symmetry in first pair: R_{ijkl} = -R_{jikl}
  assert jnp.allclose(R, -R.swapaxes(0, 1), atol=1e-10), "Skew symmetry 1 failed"

  # Skew symmetry in second pair: R_{ijkl} = -R_{ijlk}
  assert jnp.allclose(R, -R.swapaxes(-1, -2), atol=1e-10), "Skew symmetry 2 failed"

  # Interchange symmetry: R_{ijkl} = R_{klij}
  assert jnp.allclose(R, R.transpose((2, 3, 0, 1)), atol=1e-10), "Interchange symmetry failed"

  # First Bianchi identity: R_{ijkl} + R_{iklj} + R_{iljk} = 0
  bianchi = R + R.transpose((0, 2, 3, 1)) + R.transpose((0, 3, 1, 2))
  assert jnp.allclose(bianchi, 0.0, atol=1e-10), "Bianchi identity failed"

  print("PASSED")


def check_flat_metric_zero_curvature():
  """Verify flat (Euclidean) metric has zero curvature."""
  print("Checking flat metric has zero curvature... ", end="")

  dim = 3
  p = jnp.zeros(dim)
  basis = get_standard_basis(p)

  # Euclidean metric with derivatives
  metric = RiemannianMetric(
    basis=basis,
    components=Jet(
      value=jnp.eye(dim),
      gradient=jnp.zeros((dim, dim, dim)),
      hessian=jnp.zeros((dim, dim, dim, dim))
    )
  )

  connection = get_levi_civita_connection(metric)
  riemann = get_riemann_curvature_tensor(connection)

  # All components should be zero
  assert jnp.allclose(riemann.components.value, 0.0, atol=1e-14), \
    f"Flat metric has non-zero curvature: max = {jnp.max(jnp.abs(riemann.components.value))}"

  print("PASSED")


def check_ricci_symmetric():
  """Verify Ricci tensor is symmetric."""
  print("Checking Ricci tensor symmetry... ", end="")

  key = random.PRNGKey(42)
  dim = 4

  metric = create_random_metric(key, dim)
  connection = get_levi_civita_connection(metric)
  ricci = get_ricci_tensor(connection)

  R_ab = ricci.components.value

  # Ricci tensor should be symmetric: R_{ab} = R_{ba}
  assert jnp.allclose(R_ab, R_ab.T, atol=1e-10), "Ricci tensor not symmetric"

  print("PASSED")


def check_christoffel_vanish_flat():
  """Verify Christoffel symbols vanish for flat metric in standard basis."""
  print("Checking Christoffel symbols vanish for flat metric... ", end="")

  dim = 3
  p = jnp.zeros(dim)
  basis = get_standard_basis(p)

  metric = RiemannianMetric(
    basis=basis,
    components=Jet(
      value=jnp.eye(dim),
      gradient=jnp.zeros((dim, dim, dim)),
      hessian=jnp.zeros((dim, dim, dim, dim))
    )
  )

  connection = get_levi_civita_connection(metric)
  Gamma = connection.christoffel_symbols.value

  assert jnp.allclose(Gamma, 0.0, atol=1e-14), \
    f"Flat metric has non-zero Christoffel symbols: max = {jnp.max(jnp.abs(Gamma))}"

  print("PASSED")


if __name__ == "__main__":
  print("=" * 50)
  print("Curvature Computation Sanity Checks")
  print("=" * 50)

  check_riemann_definition()
  check_riemann_symmetries()
  check_flat_metric_zero_curvature()
  check_ricci_symmetric()
  check_christoffel_vanish_flat()

  print("=" * 50)
  print("All sanity checks PASSED")
  print("=" * 50)
