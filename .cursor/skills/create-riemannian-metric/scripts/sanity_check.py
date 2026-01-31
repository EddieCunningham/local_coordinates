"""
Sanity check script for create-riemannian-metric skill.

Verifies key invariants of Riemannian metrics:
1. Metric symmetry: g(X, Y) = g(Y, X)
2. Identity metric gives dot product: g(X, Y) = X . Y when g = I
3. Index round-trip: raise(lower(v)) = v

Run with: uv run python .cursor/skills/create-riemannian-metric/scripts/sanity_check.py
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from local_coordinates.jet import Jet
from local_coordinates.basis import BasisVectors
from local_coordinates.metric import RiemannianMetric, raise_index, lower_index
from local_coordinates.tangent import TangentVector
from local_coordinates.tensor import Tensor, TensorType


def check_metric_symmetry():
  """Verify g(X, Y) = g(Y, X)."""
  print("Checking metric symmetry... ", end="")

  p = jnp.array([0.0, 0.0])
  basis = BasisVectors(p=p, components=Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2))

  # Non-trivial symmetric metric
  g = jnp.array([[2.0, 0.5], [0.5, 1.0]])
  metric = RiemannianMetric(basis=basis, components=Jet(value=g, gradient=None, hessian=None, dim=2))

  X = TangentVector(p=p, components=Jet(value=jnp.array([1.0, 2.0]), gradient=None, hessian=None, dim=2), basis=basis)
  Y = TangentVector(p=p, components=Jet(value=jnp.array([3.0, -1.0]), gradient=None, hessian=None, dim=2), basis=basis)

  gXY = metric(X, Y)
  gYX = metric(Y, X)

  assert jnp.allclose(gXY.value, gYX.value), f"Symmetry failed: g(X,Y)={gXY.value}, g(Y,X)={gYX.value}"

  print("PASSED")


def check_identity_metric_dot_product():
  """Verify g(X, Y) = X . Y for identity metric."""
  print("Checking identity metric gives dot product... ", end="")

  p = jnp.array([0.0, 0.0])
  basis = BasisVectors(p=p, components=Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2))

  # Identity metric
  metric = RiemannianMetric(basis=basis, components=Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2))

  X_comp = jnp.array([1.0, 2.0])
  Y_comp = jnp.array([3.0, 4.0])

  X = TangentVector(p=p, components=Jet(value=X_comp, gradient=None, hessian=None, dim=2), basis=basis)
  Y = TangentVector(p=p, components=Jet(value=Y_comp, gradient=None, hessian=None, dim=2), basis=basis)

  result = metric(X, Y)
  expected = jnp.dot(X_comp, Y_comp)  # 1*3 + 2*4 = 11

  assert jnp.allclose(result.value, expected), f"Dot product failed: got {result.value}, expected {expected}"

  print("PASSED")


def check_index_round_trip():
  """Verify raise(lower(v)) = v and lower(raise(alpha)) = alpha."""
  print("Checking index round-trip... ", end="")

  p = jnp.array([0.0, 0.0])
  basis = BasisVectors(p=p, components=Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2))

  # Non-trivial metric
  g = jnp.array([[2.0, 0.5], [0.5, 1.0]])
  metric = RiemannianMetric(basis=basis, components=Jet(value=g, gradient=None, hessian=None, dim=2))

  # Test 1: Start with vector (contravariant), lower, raise
  vec = jnp.array([1.0, -1.0])
  vec_tensor = Tensor(
    tensor_type=TensorType(k=0, l=1),
    basis=basis,
    components=Jet(value=vec, gradient=None, hessian=None, dim=2)
  )

  lowered = lower_index(vec_tensor, metric, index=1)
  raised_back = raise_index(lowered, metric, index=1)

  assert jnp.allclose(raised_back.components.value, vec), \
    f"Vector round-trip failed: got {raised_back.components.value}, expected {vec}"

  # Test 2: Start with covector (covariant), raise, lower
  covec = jnp.array([2.0, 3.0])
  covec_tensor = Tensor(
    tensor_type=TensorType(k=1, l=0),
    basis=basis,
    components=Jet(value=covec, gradient=None, hessian=None, dim=2)
  )

  raised = raise_index(covec_tensor, metric, index=1)
  lowered_back = lower_index(raised, metric, index=1)

  assert jnp.allclose(lowered_back.components.value, covec), \
    f"Covector round-trip failed: got {lowered_back.components.value}, expected {covec}"

  print("PASSED")


def check_metric_components_symmetric():
  """Verify metric components matrix is symmetric."""
  print("Checking metric components are symmetric... ", end="")

  p = jnp.array([1.0, 2.0])
  basis = BasisVectors(p=p, components=Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2))

  # Create a symmetric metric
  g = jnp.array([[3.0, 1.5], [1.5, 2.0]])
  metric = RiemannianMetric(basis=basis, components=Jet(value=g, gradient=None, hessian=None, dim=2))

  g_val = metric.components.value
  assert jnp.allclose(g_val, g_val.T), f"Metric not symmetric: {g_val}"

  print("PASSED")


def check_nontrivial_metric_evaluation():
  """Verify g(X, Y) = X^i g_{ij} Y^j for non-trivial metric."""
  print("Checking non-trivial metric evaluation... ", end="")

  p = jnp.array([0.0, 0.0])
  basis = BasisVectors(p=p, components=Jet(value=jnp.eye(2), gradient=None, hessian=None, dim=2))

  g = jnp.array([[2.0, 0.5], [0.5, 1.0]])
  metric = RiemannianMetric(basis=basis, components=Jet(value=g, gradient=None, hessian=None, dim=2))

  X_comp = jnp.array([1.0, 2.0])
  Y_comp = jnp.array([3.0, 4.0])

  X = TangentVector(p=p, components=Jet(value=X_comp, gradient=None, hessian=None, dim=2), basis=basis)
  Y = TangentVector(p=p, components=Jet(value=Y_comp, gradient=None, hessian=None, dim=2), basis=basis)

  result = metric(X, Y)
  expected = jnp.einsum("i,ij,j->", X_comp, g, Y_comp)

  assert jnp.allclose(result.value, expected), f"Metric evaluation failed: got {result.value}, expected {expected}"

  print("PASSED")


if __name__ == "__main__":
  print("=" * 50)
  print("Riemannian Metric Sanity Checks")
  print("=" * 50)

  check_metric_symmetry()
  check_identity_metric_dot_product()
  check_index_round_trip()
  check_metric_components_symmetric()
  check_nontrivial_metric_evaluation()

  print("=" * 50)
  print("All sanity checks PASSED")
  print("=" * 50)
