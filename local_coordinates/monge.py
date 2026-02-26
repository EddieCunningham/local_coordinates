from typing import Any, Callable, Tuple, Annotated, Optional, List, Union
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from linsdex import AbstractBatchableObject
from plum import dispatch
from local_coordinates.jet import Jet, jet_decorator, function_to_jet
from local_coordinates.basis import BasisVectors, get_dual_basis_transform, get_standard_basis
from local_coordinates.tensor import Tensor, TensorType
from local_coordinates.tangent import TangentVector, change_basis
from local_coordinates.jacobian import function_to_jacobian
from local_coordinates.jet import get_identity_jet
from local_coordinates.metric import RiemannianMetric

def get_monge_metric(
  f: Callable[[Array], Array],
  x: Array,
) -> RiemannianMetric:
  """
  Return the Monge metric at x induced by the graph of f.

  The Monge patch is the embedding phi(x) = (x, f(x)) into R^{n+1}. The
  induced metric on the surface is the pullback of the Euclidean metric on
  R^{n+1} through phi, giving the metric tensor

    g(x) = I + nabla f nabla f^T

  where nabla f = grad(f)(x) is the gradient of f.

  Args:
    f: A differentiable scalar function of x whose graph defines the Monge
      patch embedding phi(x) = (x, f(x)).
    x: Point at which to evaluate the metric, shape (n,).

  Returns:
    RiemannianMetric at x in the standard basis, with components stored
    as a Jet (value and derivatives from function_to_jet).

  References:
    See notes/monge.md for a full derivation.
  """
  def get_metric_tensor(x: Array) -> Array:
    grad_f = jax.grad(f)(x)
    return jnp.eye(x.shape[0]) + jnp.outer(grad_f, grad_f)

  metric_jet = function_to_jet(get_metric_tensor, x)
  return RiemannianMetric(basis=get_standard_basis(x), components=metric_jet)

def get_second_fundamental_form(
  f: Callable[[Array], Array],
  x: Float[Array, "N"],
) -> Float[Array, "N N"]:
  """
  Return the matrix representing the second fundamental form at x induced by the graph of f.

  The returned matrix is the scalar second fundamental form (component matrix h), so that
  h(V, W) = V^T h W. It is given by h = Hessian(f) / sqrt(1 + ||grad f||^2).

  Args:
    f: A twice-differentiable scalar function whose graph defines the Monge patch.
    x: Point at which to evaluate, shape (N,).

  Returns:
    Second fundamental form matrix of shape (N, N).

  References:
    See notes/monge.md, Second fundamental form and Shape operator sections.
  """
  grad_f = jax.grad(f)(x)
  hess_f = jax.hessian(f)(x)
  norm_sq = jnp.vdot(grad_f, grad_f)
  return hess_f / jnp.sqrt(1.0 + norm_sq)

def get_shape_matrix(
  f: Callable[[Array], Array],
  x: Float[Array, "N"],
) -> Float[Array, "N N"]:
  """
  Return the matrix representing the shape operator at x induced by the graph of f.

  The shape operator is the linear map on the tangent space at x that encodes
  how the normal to the surface rotates as we move in each tangent direction.
  Its eigenvalues are the principal curvatures and its eigenvectors are the
  principal curvature directions.

  The closed-form expression is
    s = (1 / sqrt(1 + ||grad_f||^2)) * (I - grad_f grad_f^T / (1 + ||grad_f||^2)) @ hess_f

  which is g^{-1} @ h where g is the Monge metric and h is the scalar second
  fundamental form.

  Args:
    f: A twice-differentiable scalar function whose graph defines the Monge patch.
    x: Point at which to evaluate the shape operator, shape (N,).

  Returns:
    Shape operator matrix of shape (N, N).

  References:
    See notes/monge.md, Shape operator section.
  """
  n = x.shape[0]
  grad_f = jax.grad(f)(x)
  hess_f = jax.hessian(f)(x)
  norm_sq = jnp.vdot(grad_f, grad_f)
  g_inv = jnp.eye(n) - jnp.outer(grad_f, grad_f) / (1.0 + norm_sq)
  return (g_inv @ hess_f) / jnp.sqrt(1.0 + norm_sq)