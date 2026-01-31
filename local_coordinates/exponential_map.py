"""
Exponential map on Riemannian manifolds.

The exponential map exp_p: T_p M -> M takes a tangent vector v at p and returns
the point gamma(1), where gamma is the unique geodesic with gamma(0) = p and
dot{gamma}(0) = v. In Riemann normal coordinates centered at p, the exponential
map is simply exp_p(v) = v.

This module provides two implementations:
1. A Taylor approximation via the RNC Jacobian (fast, local accuracy)
2. A numerical ODE solver using diffrax (slower, globally accurate)
"""
from typing import Callable, Optional
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from local_coordinates.basis import get_standard_basis
from local_coordinates.jet import Jet
from local_coordinates.metric import RiemannianMetric
from local_coordinates.connection import get_levi_civita_connection
from local_coordinates.normal_coords import get_rnc_jacobians
from local_coordinates.jacobian import Jacobian
from local_coordinates.tangent import TangentVector
import diffrax


def exponential_map_taylor(
  metric: RiemannianMetric,
  v: TangentVector,
  J_v_to_x: Optional[Jacobian] = None,
) -> Float[Array, "N"]:
  """
  Compute the exponential map via third-order Taylor approximation.

  This uses the RNC Jacobian J_v_to_x to compute:
    x(v) = p + J.value @ v + 0.5 * einsum(J.gradient, v, v) + (1/6) * einsum(J.hessian, v, v, v)

  This is equivalent to approximating the geodesic gamma(t) = x(tv) at t=1.

  Args:
    metric: The Riemannian metric at the base point p.
    v: The tangent vector at p.
    J_v_to_x: Optional pre-computed RNC Jacobian. If None, it will be computed.

  Returns:
    The point exp_p(v) in the original x-coordinates.
  """
  p = metric.basis.p

  # Compute RNC Jacobians
  J_x_to_v, J_v_to_x_computed = get_rnc_jacobians(metric)
  if J_v_to_x is None:
    J_v_to_x = J_v_to_x_computed

  # Convert TangentVector to RNC components
  v_std = v.to_standard_basis()
  v_std_components = v_std.components.value

  # Transform from standard coordinates to RNC: v_rnc = J_x_to_v @ v_std
  v_components = J_x_to_v.value @ v_std_components

  # x(v) = p + J*v + 0.5*H*v*v + (1/6)*T*v*v*v
  term1 = jnp.einsum("ij,j->i", J_v_to_x.value, v_components)
  term2 = 0.5 * jnp.einsum("ijk,j,k->i", J_v_to_x.gradient, v_components, v_components)

  if J_v_to_x.hessian is not None:
    term3 = (1/6) * jnp.einsum("ijkl,j,k,l->i", J_v_to_x.hessian, v_components, v_components, v_components)
  else:
    term3 = 0.0

  return p + term1 + term2 + term3

def exponential_map_ode(
  v: TangentVector,
  metric_fn: Callable[[Array], RiemannianMetric],
  num_steps: int = 100,
  return_trajectory: bool = False,
):
  """
  Compute the exponential map by numerically solving the geodesic ODE.

  Solves the geodesic equation:
    ddot{gamma}^i + Gamma^i_{jk}(gamma) dot{gamma}^j dot{gamma}^k = 0

  with initial conditions gamma(0) = p and dot{gamma}(0) = v.

  The computation is performed in the standard coordinate basis. The input
  tangent vector is automatically converted to standard coordinates.

  Args:
    v: The initial velocity as a TangentVector at point p.
    metric_fn: Function x -> RiemannianMetric(x) that computes the metric at any point.
    num_steps: Number of ODE solver steps.
    return_trajectory: If True, return the full trajectory instead of just the endpoint.

  Returns:
    If return_trajectory is False:
      The point exp_p(v) = gamma(1) in standard coordinates, shape (N,).
    If return_trajectory is True:
      A tuple (ts, trajectory) where ts has shape (num_steps,) and
      trajectory has shape (num_steps, N).
  """
  # Convert tangent vector to standard coordinates
  v_std = v.to_standard_basis()
  p = v_std.p
  v_components = v_std.components.value
  dim = p.shape[0]

  # Convert second-order ODE to first-order system
  # State y = [gamma, dot_gamma] of shape (2*dim,)
  def geodesic_vector_field(t, y, args):
    gamma = y[:dim]
    dot_gamma = y[dim:]

    metric = metric_fn(gamma)
    connection = get_levi_civita_connection(metric)
    Gamma = connection.christoffel_symbols.value

    # ddot_gamma^i = -Gamma^i_{jk} * dot_gamma^j * dot_gamma^k
    ddot_gamma = -jnp.einsum("jki,j,k->i", Gamma, dot_gamma, dot_gamma)
    return jnp.concatenate([dot_gamma, ddot_gamma])

  # Initial state
  y0 = jnp.concatenate([p, v_components])

  # Solve ODE from t=0 to t=1
  term = diffrax.ODETerm(geodesic_vector_field)
  solver = diffrax.Dopri5()
  stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)

  if return_trajectory:
    ts = jnp.linspace(0.0, 1.0, num_steps)
    saveat = diffrax.SaveAt(ts=ts)
  else:
    saveat = diffrax.SaveAt(t1=True)

  sol = diffrax.diffeqsolve(
    term,
    solver,
    t0=0.0,
    t1=1.0,
    dt0=0.1,
    y0=y0,
    saveat=saveat,
    stepsize_controller=stepsize_controller,
    max_steps=4096*2,
    throw=False
  )

  if return_trajectory:
    # Extract positions from all saved states
    trajectory = sol.ys[:, :dim]  # shape (num_steps, dim)
    return sol.ts, trajectory
  else:
    # Extract final position
    y1 = sol.ys[0]  # shape (2*dim,)
    return y1[:dim]


def exponential_map(
  metric: RiemannianMetric,
  v: TangentVector,
  method: str = "taylor",
  **kwargs,
) -> Float[Array, "N"]:
  """
  Compute the exponential map exp_p(v).

  Args:
    metric: The Riemannian metric at the base point p.
    v: The tangent vector at p.
    method: Either 'taylor' (fast, local) or 'ode' (slower, globally accurate).
    **kwargs: Additional arguments passed to the specific implementation.
              For 'ode', must include 'metric_fn: Callable[[Array], RiemannianMetric]'.

  Returns:
    The point exp_p(v) in x-coordinates.
  """
  if method == "taylor":
    return exponential_map_taylor(metric, v, **kwargs)
  elif method == "ode":
    if "metric_fn" not in kwargs:
      raise ValueError("exponential_map with method='ode' requires 'metric_fn' argument.")
    return exponential_map_ode(v, **kwargs)
  else:
    raise ValueError(f"Unknown method: {method}. Use 'taylor' or 'ode'.")


def logarithmic_map_taylor(
  metric: RiemannianMetric,
  q: Float[Array, "N"],
  J_x_to_v: Optional[Jacobian] = None,
) -> Float[Array, "N"]:
  """
  Compute the logarithmic map (inverse of exponential map) via Taylor approximation.

  Given a point q near p, compute v = log_p(q) such that exp_p(v) = q.

  This uses the inverse RNC Jacobian J_x_to_v to compute:
    v(x) = J.value @ (x - p) + 0.5 * einsum(J.gradient, x-p, x-p) + ...

  Args:
    metric: The Riemannian metric at the base point p.
    q: The target point in x-coordinates.
    J_x_to_v: Optional pre-computed inverse RNC Jacobian. If None, it will be computed.

  Returns:
    The tangent vector v = log_p(q) in RNC.
  """
  p = metric.basis.p
  dx = q - p

  if J_x_to_v is None:
    J_x_to_v, _ = get_rnc_jacobians(metric)

  # v(x) = J*dx + 0.5*H*dx*dx + (1/6)*T*dx*dx*dx
  term1 = jnp.einsum("ij,j->i", J_x_to_v.value, dx)
  term2 = 0.5 * jnp.einsum("ijk,j,k->i", J_x_to_v.gradient, dx, dx)

  if J_x_to_v.hessian is not None:
    term3 = (1/6) * jnp.einsum("ijkl,j,k,l->i", J_x_to_v.hessian, dx, dx, dx)
  else:
    term3 = 0.0

  return term1 + term2 + term3
