import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, Tuple, List, Callable, Dict, Union

from local_coordinates.basis import BasisVectors
from local_coordinates.jet import Jet, function_to_jet
from local_coordinates.curved_flow import SecondOrderFlow, ThirdOrderFlow

def plot_coordinate_grid(
  basis: BasisVectors,
  num: int = 21,
  span: float = 0.2,
  savepath: Optional[str] = None,
  title: Optional[str] = None,
  show: bool = True,
  draw_basis_vectors: bool = True,
  basis_vector_scale: float = 1.0,
):
  """
  Plot a 2D slice of the coordinate grid for a BasisVectors frame using its
  second-order Taylor approximation via the underlying Jet.

  - Varies the first two tangent coordinates (others fixed to 0).
  - If ambient output has >2 components, only the first two are plotted.
  """
  jet: Jet = Jet(
      value=basis.p,
      gradient=basis.components.value,
      hessian=basis.components.gradient,
  )

  # Infer input dim from a gradient leaf
  grad_leaves = jtu.tree_leaves(jet.gradient)
  if len(grad_leaves) == 0:
    raise ValueError("Jet.gradient has no array leaves; cannot infer input dimension.")
  g0 = grad_leaves[0]
  if g0.ndim < 1:
    raise ValueError("Gradient leaf must have at least 1 trailing input-dimension axis.")
  dim = int(g0.shape[-1])

  uvs = jnp.linspace(-span, span, num)

  def eval_line(along_u: bool, fixed_val: float):
    ts = uvs
    # Build a batch of displacements U in R^dim
    U = jnp.zeros((ts.shape[0], dim))
    if along_u:
      # vary v (2nd coord), fix u (1st coord)
      if dim > 0:
        U = U.at[:, 0].set(fixed_val)
      if dim > 1:
        U = U.at[:, 1].set(ts)
    else:
      # vary u (1st coord), fix v (2nd coord)
      if dim > 1:
        U = U.at[:, 1].set(fixed_val)
      if dim > 0:
        U = U.at[:, 0].set(ts)

    Y = jax.vmap(lambda w: jet(w))(U)
    y_leaves = jtu.tree_leaves(Y)
    if len(y_leaves) == 0:
      raise ValueError("Jet output produced no array leaves.")
    Y0 = y_leaves[0]

    # Project to first two ambient components for plotting
    if Y0.ndim == 1:
      x = Y0[0]
      y = (Y0[1] if Y0.shape[0] > 1 else jnp.zeros((), dtype=Y0.dtype))
      xs = jnp.broadcast_to(x, ts.shape)
      ys = jnp.broadcast_to(y, ts.shape)
    else:
      xs = Y0[:, 0]
      ys = Y0[:, 1] if Y0.shape[1] > 1 else jnp.zeros_like(xs)
    return jnp.array(xs), jnp.array(ys)

  fig, ax = plt.subplots(figsize=(6, 6))
  for u in uvs:
    xs, ys = eval_line(True, float(u))
    ax.plot(xs, ys, color='tab:blue', linestyle='--', linewidth=1.0, alpha=0.85)
  for v in uvs:
    xs, ys = eval_line(False, float(v))
    ax.plot(xs, ys, color='tab:orange', linestyle='--', linewidth=1.0, alpha=0.85)

  # Basepoint (first two components)
  val_leaves = jtu.tree_leaves(jet.value)
  if len(val_leaves) > 0:
    V0 = val_leaves[0]
    if V0.ndim >= 1 and V0.shape[0] >= 1:
      x0 = float(V0[0])
      y0 = float(V0[1]) if V0.shape[0] > 1 else 0.0
      ax.scatter([x0], [y0], c='k', s=20, zorder=5, label='basepoint')

      # Optionally draw the basis vector directions at p (project to first two ambient components)
      if draw_basis_vectors:
        B = basis.components.value  # shape (ambient_dim, tangent_dim)
        if B.ndim >= 2:
          # First two basis vectors (columns) projected onto first two ambient axes
          v1 = B[:2, 0] if B.shape[1] >= 1 else jnp.zeros((2,))
          v2 = B[:2, 1] if B.shape[1] >= 2 else jnp.zeros((2,))
          # Draw thin, visible blue arrows with point-sized heads (independent of data scale)
          for dx, dy in [(float(basis_vector_scale * v1[0]), float(basis_vector_scale * v1[1])),
                         (float(basis_vector_scale * v2[0]), float(basis_vector_scale * v2[1]))]:
            arrow = mpatches.FancyArrowPatch(
              (x0, y0), (x0 + dx, y0 + dy),
              arrowstyle='-|>',
              mutation_scale=12,  # head size in points
              linewidth=1.5,
              color='red',
              zorder=7,
            )
            ax.add_patch(arrow)

  ax.set_aspect('equal', 'box')
  ax.set_title('Coordinate grid (Taylor approx)' if title is None else title)
  ax.grid(True, linestyle=':', alpha=0.3)
  ax.legend(loc='upper right')
  if savepath is not None:
    fig.savefig(savepath, bbox_inches='tight', dpi=150)
  else:
    if show:
      plt.show()
  return fig, ax

def plot_flow_grid(
  flow: Union[SecondOrderFlow, ThirdOrderFlow],
  num: int = 21,
  span: float = 0.2,
  savepath: Optional[str] = None,
  title: Optional[str] = None,
  show: bool = True,
  draw_basis_vectors: bool = True,
  basis_vector_scale: float = 1.0,
):
  """
  Plot a 2D slice of the image of a grid under a flow x(z).

  - Varies the first two input coordinates (others fixed to 0).
  - If output has >2 components, only the first two are plotted.
  """
  dim = flow.J.shape[-1]
  uvs = jnp.linspace(-span, span, num)

  def eval_line(along_u: bool, fixed_val: float):
    ts = uvs
    # Build a batch of displacements U in R^dim
    U = jnp.zeros((ts.shape[0], dim))
    if along_u:
      # vary v (2nd coord), fix u (1st coord)
      if dim > 0:
        U = U.at[:, 0].set(fixed_val)
      if dim > 1:
        U = U.at[:, 1].set(ts)
    else:
      # vary u (1st coord), fix v (2nd coord)
      if dim > 1:
        U = U.at[:, 1].set(fixed_val)
      if dim > 0:
        U = U.at[:, 0].set(ts)

    Y = jax.vmap(lambda w: flow(w))(U)  # shape (len(ts), out_dim)
    # Project to first two output components for plotting
    if Y.ndim == 1:
      x = Y[0]
      y = (Y[1] if Y.shape[0] > 1 else jnp.zeros((), dtype=Y.dtype))
      xs = jnp.broadcast_to(x, ts.shape)
      ys = jnp.broadcast_to(y, ts.shape)
    else:
      xs = Y[:, 0]
      ys = Y[:, 1] if Y.shape[1] > 1 else jnp.zeros_like(xs)
    return jnp.array(xs), jnp.array(ys)

  fig, ax = plt.subplots(figsize=(6, 6))
  for u in uvs:
    xs, ys = eval_line(True, float(u))
    ax.plot(xs, ys, color='tab:blue', linestyle='--', linewidth=1.0, alpha=0.85)
  for v in uvs:
    xs, ys = eval_line(False, float(v))
    ax.plot(xs, ys, color='tab:orange', linestyle='--', linewidth=1.0, alpha=0.85)

  # Basepoint (image of 0 under the flow) and optionally draw Jacobian columns at 0
  z0 = jnp.zeros((dim,))
  y0 = flow(z0)
  x0 = float(y0[0]) if y0.shape[0] >= 1 else 0.0
  y0v = float(y0[1]) if y0.shape[0] > 1 else 0.0
  ax.scatter([x0], [y0v], c='k', s=20, zorder=5, label='basepoint')

  if draw_basis_vectors:
    # Use the Jacobian at 0, which is J for these flows
    B = flow.J
    if B.ndim >= 2:
      v1 = B[:2, 0] if B.shape[1] >= 1 else jnp.zeros((2,))
      v2 = B[:2, 1] if B.shape[1] >= 2 else jnp.zeros((2,))
      for dx, dy in [
        (float(basis_vector_scale * v1[0]), float(basis_vector_scale * v1[1])),
        (float(basis_vector_scale * v2[0]), float(basis_vector_scale * v2[1])),
      ]:
        arrow = mpatches.FancyArrowPatch(
          (x0, y0v), (x0 + dx, y0v + dy),
          arrowstyle='-|>',
          mutation_scale=12,
          linewidth=1.5,
          color='red',
          zorder=7,
        )
        ax.add_patch(arrow)

  ax.set_aspect('equal', 'box')
  ax.set_title('Flow grid' if title is None else title)
  ax.grid(True, linestyle=':', alpha=0.3)
  ax.legend(loc='upper right')
  if savepath is not None:
    fig.savefig(savepath, bbox_inches='tight', dpi=150)
  else:
    if show:
      plt.show()
  return fig, ax
