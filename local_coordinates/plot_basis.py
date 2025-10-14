import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, Tuple, List, Callable, Dict

from local_coordinates.basis import BasisVectors
from local_coordinates.jet import Jet, function_to_jet


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


# -------------------------
# Modular plotting helpers
# -------------------------

def compute_coordinate_grid_curves(
  basis: BasisVectors,
  *,
  span: float = 0.5,
  num: int = 21,
) -> Dict[str, List[Tuple[jnp.ndarray, jnp.ndarray]]]:
  """
  Compute grid curves for first two tangent coordinates and basis info.

  Returns a dict with keys:
    - 'u_lines': list of (xs, ys) for fixed u, varying v
    - 'v_lines': list of (xs, ys) for fixed v, varying u
    - 'basepoint': (x0, y0)
    - 'basis_vectors_2d': (v1[:2], v2[:2]) from basis.components.value
  """
  jet: Jet = Jet(
    value=basis.p,
    gradient=basis.components.value,
    hessian=basis.components.gradient,
  )

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
    U = jnp.zeros((ts.shape[0], dim))
    if along_u:
      if dim > 0:
        U = U.at[:, 0].set(fixed_val)
      if dim > 1:
        U = U.at[:, 1].set(ts)
    else:
      if dim > 1:
        U = U.at[:, 1].set(fixed_val)
      if dim > 0:
        U = U.at[:, 0].set(ts)

    Y = jax.vmap(lambda w: jet(w))(U)
    y_leaves = jtu.tree_leaves(Y)
    if len(y_leaves) == 0:
      raise ValueError("Jet output produced no array leaves.")
    Y0 = y_leaves[0]

    if Y0.ndim == 1:
      x = Y0[0]
      y = (Y0[1] if Y0.shape[0] > 1 else jnp.zeros((), dtype=Y0.dtype))
      xs = jnp.broadcast_to(x, ts.shape)
      ys = jnp.broadcast_to(y, ts.shape)
    else:
      xs = Y0[:, 0]
      ys = Y0[:, 1] if Y0.shape[1] > 1 else jnp.zeros_like(xs)
    return jnp.array(xs), jnp.array(ys)

  u_lines: List[Tuple[jnp.ndarray, jnp.ndarray]] = []
  v_lines: List[Tuple[jnp.ndarray, jnp.ndarray]] = []
  for u in uvs:
    u_lines.append(eval_line(True, float(u)))
  for v in uvs:
    v_lines.append(eval_line(False, float(v)))

  x0 = float(basis.p[0])
  y0 = float(basis.p[1]) if basis.p.shape[0] > 1 else 0.0
  B = basis.components.value
  v1 = B[:2, 0] if B.ndim >= 2 and B.shape[1] >= 1 else jnp.zeros((2,))
  v2 = B[:2, 1] if B.ndim >= 2 and B.shape[1] >= 2 else jnp.zeros((2,))
  return {
    "u_lines": u_lines,
    "v_lines": v_lines,
    "basepoint": (x0, y0),
    "basis_vectors_2d": (jnp.array(v1), jnp.array(v2)),
  }


def draw_coordinate_grid_on_axes(
  ax: plt.Axes,
  curves: Dict[str, List[Tuple[jnp.ndarray, jnp.ndarray]]],
  *,
  high_contrast: bool = True,
  u_linestyle: str = "--",
  v_linestyle: str = "-",
  u_color: str = "tab:blue",
  v_color: str = "tab:orange",
  basepoint_facecolor: str = "w",
  basepoint_edgecolor: str = "k",
  basepoint_size: float = 50.0,
  draw_basis_vectors: bool = True,
  basis_vector_scale: float = 1.5,
):
  for xs, ys in curves["u_lines"]:
    if high_contrast:
      ax.plot(xs, ys, color='k', linestyle=u_linestyle, linewidth=2.0, alpha=0.95, zorder=4)
      ax.plot(xs, ys, color='w', linestyle=u_linestyle, linewidth=1.0, alpha=0.95, zorder=5)
    else:
      ax.plot(xs, ys, color=u_color, linestyle=u_linestyle, linewidth=1.0, alpha=0.9, zorder=5)
  for xs, ys in curves["v_lines"]:
    if high_contrast:
      ax.plot(xs, ys, color='k', linestyle=v_linestyle, linewidth=2.0, alpha=0.95, zorder=4)
      ax.plot(xs, ys, color='w', linestyle=v_linestyle, linewidth=1.0, alpha=0.95, zorder=5)
    else:
      ax.plot(xs, ys, color=v_color, linestyle=v_linestyle, linewidth=1.0, alpha=0.9, zorder=5)

  x0, y0 = curves["basepoint"]
  ax.scatter([x0], [y0], s=basepoint_size, facecolors=basepoint_facecolor, edgecolors=basepoint_edgecolor, linewidths=1.0, zorder=6)

  if draw_basis_vectors and "basis_vectors_2d" in curves:
    v1, v2 = curves["basis_vectors_2d"]
    # Always visible blue arrows (point-sized heads)
    import matplotlib.patches as mpatches
    for v in (v1, v2):
      dx = float(basis_vector_scale * v[0])
      dy = float(basis_vector_scale * v[1])
      arrow = mpatches.FancyArrowPatch(
        (x0, y0), (x0 + dx, y0 + dy),
        arrowstyle='-|>', mutation_scale=14,
        linewidth=1.5, color='red', zorder=8,
      )
      ax.add_patch(arrow)
