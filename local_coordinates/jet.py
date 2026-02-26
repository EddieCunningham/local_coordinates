import string
from typing import Any, Callable, Tuple, Annotated, Optional, List, Union
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray, Scalar
from local_coordinates.base import AbstractBatchableObject, auto_vmap
from plum import dispatch
from functools import partial, wraps
import inspect
from local_coordinates.jacobian import Jacobian

# Optional sensitivity probing (used-groups detection) toggle.
# If False, all Jet argument groups are treated as used.
USE_SENSITIVITY_PROBE: bool = False

def set_jet_sensitivity_probe(enabled: bool) -> None:
  """Enable/disable used-groups sensitivity probing in jet_decorator."""
  global USE_SENSITIVITY_PROBE
  USE_SENSITIVITY_PROBE = bool(enabled)

class Jet(AbstractBatchableObject):
  """
  Jet data J[F]_p at p ∈ M in local coordinates.

  Let M be a smooth, d-dimensional manifold, let p ∈ M, and let F: M → ℝⁿ be
  smooth. Fix a local coordinate system (∂/∂x¹,…,∂/∂xᵈ) around p. The second
  order Jet of F at p is
    J[F]_p = ( F_p^k, ∂F_p^k/∂x^i, ∂²F_p^k/∂x^i∂x^j ),
  for i,j = 1,…,d and k = 1,…,n. This container stores these components per
  output leaf of F in the chosen coordinates.

  Truncated Taylor evaluation (per component k): for q near p with coordinate
  difference dx^i = (q^i - p^i),
    F(q)^k ≈ F_p^k + Σ_i (∂F_p^k/∂x^i) dx^i + Σ_{i,j} (∂²F_p^k/∂x^i∂x^j) dx^i dx^j.
  In this implementation, the gradient carries a trailing axis indexed by i and
  the Hessian carries trailing axes (i, j).

  Components
  - value: F_p (PyTree or array/scalar).
  - gradient: (∂F_p/∂x^i), i = 1,…,N; mirrors `value` with trailing axis (N).
  - hessian: (∂²F_p/∂x^i∂x^j), i,j = 1,…,N; mirrors `value` with trailing axes (N, N).

  Coordinate dimension d
  - Inferred from the trailing axis of gradient leaves (or Hessian if needed);
    all leaves must agree on d.

  Batching semantics
  - Leading axes of each value leaf are treated as batch/shape axes shared by
    gradient and hessian leaves. The `batch_size` property reports the leading shape.

  Taylor evaluation API
  - Calling a Jet as `jet(dx)` evaluates the truncated Taylor polynomial at
    displacement dx from p:
      - 0th order if `gradient` is None ⇒ returns `value`.
      - 1st order if `gradient` present and `hessian` is None.
      - 2nd order if both `gradient` and `hessian` are present.

  Structure invariants
  - If provided, `gradient` and `hessian` must be PyTrees with the exact same
    structure as `value`. For corresponding leaves (v, g) and (v, H):
      g.ndim == v.ndim + 1 and H.ndim == v.ndim + 2.

  Construction
  - From a function: use `function_to_jet(F, p)` to construct J[F]_p (or a
    PyTree of Jets) by differentiating F in coordinates at input p.
  - Manual construction is supported if the above invariants hold.

  Notes
  - Derivatives are expressed in the chosen local coordinates around p.
  - Change of coordinates (expressing J[F]_p under z-coordinates) is not implemented here.
  - It is valid to have `gradient` present while `hessian` is None; if
    `gradient` is None, `hessian` is ignored by evaluation.

  Example
  >>> import jax.numpy as jnp
  >>> from local_coordinates.jet import function_to_jet
  >>> F = lambda x: jnp.array([x[0]**2, jnp.sin(x[1])])
  >>> p = jnp.array([1.0, 0.5])
  >>> J = function_to_jet(F, p)
  >>> J.value.shape
  (2,)
  >>> J.gradient.shape  # Jacobian: (out_dim, d)
  (2, 2)
  >>> J.hessian.shape   # per-output Hessian stacked along output dim
  (2, 2, 2)
  >>> dx = jnp.array([0.1, -0.2])
  >>> approx = J(dx)  # 2nd-order Taylor if hessian available
  """
  value: Scalar
  gradient: Optional[Float[Array, "N"]]
  hessian: Optional[Float[Array, "N N"]]

  def __init__(self, value: Scalar, gradient: Optional[Float[Array, "N"]], hessian: Optional[Float[Array, "N N"]], dim: Optional[int] = None):
    self.value = value

    # Helper to create inf matching value leaves with an extra trailing axis/axes of size dim.
    def inf_like_value_with_trailing(trailing_shape):
      def make(v):
        va = jnp.asarray(v)
        return jnp.ones((*va.shape, *trailing_shape), dtype=va.dtype)*jnp.inf
      return jtu.tree_map(make, self.value)

    # If both derivatives are missing and no dim is supplied, we cannot infer the trailing size.
    if gradient is None and hessian is None and dim is None:
      raise ValueError("Must provide 'dim' when both gradient and hessian are None.")

    # Set gradient: use provided, otherwise fill only if dim is given; else leave as None.
    if gradient is not None:
      self.gradient = gradient
    else:
      self.gradient = None if dim is None else inf_like_value_with_trailing((dim,))

    # Set hessian: use provided, otherwise fill only if dim is given; else leave as None.
    if hessian is not None:
      self.hessian = jtu.tree_map(lambda x: 0.5*(x + jnp.swapaxes(x, -1, -2)), hessian)
    else:
      self.hessian = None if dim is None else inf_like_value_with_trailing((dim, dim))

  def __check_init__(self):
    # For PyTree validation, check that structures match and each leaf has correct dims
    if self.gradient is not None:
      # Check that gradient and value have same PyTree structure
      value_struct = jtu.tree_structure(self.value)
      grad_struct = jtu.tree_structure(self.gradient)
      if value_struct != grad_struct:
        raise ValueError("Gradient must have same PyTree structure as value")

      # Check each leaf has one extra dimension
      def check_grad_dims(v, g):
        if hasattr(v, 'ndim') and hasattr(g, 'ndim'):
          if g.ndim != v.ndim + 1:
            raise ValueError(f"Gradient leaf has {g.ndim} dims but value leaf has {v.ndim} dims")
      jtu.tree_map(check_grad_dims, self.value, self.gradient)

    if self.hessian is not None:
      # Check that hessian and value have same PyTree structure
      value_struct = jtu.tree_structure(self.value)
      hess_struct = jtu.tree_structure(self.hessian)
      if value_struct != hess_struct:
        raise ValueError("Hessian must have same PyTree structure as value")

      # Check each leaf has two extra dimensions
      def check_hess_dims(v, h):
        if hasattr(v, 'ndim') and hasattr(h, 'ndim'):
          if h.ndim != v.ndim + 2:
            raise ValueError(f"Hessian leaf has {h.ndim} dims but value leaf has {v.ndim} dims")
      jtu.tree_map(check_hess_dims, self.value, self.hessian)

  @property
  def shape(self) -> Tuple[int, ...]:
    return self.value.shape

  @property
  def ndim(self) -> int:
    return self.value.ndim

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    f = self.value
    if f.ndim == 0:
      return None
    elif f.ndim == 1:
      return f.shape[0]
    else:
      return f.shape

  def __call__(self, dx: Any):
    """
    Evaluate the Taylor approximation at displacement dx from the Jet's base point.

    - dx must be a 1D array of shape (dim,), where dim matches the trailing
      axis of the gradient.
    - The order of the approximation depends on which derivatives are available:
      - 0th order (value only) if gradient is None.
      - 1st order (linear) if gradient is present but hessian is None.
      - 2nd order (quadratic) if gradient and hessian are present.
    - Returns a PyTree with the same structure/shapes as `self.value`.
    """
    dx = jnp.asarray(dx)
    if dx.ndim != 1:
      raise ValueError("dx must be a 1D array of shape (dim,)")

    # 0th-order approximation: return value if no derivatives are available.
    if self.gradient is None:
      return self.value

    # 1st-order approximation: v + <g, dx>
    linear_approx = jtu.tree_map(
        lambda v, g: v + jnp.einsum('...r,r->...', g, dx),
        self.value,
        self.gradient,
    )

    if self.hessian is None:
      return linear_approx

    # 2nd-order approximation: add quadratic term: + 1/2 dx^T H dx
    def quad_term(h):
      return 0.5 * jnp.einsum('r,...rs,s->...', dx, h, dx)

    quadratic_correction = jtu.tree_map(quad_term, self.hessian)

    return jtu.tree_map(jnp.add, linear_approx, quadratic_correction)

  def __neg__(self):
    return Jet(
      value=-self.value,
      gradient=-self.gradient,
      hessian=-self.hessian
    )

  def __sub__(self, other):
    return self + (-other)

  def __add__(self, other):
    """Adds another Jet or a scalar/array to this Jet."""
    if not isinstance(other, Jet):
      # Adding a constant, derivatives are unchanged
      return Jet(
        value=jtu.tree_map(lambda x: x + other, self.value),
        gradient=self.gradient,
        hessian=self.hessian
      )

    @jet_decorator
    def _add(x, y):
      return x + y

    return _add(self, other)

  def __radd__(self, other):
    return self.__add__(other)

  def __rmul__(self, other: Scalar):
    """Multiplies this Jet by a scalar."""
    assert isinstance(other, Scalar), "Jet can only be right-multiplied by a scalar"

    # Scalar multiplication
    return Jet(
        value=jtu.tree_map(lambda x: x * other, self.value),
        gradient=jtu.tree_map(lambda x: x * other, self.gradient) if self.gradient is not None else None,
        hessian=jtu.tree_map(lambda x: x * other, self.hessian) if self.hessian is not None else None
    )

  def __mul__(self, other):
    raise TypeError("Jet can only be right-multiplied by a scalar")

  def get_value_jet(self) -> "Jet":
    return Jet(
      value=self.value,
      gradient=self.gradient,
      hessian=self.hessian
    )

  def get_gradient_jet(self) -> "Jet":
    # value is the gradient, whose last axis size equals the coordinate dim
    dim = None
    if self.gradient is not None:
      dim = jnp.asarray(self.gradient).shape[-1]
    elif self.hessian is not None:
      dim = jnp.asarray(self.hessian).shape[-1]
    return Jet(
      value=self.gradient,
      gradient=self.hessian,
      hessian=None,
      dim=dim
    )

  def get_hessian_jet(self) -> "Jet":
    # value is the hessian, whose last two axes are (N, N)
    dim = None
    if self.hessian is not None:
      dim = jnp.asarray(self.hessian).shape[-1]
    return Jet(
      value=self.hessian,
      gradient=None,
      hessian=None,
      dim=dim
    )

################################################################################################################

def get_identity_jet(N: int, dtype: Optional[jnp.dtype] = None) -> Jet:
  return Jet(
    value=jnp.eye(N, dtype=dtype),
    gradient=jnp.zeros((N, N, N), dtype=dtype),
    hessian=jnp.zeros((N, N, N, N), dtype=dtype)
  )

################################################################################################################

def function_to_jet(f: Callable[[Array], Any], x: Array) -> Jet:
  """Construct J[F]_p (second-order Jet) for F ≡ f at p ≡ x.

  Using the Jet notes notation: let M be a smooth, d-dimensional manifold and
  let F: M → Y (array or PyTree), p ∈ M with chosen local coordinates x.
  For each output leaf u, returns
    ( F_p^u, ∂F_p^u/∂x^i, ∂²F_p^u/∂x^i∂x^j ),  i,j = 1,…,d.

  Shapes
  - If a value leaf has shape S, the gradient leaf has shape S + (d,), and the
    Hessian leaf has shape S + (d, d).

  Returns
  - If f returns a single array, returns a single ``Jet`` (J[F]_p).
  - If f returns a PyTree of arrays, returns the same structure with each leaf
    replaced by a ``Jet``.

  Implementation notes
  - Uses ``jax.jacrev`` for (∂F_p/∂x^i) and ``jacfwd(jacrev)`` for (∂²F_p^/∂x^i∂x^j).
  - PyTree structures of value/gradient/hessian are matched per leaf.
  """
  values = f(x)
  grads = jax.jacrev(f)(x)
  hess = jax.jacfwd(jax.jacrev(f))(x)

  def to_jet(v, g, h):
    # Ensure gradient has one more dim than value; append trailing axes as needed
    while hasattr(g, 'ndim') and hasattr(v, 'ndim') and g.ndim < v.ndim + 1:
      g = jnp.expand_dims(g, axis=-1)
    # Ensure hessian has two more dims than value; append trailing axes as needed
    while hasattr(h, 'ndim') and hasattr(v, 'ndim') and h.ndim < v.ndim + 2:
      h = jnp.expand_dims(h, axis=-1)
    return Jet(value=v, gradient=g, hessian=h)

  return jtu.tree_map(to_jet, values, grads, hess)

################################################################################################################

def _get_gradient(T: Callable, Fp: Any, dFpdx: Any) -> Any:
  """Gradient pushforward for composition (Jet notes notation).

  Let G = T ∘ F with F(p) = Fp and T the outer map. At p,
    ∂G_p^k/∂x^i = (∂T^k/∂F^a) · (∂F_p^a/∂x^i).

  Given Fp = F(p) and dFpdx = (∂F_p/∂x^i) for i = 1, ..., d, compute the
  right-hand product by pushing each column of (∂F_p/∂x^i) through a JVP of T.
  """
  # The JVP of T(Fp) with tangent dFdx_i gives one column of the new Jacobian.
  first_jvp = lambda dFdx_i: jax.jvp(T, (Fp,), (dFdx_i,))[1]
  gradient = jax.vmap(first_jvp, in_axes=-1, out_axes=-1)(dFpdx) # vmap over all i
  return gradient

def _get_hessian_transport(T: Callable, Fp: Any, d2Fpdx2: Any) -> Any:
  """Hessian transport term for composition (Jet notes notation).

  For G = T ∘ F, one contribution at p is
    (∂T^k/∂F^a) · (∂²F_p^a/∂x^i∂x^j).
  Operationally, apply J_T(F(p)) to the 2-tensor (∂²F_p/∂x^i∂x^j) by JVP over
  both tensor axes.
  """
  # Apply JVP to each element of the Hessian tangents.
  first_jvp = lambda d2Fpdx2_ij: jax.jvp(T, (Fp,), (d2Fpdx2_ij,))[1]
  jvp_vmapped_over_cols = jax.vmap(first_jvp, in_axes=-1, out_axes=-1)
  transport = jax.vmap(jvp_vmapped_over_cols, in_axes=-2, out_axes=-2)(d2Fpdx2)
  return transport

def _get_hessian_curvature(T: Callable, Fp: Any, dFpdx: Any) -> Any:
  """Hessian curvature term for composition (Jet notes notation).

  For G = T ∘ F, the other contribution at p is
    (∂²T^k/∂F^b∂F^a) · (∂F_p^b/∂x^j) · (∂F_p^a/∂x^i).
  Compute via a second JVP of the outer function along tangents U, V drawn from
  columns of (∂F_p/∂x^i).
  """

  def second_jvp(dFpdx_i, dFpdx_j):
    # Computes (∂²T^k/∂F^b∂F^a) · (∂F_p^b/∂x^j) · (∂F_p^a/∂x^i) for a fixed i, j
    first_jvp_under_v = lambda _Fp: jax.jvp(T, (_Fp,), (dFpdx_i,))[1]
    return jax.jvp(first_jvp_under_v, (Fp,), (dFpdx_j,))[1]

  # Vmap over both tangent inputs U and V.
  @partial(jax.vmap, in_axes=-1, out_axes=-1)
  def inner_vmap(dFpdx_i):

    @partial(jax.vmap, in_axes=-1, out_axes=-1)
    def inner_vmap2(dFpdx_j):
      return second_jvp(dFpdx_i, dFpdx_j)

    return inner_vmap2(dFpdx)

  return inner_vmap(dFpdx)

def _is_jet(x: Any) -> bool:
  """Checks if a value is a Jet instance."""
  return isinstance(x, Jet)

def _get_coordinate_dim(gradient_or_hessian):
  """Extract coordinate dimension from a gradient or hessian (which might be a PyTree)."""
  if gradient_or_hessian is None:
    return None
  # Get any leaf from the PyTree
  leaves = jtu.tree_leaves(gradient_or_hessian)
  if not leaves:
    return None
  # Return the last axis size of the first leaf
  return leaves[0].shape[-1]

def jet_decorator(f: Callable) -> Callable:
  """
  Lift a function to operate on `Jet` inputs and propagate derivatives.
  Suppose inner maps provide jet data F at p ∈ M and an outer function T acts
  on F(p). If inputs carry 1- or 2-jet data at p, this returns the pushed-forward
  jet of G = T ∘ F at p.

  Pushforward formulas (Jet notes; per-output k):
  - First order:  ∂G_p^k/∂x^i = (∂T^k/∂F^a) (F(p)) · (∂F_p^a/∂x^i).
  - Second order: ∂²G_p^k/∂x^i∂x^j = (∂T^k/∂F^a) (F(p)) · (∂²F_p^a/∂x^i∂x^j)
                                     + (∂²T^k/∂F^b∂F^a) (F(p)) · (∂F_p^b/∂x^j) · (∂F_p^a/∂x^i).
    Implemented via JVP-of-JVP using ``_get_hessian_transport`` and
    ``_get_hessian_curvature``.

  Change of coordinates for J[F]_p is not implemented here.

  This decorator allows a function that normally accepts JAX arrays or PyTrees
  of arrays to accept `Jet` objects and to return the corresponding value,
  gradient, and (optionally) Hessian of the composed function.

  Inputs
  - Positional arguments may be a mix of:
    - `Jet` objects (scalars/arrays or PyTree-valued), or
    - regular JAX arrays / PyTrees of arrays.
  - Keyword arguments are passed through unchanged and are not treated as Jets.

  Output
  - If the function returns a single array-like value, the decorator returns a
    single `Jet` with fields `(value, gradient, hessian)`.
  - If the function returns a PyTree of arrays:
    - If there was exactly one Jet input whose `value` is a PyTree, the return
      is a single `Jet` whose `value/gradient/hessian` mirror that PyTree.
    - Otherwise, the return is a PyTree with the same structure as the output,
      with each leaf replaced by a `Jet`.

  Selective derivative propagation
  - The decorator determines which argument groups (each positional argument may
    be a Jet or a PyTree of Jets) actually influence the function's output via
    a light-weight JVP probe per group.
  - The coordinate dimension N is the sum of coordinate dimensions across only
    the used argument groups that provide gradients.
  - If any used group does not provide gradients, no derivatives are returned
    (the result has `gradient=None` and `hessian=None`).
  - If all used groups provide gradients but at least one used group lacks a
    Hessian, only first-order derivatives are returned (the result has
    `hessian=None`).

  Support for Jet-annotated parameters
  - If the function signature annotates one or more parameters with `Jet`, the
    decorator will rewrite the function to accept component-level Jets for those
    parameters. This enables differentiating through functions that operate on
    `Jet` internals, e.g. `jet_in.value`, `jet_in.gradient`, etc.

  PyTree semantics
  - `Jet` may hold PyTree-valued fields; in this case, `gradient` and `hessian`
    must mirror the PyTree structure of `value` and satisfy the shape rules
    described in the `Jet` docstring.

  Examples
  >>> @jet_decorator
  ... def square(x):
  ...   return x**2
  >>> j = function_to_jet(lambda t: t, jnp.array(2.0))
  >>> out = square(j)
  >>> out.value, out.gradient, out.hessian
  (Array(4., ...), Array([4.], ...), Array([[2.]], ...))

  Unused argument with missing derivatives is ignored for propagation:
  >>> @jet_decorator
  ... def add_x_only(x, y):
  ...   return x + 1  # does not depend on y
  >>> x = function_to_jet(lambda t: t, jnp.array(3.0))
  >>> y = Jet(value=jnp.array(5.0), gradient=None, hessian=None)
  >>> out = add_x_only(x, y)
  >>> out.gradient is not None and out.hessian is not None
  True
  """
  # Functions with Jet annotations are not supported.
  sig = inspect.signature(f)
  if any(p.annotation == Jet for p in sig.parameters.values()):
    raise TypeError(
        "jet_decorator cannot be applied to functions that accept Jet objects "
        "as arguments. To differentiate through such a function, rewrite it "
        "to operate on Jet components (value, gradient, hessian) and pass "
        "them as separate arguments."
    )

  @wraps(f)
  def decorated_f(*args, **kwargs):
    # 1. SETUP: Extract primals and identify all Jet leaves in the arguments.
    primals = jtu.tree_map(lambda x: x.value if _is_jet(x) else x, args, is_leaf=_is_jet)

    # Collect Jets, but group them by argument (to handle PyTrees of Jets correctly)
    # Each arg might be a Jet, a PyTree of Jets, or a regular value
    arg_jet_groups = []
    group_arg_positions: List[int] = []
    for arg_idx, arg in enumerate(args):
      jets_in_arg = [x for x in jtu.tree_leaves(arg, is_leaf=_is_jet) if _is_jet(x)]
      if jets_in_arg:
        arg_jet_groups.append(jets_in_arg)
        group_arg_positions.append(arg_idx)

    if not arg_jet_groups:
      return f(*args, **kwargs)

    # Determine which argument groups are actually used by f by probing with JVP
    # along primals directions for each group's argument subtree.
    f_primals = lambda a: f(*a, **kwargs)
    primals_per_arg = primals if isinstance(primals, tuple) else (primals,)

    # Helpers to build tangents matching a primals subtree, handling None leaves
    def make_tangent_like(subtree, fill: str):
      def map_leaf(v):
        if v is None:
          return None
        va = jnp.asarray(v)
        return jnp.ones_like(va) if fill == 'ones' else jnp.zeros_like(va)
      return jtu.tree_map(map_leaf, subtree, is_leaf=lambda x: x is None)

    if USE_SENSITIVITY_PROBE:
      used_groups: List[bool] = []
      for group_idx, arg_pos in enumerate(group_arg_positions):
        tangents_per_arg = []
        for idx in range(len(primals_per_arg)):
          if idx == arg_pos:
            # Ones for Jet-originated leaves in this argument; zeros elsewhere
            def per_leaf(a, p):
              if _is_jet(a):
                return make_tangent_like(p, 'ones')
              else:
                return make_tangent_like(p, 'zeros')
            t_arg = jtu.tree_map(per_leaf, args[idx], primals_per_arg[idx], is_leaf=_is_jet)
          else:
            t_arg = make_tangent_like(primals_per_arg[idx], 'zeros')
          tangents_per_arg.append(t_arg)

        tangents = tuple(tangents_per_arg) if isinstance(primals, tuple) else tangents_per_arg[0]
        sensitivity = jax.jvp(f_primals, (primals,), (tangents,))[1]
        leaves = jtu.tree_leaves(sensitivity)
        is_used = any(bool(jnp.any(jnp.asarray(leaf))) for leaf in leaves) if leaves else False
        used_groups.append(is_used)
    else:
      # Treat all Jet argument groups as used (no sensitivity probing)
      used_groups = [True] * len(group_arg_positions)

    # Per-group derivative availability
    group_has_grad = [all(j.gradient is not None for j in group) for group in arg_jet_groups]
    group_has_hess = [all(j.hessian is not None for j in group) for group in arg_jet_groups]

    # Debug prints: which groups are used and shapes of primals
    primals_shapes = jtu.tree_map(lambda v: getattr(v, 'shape', None), primals)
    # print(f"[jet] f={getattr(f, '__name__', '<anon>')} used_groups={used_groups} group_has_grad={group_has_grad}")
    # print(f"[jet] primals_shapes={primals_shapes}")

    # If any used group lacks gradients, we cannot compute derivatives safely
    if any(used and (not has_g) for used, has_g in zip(used_groups, group_has_grad)):
      value = f(*primals, **kwargs) if isinstance(primals, tuple) else f(primals)
      if isinstance(value, (dict, list, tuple)):
        return jtu.tree_map(lambda v: Jet(value=v, gradient=None, hessian=None), value)
      else:
        return Jet(value=value, gradient=None, hessian=None)

    # Active groups are used and have gradients
    active_group_indices = [i for i, used in enumerate(used_groups) if used]

    # All active jets must have the same coordinate dimension.
    sizes = [_get_coordinate_dim(arg_jet_groups[i][0].gradient) for i in active_group_indices]
    valid_sizes = [s for s in sizes if s is not None]

    if not valid_sizes:
      N = 0
    else:
      first_size = valid_sizes[0]
      if not all(s == first_size for s in valid_sizes):
        raise ValueError(f"All active Jet inputs must have the same coordinate dimension. Mismatched dimensions: {valid_sizes}")
      N = first_size

    # Create a mapping from Jet id to its active group-local index
    jet_to_active_group = {}
    for group_idx in active_group_indices:
      for jet in arg_jet_groups[group_idx]:
        jet_to_active_group[id(jet)] = group_idx

    # 2. TANGENT CONSTRUCTION: Create unified tangent PyTrees for gradients and Hessians.
    def grad_mapper(x):
      if _is_jet(x) and (id(x) in jet_to_active_group):
        return x.gradient
      else:
        # Non-Jet leaves and inactive Jets get zero tangents.
        value_tree = x.value if _is_jet(x) else x

        def zero_leaf(leaf):
            if leaf is None:
                return None
            vx = jnp.asarray(leaf)
            if N == 0:
              # Cannot create a zero-sized dimension, so return a zero-like array
              # with the same shape as the value. This case should be handled
              # by an earlier check that returns a Jet with None derivatives,
              # but this is a safeguard.
              return jnp.zeros_like(vx)
            return jnp.zeros((*vx.shape, N), dtype=vx.dtype)

        return jtu.tree_map(zero_leaf, value_tree, is_leaf=lambda n: n is None)

    total_grad_tangent = jtu.tree_map(grad_mapper, args, is_leaf=_is_jet)

    # Debug prints: N and tangent shapes
    tg_shapes = jtu.tree_map(lambda v: getattr(v, 'shape', None), total_grad_tangent)
    # print(f"[jet] N={N} active_groups={[i for i, u in enumerate(used_groups) if u]}")
    # print(f"[jet] total_grad_tangent_shapes={tg_shapes}")
    # Note: primals may be a PyTree (dict/tuple/list) or eqx.Module; do not assume .ndim.
    # Shape consistency is enforced leafwise elsewhere (Jet.__check_init__).

    # Hessian is only computable if all used groups have Hessians
    hessian_possible = all(
      (not used) or has_h for used, has_h in zip(used_groups, group_has_hess)
    ) and len(active_group_indices) > 0

    if hessian_possible:
      def hess_mapper(x):
        if _is_jet(x) and (id(x) in jet_to_active_group) and (x.hessian is not None):
          return x.hessian
        else:
          # Non-Jet leaves and inactive Jets get zero tangents.
          value_tree = x.value if _is_jet(x) else x

          def zero_leaf(leaf):
            if leaf is None:
                return None
            vx = jnp.asarray(leaf)
            if N == 0:
              return jnp.zeros_like(vx)
            return jnp.zeros((*vx.shape, N, N), dtype=vx.dtype)

          return jtu.tree_map(zero_leaf, value_tree, is_leaf=lambda n: n is None)

      total_hess_tangent = jtu.tree_map(hess_mapper, args, is_leaf=_is_jet)

    # 3. DIFFERENTIATION: Decompose into gradient and Hessian calculations.
    # f(z)
    T = lambda a: f(*a, **kwargs)

    # df/dx^u =
    gradient = _get_gradient(T, primals, total_grad_tangent)

    # d^2f/dx^u dx^v = d/dx^v[ df/dx^u ]
    if hessian_possible:
      transport = _get_hessian_transport(T, primals, total_hess_tangent)
      intrinsic = _get_hessian_curvature(T, primals, total_grad_tangent)
      hessian = jtu.tree_map(jnp.add, transport, intrinsic)
    else:
      hessian = None

    # 4. PACKAGE RESULTS: Compute final value and assemble output Jets.
    value = f(*primals, **kwargs) if isinstance(primals, tuple) else f(primals)

    # Determine if we have a single Jet with PyTree values as input
    # If so, we should return a single Jet with PyTree values, not a PyTree of Jets
    single_jet_pytree_input = (
      len(args) == 1 and
      _is_jet(args[0]) and
      isinstance(args[0].value, (dict, list, tuple))
    )

    def pack_leaf(v, g, h):
      return Jet(value=v, gradient=g, hessian=h)

    if isinstance(value, (dict, list, tuple)):
      if single_jet_pytree_input:
        # Return a single Jet with PyTree values
        return Jet(value=value, gradient=gradient, hessian=hessian)
      else:
        # Return a PyTree of Jets (original behavior)
        return jtu.tree_map(pack_leaf, value, gradient, hessian)
    else:
      return Jet(value=value, gradient=gradient, hessian=hessian)

  return decorated_f

################################################################################################################

@dispatch
def change_coordinates(
  jet: Jet,
  x_to_z: Callable[[Array], Array],
  x: Array
) -> Jet:
  """
  Suppose that jet is J[F]_p in coordinates x and we want to express it in coordinates z.
  J[F]_p in coordinates z is given by J[F]_z = (F_z^k, ∂F_z^k/∂z^i, ∂²F_z^k/∂z^i∂z^j),
  where
  F_z^k = F_x^k is unchanged,
  ∂F_z^k/∂z^i = (∂F_x^k/∂x^j) · (∂x^j/∂z^i),
  ∂²F_z^k/∂z^i∂z^j = (∂F_x^k/∂x^l) · (∂²x^l/∂z^i∂z^j) + (∂²F_x^k/∂x^l∂x^m) · (∂x^l/∂z^i) · (∂x^m/∂z^j),
  for i,j = 1,…,d and k = 1,…,n.
  """
  Fp = jet.value
  dFpdx = jet.gradient
  d2Fpdx2 = jet.hessian

  dzdx = jax.jacrev(x_to_z)(x)
  assert dzdx.shape == (x.shape[0], x.shape[0]), "dzdx should be a square matrix"
  d2zdx2 = jax.jacfwd(jax.jacrev(x_to_z))(x)
  dxdz = jnp.linalg.inv(dzdx)

  dFpdz = jnp.einsum("...a,ai->...i", dFpdx, dxdz)

  hess_inner_term1 = jnp.einsum("cbd,...c->...bd", d2zdx2, dFpdz)
  hess_inner_term = -hess_inner_term1 + d2Fpdx2

  d2Fpdz2 = jnp.einsum("...db,bj,di->...ij", hess_inner_term, dxdz, dxdz)

  return Jet(value=Fp, gradient=dFpdz, hessian=d2Fpdz2)


@dispatch
def change_coordinates(
  jet: Jet,
  x_to_z_jacobian: Jacobian,
) -> Jet:
  """
  Change coordinates for a Jet using a precomputed Jacobian J[z](x).

  The Jacobian object encodes
    - value[a, i]   = ∂z^a / ∂x^i,
    - gradient[a, i, j] = ∂²z^a / ∂x^i ∂x^j,
  at the point x = jacobian.p.

  Given J[F]_p in x-coordinates, this returns J[F]_p expressed in z-coordinates
  using the same formulas as the function-based change_coordinates overload.
  """
  Fp = jet.value
  dFpdx = jet.gradient
  d2Fpdx2 = jet.hessian

  if dFpdx is None or d2Fpdx2 is None:
    raise ValueError("change_coordinates(jet, x_to_z_jacobian) requires jet to have gradient and hessian.")

  dzdx = x_to_z_jacobian.value
  dim = dzdx.shape[0]

  if dzdx.shape != (dim, dim):
    raise ValueError(f"Jacobian value must be square of shape (N, N), got {dzdx.shape}.")

  if x_to_z_jacobian.gradient is None:
    d2zdx2 = jnp.zeros((dim, dim, dim), dtype=dzdx.dtype)
  else:
    d2zdx2 = x_to_z_jacobian.gradient

  dxdz = jnp.linalg.inv(dzdx)

  dFpdz = jnp.einsum("...a,ai->...i", dFpdx, dxdz)

  hess_inner_term1 = jnp.einsum("cbd,...c->...bd", d2zdx2, dFpdz)
  hess_inner_term = -hess_inner_term1 + d2Fpdx2

  d2Fpdz2 = jnp.einsum("...db,bj,di->...ij", hess_inner_term, dxdz, dxdz)

  return Jet(value=Fp, gradient=dFpdz, hessian=d2Fpdz2)
