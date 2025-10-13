import string
from typing import Any, Callable, Tuple, Annotated, Optional, List, Union
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray, Scalar
from linsdex import AbstractBatchableObject
from plum import dispatch
from functools import partial, wraps
import inspect

# Optional sensitivity probing (used-groups detection) toggle.
# If False, all Jet argument groups are treated as used.
USE_SENSITIVITY_PROBE: bool = False

def set_jet_sensitivity_probe(enabled: bool) -> None:
  """Enable/disable used-groups sensitivity probing in jet_decorator."""
  global USE_SENSITIVITY_PROBE
  USE_SENSITIVITY_PROBE = bool(enabled)

class Jet(AbstractBatchableObject):
  """
  Jet data at a single expansion point in Euclidean coordinates.

  This container holds the truncated Taylor data of order ≤ 2 for a function
  evaluated at a fixed base point x₀ ∈ ℝᴺ:
    ( f(x₀), ∂f/∂xᶦ(x₀), ∂²f/∂xᶦ∂xʲ(x₀) ), for i,j = 1,…,N.

  In the classical language, this is the k-jet (k ≤ 2) of f at x₀, regarded as
  an abstract polynomial in an indeterminate z (not as a function of z):
    Jₓ₀ᵏ f(z) = Σ_{i=0}^k (1/i!) D^i f(x₀)[z^{⊗ i}].
  See “Jets of functions from the real line to a manifold” for background
  and transformation laws under coordinate changes [Wikipedia].

  Mathematics (per-leaf; vector-valued leaves are treated componentwise)
  - First order: for a displacement dx ∈ ℝᴺ,
      f(x₀ + dx) ≈ f(x₀) + ⟨∇f(x₀), dx⟩.
  - Second order:
      f(x₀ + dx) ≈ f(x₀) + ⟨∇f(x₀), dx⟩ + 1/2 · dxᵀ H(x₀) dx.
    Here ∇f has trailing axis size N and H has trailing axes (N, N).

  Components
  - value: f(x₀)
      PyTree of JAX arrays (or a single array/scalar).
  - gradient: ∂f/∂xᶦ(x₀), i = 1,…,N
      Optional PyTree mirroring `value`. Each gradient leaf has shape
      value_leaf.shape + (N,). The last axis indexes input-coordinate directions.
  - hessian: ∂²f/∂xᶦ∂xʲ(x₀), i,j = 1,…,N
      Optional PyTree mirroring `value`. Each hessian leaf has shape
      value_leaf.shape + (N, N).

  Coordinate dimension N
  - N is inferred from the trailing axis of gradient leaves (or from Hessian
    if gradient is absent). All leaves must agree on N.

  Batching semantics
  - Any leading axes of a value leaf are treated as batch/shape axes that are
    shared by gradient/hessian leaves. The `batch_size` property reports the
    leading shape of `value` for convenience.

  Taylor evaluation
  - Calling a Jet as `jet(dx)` evaluates the truncated Taylor polynomial at a
    displacement dx ∈ ℝᴺ from the base point:
      - 0th order if `gradient` is None ⇒ returns `value`.
      - 1st order if `gradient` present and `hessian` is None.
      - 2nd order if both `gradient` and `hessian` are present.

  Structure invariants
  - If provided, `gradient` and `hessian` must be PyTrees with the exact same
    structure as `value`.
  - For each corresponding leaf pair (v, g) and (v, H):
      g.ndim == v.ndim + 1 and H.ndim == v.ndim + 2.

  Construction
  - From a function: use `function_to_jet(f, x)` to construct a Jet (or PyTree
    of Jets) by differentiating `f` at input `x`.
  - Manual construction is also supported as long as the above invariants hold.

  Notes
  - Gradients and Hessians are expressed in standard Euclidean coordinates; no
    basis object is stored.
  - It is valid to have `gradient` present while `hessian` is None. If
    `gradient` is None, `hessian` is ignored by evaluation.

  Example
  >>> import jax.numpy as jnp
  >>> from local_coordinates.jet import function_to_jet
  >>> f = lambda x: jnp.array([x[0]**2, jnp.sin(x[1])])
  >>> x0 = jnp.array([1.0, 0.5])
  >>> j = function_to_jet(f, x0)
  >>> j.value.shape
  (2,)
  >>> j.gradient.shape  # Jacobian: (out_dim, N)
  (2, 2)
  >>> j.hessian.shape   # per-output Hessian stacked along output dim
  (2, 2, 2)
  >>> dx = jnp.array([0.1, -0.2])
  >>> approx = j(dx)  # 2nd-order Taylor if hessian available
  """
  value: Scalar
  gradient: Optional[Float[Array, "N"]]
  hessian: Optional[Float[Array, "N N"]]

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
      return f.shape[:-1]

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
    return Jet(
      value=self.gradient,
      gradient=self.hessian,
      hessian=None
    )

  def get_hessian_jet(self) -> "Jet":
    return Jet(
      value=self.hessian,
      gradient=None,
      hessian=None
    )

def function_to_jet(f: Callable[[Array], Any], x: Array) -> Jet:
  """Construct the 2-jet data of ``f`` at ``x``.

  Let f: ℝᴺ → Y, where Y is an array or a PyTree of arrays. This returns a
  Jet per output leaf u with components
    ( f_u(x), ∂f_u/∂xᶦ(x), ∂²f_u/∂xᶦ∂xʲ(x) ), i,j = 1,…,N.

  Mathematics (per output leaf u)
  - Gradient: (∇f_u(x))_i = ∂ f_u / ∂ x^i.
  - Hessian: (H_u(x))_{ij} = ∂² f_u / ∂ x^i ∂ x^j.

  Shapes
  - If a value leaf has shape S, then its gradient leaf has shape S + (N,),
    and its Hessian leaf has shape S + (N, N).

  Returns
  - If f returns a single array, a single ``Jet`` is returned.
  - If f returns a PyTree of arrays, the same structure is returned with each
    leaf replaced by a ``Jet``.

  Implementation notes
  - Uses ``jax.jacrev`` for Jacobians and ``jacfwd(jacrev)`` for Hessians.
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

def _get_gradient(f_primals: Callable, primals: Any, total_grad_tangent: Any) -> Any:
  """Gradient propagation for composition.

  If x ∈ ℝᴺ and the input is a composition y = A(x) with Jacobian JA(x)
  and outer function f, then by the chain rule
    ∇(f∘A)(x) = Jf(A(x)) · JA(x).

  Given primals = A(x) and total_grad_tangent = JA(x), this computes the
  right-hand product Jf(A(x)) · JA(x) by pushing each basis column of JA(x)
  through one JVP of f.
  """
  # The JVP of f(primals) with tangent S gives one column of the new Jacobian.
  first_jvp = lambda S: jax.jvp(f_primals, (primals,), (S,))[1]
  gradient = jax.vmap(first_jvp, in_axes=-1, out_axes=-1)(total_grad_tangent)
  return gradient

def _get_hessian_transport(f_primals: Callable, primals: Any, total_hess_tangent: Any) -> Any:
  """Hessian transport term for composition.

  For y = A(x) with Hessian HA(x) and outer function f, one term of the
  second-order chain rule is
    transport(x) = Jf(A(x)) · HA(x),
  where Jf(A(x)) multiplies the 2-tensor HA(x) along its columns and rows.
  This routine applies JVP of f to each column of HA(x), vmapped over both
  tensor axes, to realize that multiplication.
  """
  # Apply JVP to each element of the Hessian tangents.
  first_jvp = lambda X_elem: jax.jvp(f_primals, (primals,), (X_elem,))[1]
  jvp_vmapped_over_cols = jax.vmap(first_jvp, in_axes=-1, out_axes=-1)
  transport = jax.vmap(jvp_vmapped_over_cols, in_axes=-2, out_axes=-2)(total_hess_tangent)
  return transport

def _get_hessian_curvature(f_primals: Callable, primals: Any, total_grad_tangent: Any) -> Any:
  """Hessian curvature term for composition.

  The remaining second-order chain-rule contribution for y = A(x) is
    intrinsic(x)_{ij} = ⟨∂²f(A(x)); (∂A/∂x^i, ∂A/∂x^j)⟩,
  i.e. the pullback of f's Hessian along the columns of JA(x). Operationally
  this is computed as a second JVP of f evaluated at primals along two tangent
  directions U, V drawn from the columns of JA(x).
  """

  def second_jvp(U, V):
    # Computes d/dV[ d/dU[f] ]
    first_jvp_under_v = lambda p: jax.jvp(f_primals, (p,), (U,))[1]
    return jax.jvp(first_jvp_under_v, (primals,), (V,))[1]

  # Vmap over both tangent inputs U and V.
  inner_vmap = lambda U: jax.vmap(lambda V: second_jvp(U, V), in_axes=-1, out_axes=-1)(total_grad_tangent)
  intrinsic = jax.vmap(inner_vmap, in_axes=-1, out_axes=-1)(total_grad_tangent)
  return intrinsic

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
  Suppose f maps inputs to outputs, potentially composing with inner maps that
  provide jet data. If inputs carry 1- or 2-jet data at a base point, the
  decorator returns the pushed-forward jet of f∘(inputs).

  Mathematics (composition formulas; per-output leaf)
  - First order (chain rule): If y = A(x) and we compute f(y), with JA(x) the
    Jacobian of A and Jf(y) the Jacobian of f at y, then
      ∇(f∘A)(x) = Jf(A(x)) · JA(x).
  - Second order: Decomposed into transport + curvature
      D²(f∘A)(x)[u, v] = Jf(A(x)) · D²A(x)[u, v]
                         + D²f(A(x))[ JA(x)u, JA(x)v ].
    The implementation constructs these two terms respectively via
    ``_get_hessian_transport`` and ``_get_hessian_curvature`` using JVP-of-JVP.

  See Wikipedia: “Jets of functions from the real line to a manifold” for
  jet concepts and transformation behavior [Wikipedia].


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
    f_primals = lambda a: f(*a, **kwargs)

    # df/dx^u =
    gradient = _get_gradient(f_primals, primals, total_grad_tangent)

    # d^2f/dx^u dx^v = d/dx^v[ df/dx^u ]
    if hessian_possible:
      transport = _get_hessian_transport(f_primals, primals, total_hess_tangent)
      intrinsic = _get_hessian_curvature(f_primals, primals, total_grad_tangent)
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


