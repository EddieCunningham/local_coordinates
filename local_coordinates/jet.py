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

class Jet(AbstractBatchableObject):
  """
  A Jet is a Scalar valued function as well as its derivatives at a point.
  The derivatives are always expressed in standard Euclidean coordinates.
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


def function_to_jet(f: Callable[[Array], Any], x: Array) -> Jet:
  """Return a PyTree of Jet objects matching f's output structure at point x.

  - If f returns a single array, returns a single Jet.
  - If f returns a PyTree (e.g., tuple/list/dict) of arrays, returns the same
    structure with each leaf replaced by a Jet with value, gradient, and Hessian.
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

def _compute_gradient(f_primals: Callable, primals: Any, total_grad_tangent: Any) -> Any:
  """Computes the gradient term: Jf(A) . JA"""
  # The JVP of f(primals) with tangent S gives one column of the new Jacobian.
  first_jvp = lambda S: jax.jvp(f_primals, (primals,), (S,))[1]
  gradient = jax.vmap(first_jvp, in_axes=-1, out_axes=-1)(total_grad_tangent)
  return gradient

def _compute_transport_term(f_primals: Callable, primals: Any, total_hess_tangent: Any) -> Any:
  """Computes the Hessian's transport term: Jf(A) . HA"""
  # Apply JVP to each element of the Hessian tangents.
  first_jvp = lambda X_elem: jax.jvp(f_primals, (primals,), (X_elem,))[1]
  jvp_vmapped_over_cols = jax.vmap(first_jvp, in_axes=-1, out_axes=-1)
  transport = jax.vmap(jvp_vmapped_over_cols, in_axes=-2, out_axes=-2)(total_hess_tangent)
  return transport

def _compute_intrinsic_term(f_primals: Callable, primals: Any, total_grad_tangent: Any) -> Any:
  """Computes the Hessian's intrinsic term: Hf(A)[JA, JA]"""
  # Define the nested JVP for the second derivative.
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
  A decorator that extends a function to operate on `Jet` objects.

  This decorator enables functions that normally operate on JAX arrays or PyTrees of
  arrays to perform second-order automatic differentiation by propagating `Jet`
  objects. It computes the value, gradient, and Hessian of the composition of
  the decorated function with the function(s) represented by the input Jets.

  The decorated function can accept any combination of `Jet` objects and regular
  JAX arrays or PyTrees as positional arguments. Keyword arguments are passed
  through without Jet propagation.
  """

  # Check if any parameters are annotated as Jet (for higher-order differentiation)
  sig = inspect.signature(f)
  params = list(sig.parameters.values())
  jet_param_indices = [i for i, p in enumerate(params) if p.annotation == Jet]

  if jet_param_indices:
    # This function accepts Jets as inputs. Rewrite it to accept components.
    def rewritten_f(*component_args):
      # Reconstruct Jets from their components (value, gradient, hessian triples)
      # Note: jet_decorator has already extracted .value from the shifted Jets,
      # so component_args contains the raw values/gradients/hessians
      reconstructed_args = []
      comp_idx = 0
      for i in range(len(params)):
        if i in jet_param_indices:
          # This parameter expects a Jet, reconstruct it from 3 components
          # These are the raw array values after jet_decorator extracted them
          value_comp = component_args[comp_idx]
          grad_comp = component_args[comp_idx + 1]
          hess_comp = component_args[comp_idx + 2]

          # Reconstruct the Jet the function expects
          reconstructed_jet = Jet(
            value=value_comp,
            gradient=grad_comp,
            hessian=hess_comp
          )
          reconstructed_args.append(reconstructed_jet)
          comp_idx += 3
        else:
          # Regular argument, pass through
          reconstructed_args.append(component_args[comp_idx])
          comp_idx += 1

      return f(*reconstructed_args)

    # Apply jet_decorator to the rewritten function
    rewritten_decorated = jet_decorator(rewritten_f)

    @wraps(f)
    def jet_aware_wrapper(*args, **kwargs):
      # Shift input Jets: for each Jet parameter, create shifted Jets
      shifted_args = []
      for i, arg in enumerate(args):
        if i in jet_param_indices and isinstance(arg, Jet):
          # Create shifted Jets: gradient→value, hessian→gradient, None→hessian
          value_jet = Jet(
            value=arg.value,
            gradient=arg.gradient,
            hessian=arg.hessian
          )

          if arg.gradient is not None:
            gradient_jet = Jet(
              value=arg.gradient,
              gradient=arg.hessian,
              hessian=None
            )
          else:
            # If no gradient, create a Jet with None derivatives
            gradient_jet = Jet(
              value=arg.gradient,  # This is None
              gradient=None,
              hessian=None
            )

          if arg.hessian is not None:
            hessian_jet = Jet(
              value=arg.hessian,
              gradient=None,
              hessian=None
            )
          else:
            hessian_jet = Jet(
              value=arg.hessian,  # This is None
              gradient=None,
              hessian=None
            )

          shifted_args.extend([value_jet, gradient_jet, hessian_jet])
        else:
          shifted_args.append(arg)

      return rewritten_decorated(*shifted_args, **kwargs)

    return jet_aware_wrapper

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

    # Per-group derivative availability
    group_has_grad = [all(j.gradient is not None for j in group) for group in arg_jet_groups]
    group_has_hess = [all(j.hessian is not None for j in group) for group in arg_jet_groups]

    # If any used group lacks gradients, we cannot compute derivatives safely
    if any(used and (not has_g) for used, has_g in zip(used_groups, group_has_grad)):
      value = f(*primals, **kwargs) if isinstance(primals, tuple) else f(primals)
      if isinstance(value, (dict, list, tuple)):
        return jtu.tree_map(lambda v: Jet(value=v, gradient=None, hessian=None), value)
      else:
        return Jet(value=value, gradient=None, hessian=None)

    # Active groups are used and have gradients
    active_group_indices = [i for i, used in enumerate(used_groups) if used]

    # Determine the total coordinate dimension (R) by summing dimensions of active Jet ARGUMENTS.
    sizes = [_get_coordinate_dim(arg_jet_groups[i][0].gradient) for i in active_group_indices]
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    R = sum(sizes)

    # Create a mapping from Jet id to its active group-local index (for padding logic)
    jet_to_active_group = {}
    for active_idx, group_idx in enumerate(active_group_indices):
      for jet in arg_jet_groups[group_idx]:
        jet_to_active_group[id(jet)] = active_idx

    # 2. TANGENT CONSTRUCTION: Create unified tangent PyTrees for gradients and Hessians.
    # Each tangent has a final axis (or two for Hessians) of size R, where each
    # Jet's derivatives are placed in a unique slice, padded with zeros.

    def grad_mapper(x):
      if _is_jet(x) and (id(x) in jet_to_active_group):
        active_idx = jet_to_active_group[id(x)]
        m, before = sizes[active_idx], offsets[active_idx]
        after = R - before - m

        g = x.gradient
        # Pad the gradient PyTree leaves to the total coordinate dimension R.
        def pad_leaf(leaf):
          zb_g = jnp.zeros((*leaf.shape[:-1], before), dtype=leaf.dtype)
          za_g = jnp.zeros((*leaf.shape[:-1], after), dtype=leaf.dtype)
          return jnp.concatenate([zb_g, leaf, za_g], axis=-1)
        return jtu.tree_map(pad_leaf, g)
      else:
        # Non-Jet leaves and inactive Jets get zero tangents.
        if _is_jet(x):
          return jtu.tree_map(
            lambda leaf: jnp.zeros((*leaf.shape, R), dtype=leaf.dtype),
            x.value,
          )
        else:
          vx = jnp.asarray(x)
          return jnp.zeros((*vx.shape, R), dtype=vx.dtype)

    total_grad_tangent = jtu.tree_map(grad_mapper, args, is_leaf=_is_jet)

    # Hessian is only computable if all used groups have Hessians
    hessian_possible = all(
      (not used) or has_h for used, has_h in zip(used_groups, group_has_hess)
    ) and len(active_group_indices) > 0

    if hessian_possible:
      def hess_mapper(x):
        if _is_jet(x) and (id(x) in jet_to_active_group) and (x.hessian is not None):
          active_idx = jet_to_active_group[id(x)]
          m, before = sizes[active_idx], offsets[active_idx]
          after = R - before - m

          H = x.hessian
          # Pad the hessian PyTree leaves to the total coordinate dimensions R x R.
          def pad_leaf(leaf):
            # Pad columns to R.
            zb_h_cols = jnp.zeros((*leaf.shape[:-1], before), dtype=leaf.dtype)
            za_h_cols = jnp.zeros((*leaf.shape[:-1], after), dtype=leaf.dtype)
            H_cols = jnp.concatenate([zb_h_cols, leaf, za_h_cols], axis=-1)

            # Pad rows to R.
            zb_h_rows = jnp.zeros((*H_cols.shape[:-2], before, R), dtype=leaf.dtype)
            za_h_rows = jnp.zeros((*H_cols.shape[:-2], after, R), dtype=leaf.dtype)
            return jnp.concatenate([zb_h_rows, H_cols, za_h_rows], axis=-2)
          return jtu.tree_map(pad_leaf, H)
        else:
          if _is_jet(x):
            return jtu.tree_map(
              lambda leaf: jnp.zeros((*leaf.shape, R, R), dtype=leaf.dtype),
              x.value,
            )
          else:
            vx = jnp.asarray(x)
            return jnp.zeros((*vx.shape, R, R), dtype=vx.dtype)

      total_hess_tangent = jtu.tree_map(hess_mapper, args, is_leaf=_is_jet)

    # 3. DIFFERENTIATION: Decompose into gradient and Hessian calculations.
    f_primals = lambda a: f(*a, **kwargs)

    gradient = _compute_gradient(f_primals, primals, total_grad_tangent)

    if hessian_possible:
      transport = _compute_transport_term(f_primals, primals, total_hess_tangent)
      intrinsic = _compute_intrinsic_term(f_primals, primals, total_grad_tangent)
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


