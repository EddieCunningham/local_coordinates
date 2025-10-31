import jax.numpy as jnp
from jax import jit, random
from functools import partial, reduce
import numpy as np
import jax
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Iterator
import jax.lax as lax
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, PyTree
import einops
import jax.tree_util as jtu

def svd(A):
  if A.shape[-1] == A.shape[-2]:
    return my_svd(A)
  return jnp.linalg.svd(A)

@jax.custom_jvp
def my_svd(A):
  U, s, VT = jnp.linalg.svd(A)
  V = jnp.einsum("...ji->...ij", VT)
  return U, s, V

@my_svd.defjvp
def my_svd_jvp(primals, tangents):
  A, = primals
  dA, = tangents
  U, s, V = my_svd(A)
  dU, ds, dV = svd_jvp_work(U, s, V, dA)
  return (U, s, V), (dU, ds, dV)

@partial(jnp.vectorize, signature="(n,n),(n),(n,n),(n,n)->(n,n),(n),(n,n)")
def svd_jvp_work(U, s, V, dA):
  dS = jnp.einsum("ij,iu,jv->uv", dA, U, V)
  ds = jnp.diag(dS)

  sdS = s*dS
  dSs = s[:,None]*dS

  s_diff = s[:,None]**2 - s**2 + 1e-5
  N = s.shape[-1]
  one_over_s_diff = jnp.where(jnp.arange(N)[:,None] == jnp.arange(N), 0.0, 1/s_diff)
  u_components = one_over_s_diff*(sdS + sdS.T)
  v_components = one_over_s_diff*(dSs + dSs.T)

  dU = jnp.einsum("uv,iv->iu", u_components, U)
  dV = jnp.einsum("uv,iv->iu", v_components, V)
  return (dU, ds, dV)
