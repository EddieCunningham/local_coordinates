from typing import Any, Callable, Tuple, Annotated, Optional, List, Union
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from linsdex import AbstractBatchableObject
import itertools
from plum import dispatch
from local_coordinates.jet import Jet, jet_decorator, function_to_jet, change_coordinates
from local_coordinates.basis import BasisVectors, get_basis_transform, get_standard_basis, apply_contravariant_transform
from local_coordinates.metric import RiemannianMetric
from local_coordinates.jet import get_identity_jet


class SecondOrderFlow(AbstractBatchableObject):
  """
  x^k(z) = J^k_i(z) + 1/2 H^k_{ij}z^i z^j
  """

  J: Float[Array, "N N"]
  H: Float[Array, "N N N"]

  def __init__(self, J: Float[Array, "N N"], H: Float[Array, "N N N"]):
    self.J = J
    # Ensure symmetry in the last two axes of H
    self.H = 0.5 * (H + jnp.swapaxes(H, -1, -2))

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.J.ndim == 2:
      return None
    elif self.J.ndim == 3:
      return self.J.shape[0]
    elif self.J.ndim > 3:
      return self.J.shape[:-2]
    else:
      raise ValueError(f"Invalid number of dimensions: {self.J.ndim}")

  def __call__(self, z: Float[Array, "N"]) -> Float[Array, "N"]:
    return self.J@z + 0.5*jnp.einsum("kij,i,j->k", self.H, z, z)

  def get_metric(self, z: Float[Array, "N"]) -> RiemannianMetric:
    dxdz = self.J + jnp.einsum("kij,j->ki", self.H, z)
    d2xdz2 = self.H

    # Get the metric components
    metric_components_z = Jet(value=dxdz, gradient=d2xdz2, hessian=None, dim=self.J.shape[-1])
    metric_components_x = change_coordinates(metric_components_z, self, z)
    x = self.__call__(z)
    basis = BasisVectors(p=x, components=metric_components_x)
    metric = RiemannianMetric(basis=basis, components=get_identity_jet(self.J.shape[-1], dtype=z.dtype))
    return metric

class ThirdOrderFlow(AbstractBatchableObject):
  """
  x^l(z) = J^l_i(z) + 1/2 H^l_{ij}z^i z^j + 1/6 T^l_{ijk}z^i z^j z^k
  """

  J: Float[Array, "N N"]
  H: Float[Array, "N N N"]
  T: Float[Array, "N N N N"]

  def __init__(self, J: Float[Array, "N N"], H: Float[Array, "N N N"], T: Float[Array, "N N N N"]):
    self.J = J
    # Symmetrize H over its last two axes
    self.H = 0.5 * (H + jnp.swapaxes(H, -1, -2))

    # Symmetrize T over its last three axes by averaging over all 3! permutations
    nd = T.ndim
    if nd < 3:
      raise ValueError("T must have at least 3 axes to symmetrize its last 3")
    prefix = list(range(nd - 3))
    last3 = [nd - 3, nd - 2, nd - 1]
    perms = list(itertools.permutations(last3))
    accum = 0.0
    for p in perms:
      perm_axes = prefix + list(p)
      accum = accum + jnp.transpose(T, perm_axes)
    self.T = accum / float(len(perms))

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.J.ndim == 2:
      return None
    elif self.J.ndim == 3:
      return self.J.shape[0]
    elif self.J.ndim > 3:
      return self.J.shape[:-2]
    else:
      raise ValueError(f"Invalid number of dimensions: {self.J.ndim}")

  def __call__(self, z: Float[Array, "N"]) -> Float[Array, "N"]:
    return self.J@z + 0.5*jnp.einsum("lij,i,j->l", self.H, z, z) + 1/6*jnp.einsum("lijk,i,j,k->l", self.T, z, z, z)

  def get_metric(self, z: Float[Array, "N"]) -> RiemannianMetric:
    dxdz = self.J + jnp.einsum("kij,j->ki", self.H, z) + 0.5*jnp.einsum("lijk,j,k->li", self.T, z, z)
    d2xdz2 = self.H + jnp.einsum("lijk,j,k->lij", self.T, z, z)
    d3xdz3 = self.T

    # Get the metric components
    metric_components_z = Jet(value=dxdz, gradient=d2xdz2, hessian=d3xdz3, dim=self.J.shape[-1])
    metric_components_x = change_coordinates(metric_components_z, self, z)
    x = self.__call__(z)
    basis = BasisVectors(p=x, components=metric_components_x)
    metric = RiemannianMetric(basis=basis, components=get_identity_jet(self.J.shape[-1], dtype=z.dtype))
    return metric