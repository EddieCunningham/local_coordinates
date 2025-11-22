from typing import Any, Callable, Tuple, Annotated, Optional, List
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from linsdex import AbstractBatchableObject
from typing import Union

class Jacobian(AbstractBatchableObject):
  """
  A Jacobian is a matrix of partial derivatives.
  """
  p: Float[Array, "N"]
  value: Float[Array, "N N"]
  gradient: Optional[Float[Array, "N N N"]]
  hessian: Optional[Float[Array, "N N N N"]]

  def __check_init__(self):
    if self.value.shape != (self.p.shape[0], self.p.shape[0]):
      raise ValueError(f"Invalid number of dimensions: {self.value.shape}")

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.value.ndim == 2:
      return None
    elif self.value.ndim == 3:
      return self.value.shape[0]
    else:
      raise ValueError(f"Invalid number of dimensions: {self.value.ndim}")

