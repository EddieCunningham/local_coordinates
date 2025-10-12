from typing import Any, Callable, Tuple, Annotated, Optional, List, Union
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from linsdex import AbstractBatchableObject
from plum import dispatch
from local_coordinates.jet import Jet, jet_decorator
from local_coordinates.basis import BasisVectors
from local_coordinates.tensor import Tensor, TensorType

class RiemannianMetric(Tensor):
  """
  A Riemannian metric is a map from a tangent space to the real numbers.
  """
  tensor_type: TensorType = eqx.field(static=True)
  basis: BasisVectors
  components: Annotated[Jet, "D D"] # The components of the tensor written in the chosen basis

  def __init__(self, basis: BasisVectors, components: Annotated[Jet, "D D"]):
    super().__init__(tensor_type=TensorType(k=0, l=2), basis=basis, components=components)

  @property
  def batch_size(self):
    return self.basis.batch_size

  def __check_init__(self):
    super().__check_init__()
    if self.components.shape[-2] != self.components.shape[-1]:
      raise ValueError(f"Metric must be a square matrix")

    expected_batch_shape = self.basis.p.shape[:-1]
    actual_batch_shape = self.components.shape[:-2]
    if expected_batch_shape != actual_batch_shape:
        raise ValueError(
            f"Batch shape mismatch: basis implies {expected_batch_shape} but components have {actual_batch_shape}"
        )
