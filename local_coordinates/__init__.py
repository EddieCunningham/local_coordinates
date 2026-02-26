"""local_coordinates: differential geometry computations on Riemannian manifolds."""

__version__ = "0.1.0"

from local_coordinates.base import AbstractBatchableObject, auto_vmap
from local_coordinates.jet import Jet, function_to_jet, jet_decorator
from local_coordinates.jacobian import Jacobian
from local_coordinates.basis import BasisVectors, get_standard_basis
from local_coordinates.tangent import TangentVector
from local_coordinates.tensor import Tensor, TensorType
from local_coordinates.frame import Frame
from local_coordinates.metric import RiemannianMetric
from local_coordinates.connection import Connection, get_levi_civita_connection
from local_coordinates.riemann import (
  RiemannCurvatureTensor,
  RicciTensor,
  get_riemann_curvature_tensor,
  get_ricci_tensor,
)
