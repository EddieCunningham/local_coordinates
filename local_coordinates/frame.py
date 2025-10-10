from typing import Annotated
import equinox as eqx
from jaxtyping import Array, Float
from local_coordinates.basis import BasisVectors
from local_coordinates.connection import Connection
from local_coordinates.jet import Jet
from linsdex import AbstractBatchableObject
from plum import dispatch


class Frame(AbstractBatchableObject):
  """
  A Frame represents a set of basis vectors at a point on a manifold that is
  "aware" of the manifold's geometric structure via a connection.
  It is the Riemannian-aware upgrade to the `BasisVectors` class.
  """
  basis: BasisVectors
  connection: Connection

  def __check_init__(self):
    if self.basis is not self.connection.basis:
      raise ValueError(
        "The `basis` of the Frame must be the same object as the "
        "`basis` of the Connection."
      )

  @property
  def p(self) -> Float[Array, "N"]:
    """The base point of the frame in Euclidean coordinates."""
    return self.basis.p

  @property
  def components(self) -> Annotated[Jet, "N D"]:
    """The components of the basis vectors, including derivatives, as a Jet."""
    return self.basis.components

  @property
  def batch_size(self):
    return self.basis.batch_size

@dispatch
def change_coordinates(frame: Frame, new_basis: BasisVectors) -> Frame:
  """
  Transform a frame from one basis to another.
  """
  new_basis = change_coordinates(frame.basis, new_basis)
  new_connection = change_coordinates(frame.connection, new_basis)
  return Frame(new_basis, new_connection)