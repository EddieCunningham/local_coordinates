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
  value: Float[Array, "N N"]        # First derivatives: J[a, i] = ∂z^a / ∂x^i
  gradient: Optional[Float[Array, "N N N"]]  # Second derivatives: ∂²z^a / ∂x^i ∂x^j
  hessian: Optional[Float[Array, "N N N N"]] # (Reserved) Third derivatives if needed

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.value.ndim == 2:
      return None
    elif self.value.ndim == 3:
      return self.value.shape[0]
    else:
      raise ValueError(f"Invalid number of dimensions: {self.value.ndim}")

  def get_inverse(self) -> "Jacobian":
    """
    Return the inverse Jacobian object.

    If self encodes the derivatives of a coordinate map z(x), with
      - self.value[a, i]   = ∂z^a / ∂x^i,
      - self.gradient[a, i, j] = ∂²z^a / ∂x^i ∂x^j,
    then the returned object encodes the derivatives of the local inverse map
    x(z) at the same base point, with
      - value[i, a]   = ∂x^i / ∂z^a,
      - gradient[i, a, b] = ∂²x^i / ∂z^a ∂z^b.

    The formulas are the standard inverse-function Taylor relations:
      ∂x/∂z = (∂z/∂x)^{-1},
      ∂²x/∂z^a∂z^b = - (∂z/∂x)^{-1} ⋅ (∂²z/∂x^j∂x^k)
                                 ⋅ (∂x/∂z)^j_a ⋅ (∂x/∂z)^k_b.
    """
    A = self.value
    dim = A.shape[0]

    if A.shape != (dim, dim):
      raise ValueError(f"Jacobian value must be square of shape (N, N), got {A.shape}.")

    A_inv = jnp.linalg.inv(A)

    B = self.gradient  # shape (a, j, k) if present
    C = self.hessian   # shape (a, j, k, l) if present

    if B is None:
      grad_inv = None
      hess_inv = None
    else:
      # First inverse derivatives:
      #   \frac{\partial^2 x^i}{\partial z^j \partial z^k} =
      #       -\frac{\partial^2 z^a}{\partial x^m \partial x^b}
      #       \frac{\partial x^i}{\partial z^a}
      #       \frac{\partial x^b}{\partial z^j}
      #       \frac{\partial x^m}{\partial z^k}
      grad_inv = -jnp.einsum("amb,ia,bj,mk->ijk", B, A_inv, A_inv, A_inv)

      if C is None:
        hess_inv = None
      else:
        # Third derivatives of the inverse (Hessian field of the Jacobian object).
        # Using the general inverse-map Taylor relation:
        #
        #   \frac{\partial^3 x^i}{\partial z^j \partial z^k \partial z^l} =
        #       -\frac{\partial x^i}{\partial z^a} \left(
        #           \frac{\partial^3 z^a}{\partial x^n \partial x^m \partial x^b}
        #           \frac{\partial x^n}{\partial z^l}
        #           \frac{\partial x^b}{\partial z^j}
        #           \frac{\partial x^m}{\partial z^k}
        #         + \frac{\partial^2 z^a}{\partial x^m \partial x^b} \left(
        #             \frac{\partial^2 x^m}{\partial z^j \partial z^k} \frac{\partial x^b}{\partial z^l}
        #           + \frac{\partial^2 x^m}{\partial z^k \partial z^l} \frac{\partial x^b}{\partial z^j}
        #           + \frac{\partial^2 x^m}{\partial z^l \partial z^j} \frac{\partial x^b}{\partial z^k}
        #           \right)
        #       \right)
        #

        # term_C represents:
        # -\frac{\partial x^i}{\partial z^a} \frac{\partial^3 z^a}{\partial x^n \partial x^m \partial x^b} \frac{\partial x^n}{\partial z^l} \frac{\partial x^b}{\partial z^j} \frac{\partial x^m}{\partial z^k}
        #
        # Note: The indices in C are (a, n, m, b) corresponding to ∂³z^a / ∂x^n ∂x^m ∂x^b
        # (This matches the order in function_to_jacobian: jacfwd(jacfwd(jacrev(f))) => (a, n, m, b))
        term_C = -jnp.einsum("ia,anmb,nl,bj,mk->ijkl", A_inv, C, A_inv, A_inv, A_inv)

        # term_B represents the second part involving the second derivatives.
        # We calculate one term of the cyclic sum first:
        # T = -\frac{\partial x^i}{\partial z^a} \frac{\partial^2 z^a}{\partial x^m \partial x^b} \frac{\partial^2 x^m}{\partial z^j \partial z^k} \frac{\partial x^b}{\partial z^l}
        #
        # Note: B indices are (a, m, b) corresponding to ∂²z^a / ∂x^m ∂x^b
        # grad_inv indices are (m, j, k) corresponding to ∂²x^m / ∂z^j ∂z^k
        term_B_partial = -jnp.einsum("ia,amb,mjk,bl->ijkl", A_inv, B, grad_inv, A_inv)

        # Symmetrize over j, k, l (cyclic permutations):
        # The expression in notes is symmetric in j, k, l (denoted p, q, r there).
        # We sum the partial term and its cyclic permutations of the denominator indices (j, k, l).
        # Original term indices: i, j, k, l.
        # Permutations needed for (j, k, l):
        # 1. (j, k, l) -> original
        # 2. (k, l, j) -> transpose axes (0, 2, 3, 1)
        # 3. (l, j, k) -> transpose axes (0, 3, 1, 2)
        term_B = (term_B_partial +
                  jnp.transpose(term_B_partial, (0, 2, 3, 1)) +
                  jnp.transpose(term_B_partial, (0, 3, 1, 2)))

        hess_inv = term_C + term_B

    return Jacobian(
      value=A_inv,
      gradient=grad_inv,
      hessian=hess_inv,
    )


def function_to_jacobian(f: Callable[[Array], Any], x: Array) -> Jacobian:
  # Build Jacobian of the coordinate change z(x) up to third order
  dzdx = jax.jacrev(f)(x)                       # (a, i)
  d2zdx2 = jax.jacfwd(jax.jacrev(f))(x)         # (a, i, j)
  d3zdx3 = jax.jacfwd(jax.jacfwd(jax.jacrev(f)))(x)  # (a, i, j, k)
  return Jacobian(value=dzdx, gradient=d2zdx2, hessian=d3zdx3)
