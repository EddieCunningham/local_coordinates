from typing import Any, Callable, Tuple, Annotated, Optional, List, Union
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from linsdex import AbstractBatchableObject
from local_coordinates.basis import BasisVectors, get_basis_transform, get_standard_basis
from local_coordinates.basis import change_coordinates
from plum import dispatch
from local_coordinates.tensor import Tensor, change_basis
from local_coordinates.tensor import change_coordinates
from local_coordinates.jet import Jet
from local_coordinates.jet import jet_decorator
from local_coordinates.jet import change_coordinates
from local_coordinates.metric import RiemannianMetric, lower_index
from local_coordinates.jet import get_identity_jet
from local_coordinates.frame import Frame, get_lie_bracket_between_frame_pairs, basis_to_frame
from local_coordinates.frame import change_coordinates
from local_coordinates.tangent import TangentVector, lie_bracket
from local_coordinates.tangent import change_coordinates
from local_coordinates.tensor import TensorType
from local_coordinates.connection import Connection, get_levi_civita_connection
from local_coordinates.connection import change_coordinates
from local_coordinates.jacobian import Jacobian
from local_coordinates.riemann import get_riemann_curvature_tensor, RiemannCurvatureTensor
import optimistix

class LocalOCT(AbstractBatchableObject):
  """
  A Local Orthogonal Coordinate Transform (OCT) is a coordinate transform that is orthogonal in a local neighborhood.

  It consists of an orthonormal frame U, whose columns are called the principal directions, and a set of function
  Psi_i: M -> R, which are the log of the loadings s_i.
  The derivatives of Psi_i are called the rotation coefficients and correspond to directional derivatives of the
  log loadings in the direction of the principal directions.

  Indexing Convention:
    Following Jet conventions, derivative indices are trailing. We use column-vector convention:

      - U[a, j] = U^a_j = a-th x-component of j-th basis vector U_j
      - Columns are vectors: U[:, j] = U_j
      - Orthonormality: U.T @ U = I (columns are orthonormal)

    This matches the thesis notation U^i_j directly: U[i, j] = U^i_j.

  """
  p: Float[Array, "N"] # The point at which the OCT is defined
  U: Float[Array, "N N"] # The orthonormal frame (columns are basis vectors, U[:, j] = U_j)
  log_s: Float[Array, "N"] # The log of the loadings
  beta: Float[Array, "N N"] # The rotation coefficients: β_{ij} = U_j(log s_i)
  dbeta: Float[Array, "N N N"] # The directional derivatives: dbeta[i,j,k] = U_k(β_{ij})

  def get_log_loadings_jet(self) -> Annotated[Jet, "N"]:
    """
    Get the log loadings jet.
    """
    return Jet(value=self.log_s, gradient=self.beta, hessian=self.dbeta)

  def get_log_det_jet(self) -> Jet:
    return Jet(value=-self.log_s.sum(axis=0), gradient=-self.beta.sum(axis=0), hessian=-self.dbeta.sum(axis=0))

  def get_beta_jet(self) -> Annotated[Jet, "N N"]:
    """
    Get the beta jet.
    """
    return Jet(value=self.beta, gradient=self.dbeta, hessian=None)

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    if self.p.ndim > 2:
      return self.p.shape[:-1]
    elif self.p.ndim == 2:
      return self.p.shape[0]
    elif self.p.ndim == 1:
      return None
    else:
      raise ValueError(f"Invalid number of dimensions: {self.p.ndim}")

  def get_jacobian(self) -> Jacobian:
    """
    Construct the Jacobian of the coordinate map x(z) from the LocalOCT parameters.

    From notes/oct_math.md, the Taylor coefficients are:
      J^i_j = s_j U^i_j                           (Jacobian)
      H^i_{jk} = s_k s_j (β_jk U^i_j + Γ^a_kj U^i_a)   (Hessian)
      T^i_{jkl} = s_l s_k s_j (...)                    (Third derivative)

    Note on indexing:
      - Thesis notation: U^i_j = component i of vector U_j
      - Code convention: self.U[:, j] = U_j (column j is basis vector j)
      - Therefore: U^i_j = self.U[i, j] (direct match!)

    Returns:
      Jacobian object with value (J), gradient (H), and hessian (T).
    """
    dim = self.p.shape[0]
    s = jnp.exp(self.log_s)

    I = jnp.eye(dim)
    # Christoffel symbols: Γ^k_{ij} = β_kj δ^k_i - β_jk δ^j_i
    # Stored as Gamma[i,j,k] = Γ^k_{ij}
    Gamma = jnp.einsum("kj,ki->ijk", self.beta, I) - jnp.einsum("jk,ij->ijk", self.beta, I)

    # U^i_j = self.U[i,j] directly (columns are vectors)
    # J^i_j = s_j U^i_j  =>  J[i,j] = s[j] * U[i,j]
    J = jnp.einsum("ij,j->ij", self.U, s)

    # H^i_{jk} = s_k s_j (β_jk U^i_j + Γ^a_kj U^i_a)
    # U^i_j = U[i,j], U^i_a = U[i,a]
    H = jnp.einsum("k,j,jk,ij->ijk", s, s, self.beta, self.U)
    H = H + jnp.einsum("k,j,kja,ia->ijk", s, s, Gamma, self.U)

    # T^i_{jkl} = s_l s_k s_j * (bracket expression)
    # where bracket = (β_kl + β_jl)(β_jk U^i_j + Γ^a_kj U^i_a)
    #                 + U_l(β_jk) U^i_j + U_l(β_kj) U^i_k + β_jk Γ^b_lj U^i_b
    #                 + Γ^a_kj Γ^b_la U^i_b - δ^j_k U_l(β_ja) U^i_a

    # First compute (β_jk U^i_j + Γ^a_kj U^i_a)
    # U^i_j = U[i,j], U^i_a = U[i,a]
    bracket_inner = jnp.einsum("jk,ij->ijk", self.beta, self.U)
    bracket_inner = bracket_inner + jnp.einsum("kja,ia->ijk", Gamma, self.U)

    # term_1: (β_kl + β_jl) * bracket_inner
    term_1 = jnp.einsum("kl,ijk->ijkl", self.beta, bracket_inner)
    term_1 = term_1 + jnp.einsum("jl,ijk->ijkl", self.beta, bracket_inner)

    # term_2: U_l(β_jk U^i_j + Γ^a_kj U^i_a) expanded via product rule
    # Note: dbeta[i,j,k] = U_k(β_ij)

    # a) U_l(β_jk) U^i_j  =>  dbeta[j,k,l] * U[i,j]
    terma = jnp.einsum("jkl,ij->ijkl", self.dbeta, self.U)

    # b) β_jk Γ^b_lj U^i_b  =>  beta[j,k] * Gamma[l,j,b] * U[i,b]
    termb = jnp.einsum("jk,ljb,ib->ijkl", self.beta, Gamma, self.U)

    # c) U_l(β_kj) U^i_k  =>  dbeta[k,j,l] * U[i,k]
    termc = jnp.einsum("kjl,ik->ijkl", self.dbeta, self.U)

    # d) -δ^j_k U_l(β_ja) U^i_a  =>  -I[k,j] * dbeta[j,a,l] * U[i,a]
    termd = -jnp.einsum("kj,jal,ia->ijkl", I, self.dbeta, self.U)

    # e) Γ^a_kj Γ^b_la U^i_b  =>  Gamma[k,j,a] * Gamma[l,a,b] * U[i,b]
    terme = jnp.einsum("kja,lab,ib->ijkl", Gamma, Gamma, self.U)

    term_2 = terma + termb + termc + termd + terme

    # T^i_{jkl} = s_l s_k s_j (term_1 + term_2)
    T_unsym = jnp.einsum("l,k,j,ijkl->ijkl", s, s, s, term_1 + term_2)

    # The formula computes ∂H/∂z^l which treats l differently from j,k.
    # By Clairaut's theorem, T must be symmetric in (j,k,l), so we symmetrize.
    # See notes/rnc.md lines 324-337 for the analogous symmetrization in RNC.
    def symmetrize_jkl(T):
      """Symmetrize T[i,j,k,l] in its last 3 indices (j,k,l)."""
      perms = [
        (0, 1, 2, 3),  # identity: (j,k,l)
        (0, 1, 3, 2),  # (j,l,k)
        (0, 2, 1, 3),  # (k,j,l)
        (0, 2, 3, 1),  # (k,l,j)
        (0, 3, 1, 2),  # (l,j,k)
        (0, 3, 2, 1),  # (l,k,j)
      ]
      return sum(jnp.transpose(T, p) for p in perms) / 6

    T = symmetrize_jkl(T_unsym)

    jacobian = Jacobian(value=J, gradient=H, hessian=T)
    return jacobian

  def get_coordinate_frame(self) -> Frame:
    """
    Construct the coordinate frame E as a Frame object in the standard x-basis.

    The coordinate basis vectors are E_j = ∂/∂z^j = J^i_j ∂/∂x^i where J^i_j = ∂x^i/∂z^j.
    Components are expressed in the standard x-basis with x-derivatives.

    The coordinate basis vectors satisfy:
    - g(E_i, E_j) = s_i^2 δ_{ij} (orthogonality with diagonal metric)
    - [E_i, E_j] = 0 (commuting, since they form a coordinate basis)

    Returns:
      Frame object with components[i, j] = E_j^i = J^i_j (columns are basis vectors).
    """
    standard_basis = get_standard_basis(self.p)
    dxdz: Jacobian = self.get_jacobian()

    # The Jacobian J[i,j] = J^i_j = ∂x^i/∂z^j
    # Frame expects columns to be vectors: components[:, j] = E_j
    # Since J[:, j] = E_j, no transpose needed!
    J_z: Jet = Jet(value=dxdz.value, gradient=dxdz.gradient, hessian=dxdz.hessian)

    # Convert z-derivatives to x-derivatives
    J_x: Jet = change_coordinates(J_z, dxdz)

    return Frame(p=self.p, components=J_x, basis=standard_basis)

  def get_loadings_jet(self) -> Jet:
    """
    Get the loadings s_i as a Jet (with derivatives).

    The loadings are s_j = ||E_j|| where E_j is the j-th coordinate basis vector.
    """
    E_frame: Frame = self.get_coordinate_frame()

    @jet_decorator
    def compute_loadings_jet(E_vals) -> Float[Array, "N"]:
      # E_vals[i, j] = E_j^i (column j = vector j, row i = component i)
      # ||E_j|| = sqrt(sum_i (E_j^i)^2) = norm along axis=0
      return jnp.linalg.norm(E_vals, axis=0)

    s_jet: Jet = compute_loadings_jet(E_frame.components.get_value_jet())
    return s_jet

  def get_principal_frame(self) -> Frame:
    """
    Construct the principal frame U as a Frame object.

    The principal basis U_j = E_j / s_j where E_j is the coordinate basis.
    Components are expressed in the standard x-basis.

    Returns:
      Frame object with components representing U_j in the standard basis (columns are vectors).
    """
    E_frame: Frame = self.get_coordinate_frame()

    # U_j = E_j / s_j, so we scale each column by 1/s_j
    @jet_decorator
    def scale_by_inv_s(E_val) -> Float[Array, "N N"]:
      # E_val[i, j] = E_j^i (column j = vector j, row i = component i)
      # U_j = E_j / s_j, so U_val[:, j] = E_val[:, j] / s[j]
      s_val = jnp.linalg.norm(E_val, axis=0)  # norm of each column
      return E_val / s_val[None, :]  # divide each column by its norm

    U_components: Jet = scale_by_inv_s(E_frame.components)
    return Frame(p=self.p, components=U_components, basis=E_frame.basis)

  def get_metric(self) -> RiemannianMetric:
    """
    Construct the metric in the principal frame basis.

    In the principal frame, the metric is the identity (orthonormal basis).
    The metric's basis is U (the principal frame vectors expressed in x-coordinates).

    Both Frame and BasisVectors use column-vector convention, so no transpose needed.
    """
    U_frame: Frame = self.get_principal_frame()
    U_basis = BasisVectors(p=self.p, components=U_frame.components)
    return RiemannianMetric(basis=U_basis, components=get_identity_jet(self.p.shape[0]))

  def _get_christoffel_symbols_value(self) -> Float[Array, "N N N"]:
    dim = self.p.shape[0]
    I = jnp.eye(dim)
    beta_val = self.beta

    # Γ^k_{ij} = β_{jk} δ^k_i - β_{kj} δ^j_i
    # Gamma[i,j,k] = β_{kj} * I[k,i] - β_{jk} * I[j,i]
    Gamma = jnp.einsum("kj,ki->ijk", beta_val, I) - jnp.einsum("jk,ij->ijk", beta_val, I)
    return Gamma

  def get_connection(self) -> Connection:
    """
    Compute the connection from β using the thesis formula.

    The Christoffel symbols are computed in the principal (U) frame:
        Γ^k_{ij} = β_{jk} δ^k_i - β_{kj} δ^j_i

    Returns:
      Connection object with Christoffel symbols in the U-frame basis.
    """
    beta_jet = self.get_beta_jet()

    @jet_decorator
    def components(beta_val) -> Array:
      dim = self.p.shape[0]
      I = jnp.eye(dim)

      # Γ^k_{ij} = β_{jk} δ^k_i - β_{kj} δ^j_i
      # Gamma[i,j,k] = β_{kj} * I[k,i] - β_{jk} * I[j,i]
      Gamma = jnp.einsum("kj,ki->ijk", beta_val, I) - jnp.einsum("jk,ij->ijk", beta_val, I)
      return Gamma

    christoffel_symbols: Jet = components(beta_jet)

    # The Christoffel symbols are in the U-frame, so the basis should be U
    # Both Frame and BasisVectors use column-vector convention, so no transpose needed
    U_frame: Frame = self.get_principal_frame()
    U_basis = BasisVectors(p=self.p, components=U_frame.components)
    return Connection(basis=U_basis, christoffel_symbols=christoffel_symbols)

  def get_lie_bracket_between_frame_pairs(self) -> Annotated[TangentVector, "D D"]:
    """
    Compute Lie brackets between all pairs of principal basis vectors.

    From the thesis (Proposition: Lie bracket of principal basis):
      [U_i, U_j] = β_{ij} U_i - β_{ji} U_j

    In x-components: [U_i, U_j]^a = β_{ij} U_i^a - β_{ji} U_j^a

    Returns:
      TangentVector with batched components of shape (D, D, D) where
      components[i, j, a] = [U_i, U_j]^a in x-coordinates.
    """
    from functools import partial

    U_frame: Frame = self.get_principal_frame()
    beta_jet = self.get_beta_jet()

    # Compute [U_i, U_j]^a = β_{ij} U_i^a - β_{ji} U_j^a
    @jet_decorator
    def compute_lie_brackets(beta_val, U_val) -> Float[Array, "D D D"]:
      # beta_val[i, j] = β_{ij}
      # U_val[a, i] = U_i^a (a-th x-component of i-th basis vector, columns are vectors)
      # [U_i, U_j]^a = β_{ij} U_i^a - β_{ji} U_j^a
      #             = β_{ij} U[a, i] - β_{ji} U[a, j]
      term1 = jnp.einsum("ij,ai->ija", beta_val, U_val)  # β_{ij} U_i^a
      term2 = jnp.einsum("ji,aj->ija", beta_val, U_val)  # β_{ji} U_j^a
      return term1 - term2

    lb_components: Jet = compute_lie_brackets(beta_jet, U_frame.components)

    # Create batched TangentVectors
    @partial(eqx.filter_vmap, in_axes=(0,))
    @partial(eqx.filter_vmap, in_axes=(0,))
    def make_tangent_vectors(lb_ij: Jet) -> TangentVector:
      return TangentVector(p=U_frame.p, components=lb_ij, basis=U_frame.basis)

    out: Annotated[TangentVector, "D D"] = make_tangent_vectors(lb_components)
    return out



  def _check_symmetries(self, atol: float = 1e-5, rtol: float = 1e-5, check_beta_symmetry: bool = True):
    """
    Check that attributes of this object have the symmetries that we expect.

    This verifies the constraints derived from the flatness conditions:
    1. U must be orthonormal: U^T U = I (columns are orthonormal)
    2. beta must be symmetric: beta = beta^T (optional, can be skipped for exploration)
    3. First Lamé equation (for k != i != j):
       U_k(beta_ij) = beta_ik * beta_kj - beta_ij * beta_kj
    4. Second Lamé equation (for i != j):
       U_i(beta_ji) + U_j(beta_ij) + beta_ji^2 + beta_ij^2 + sum_{k not in {i,j}} beta_ik * beta_jk = 0

    Args:
      atol: Absolute tolerance for numerical comparisons.
      rtol: Relative tolerance for numerical comparisons.
      check_beta_symmetry: If True, verify beta is symmetric. Set to False to explore
                           non-symmetric beta configurations (which violate integrability).

    Raises:
      AssertionError: If any of the symmetry constraints are violated.
    """
    n = self.U.shape[0]

    # 1) U must be orthonormal: U^T U = I (columns are orthonormal)
    UTU = self.U.T @ self.U
    assert jnp.allclose(UTU, jnp.eye(n), atol=atol, rtol=rtol), \
        f"U is not orthonormal: U^T U = {UTU}"

    # 2) beta must be symmetric: beta = beta^T (optional check)
    if check_beta_symmetry:
      assert jnp.allclose(self.beta, self.beta.T, atol=atol, rtol=rtol), \
          f"beta is not symmetric: beta = {self.beta}, beta^T = {self.beta.T}"

    # 3) First Lamé equation: For k != i != j (all three distinct):
    #    dbeta[i,j,k] = beta[i,k] * beta[k,j] - beta[i,j] * beta[k,j]
    #    where dbeta[i,j,k] = U_k(beta_ij)
    for i in range(n):
      for j in range(n):
        if i == j:
          continue
        for k in range(n):
          if k == i or k == j:
            continue
          # First Lamé equation: U_k(beta_ij) = beta_ik * beta_kj - beta_ij * beta_kj
          lhs = self.dbeta[i, j, k]
          rhs = self.beta[i, k] * self.beta[k, j] - self.beta[i, j] * self.beta[k, j]
          assert jnp.allclose(lhs, rhs, atol=atol, rtol=rtol), \
              f"First Lamé equation violated for (i,j,k)=({i},{j},{k}): " \
              f"U_k(beta_ij) = {lhs}, but beta_ik*beta_kj - beta_ij*beta_kj = {rhs}"

    # 4) Second Lamé equation: For i != j:
    #    U_i(beta_ji) + U_j(beta_ij) + beta_ji^2 + beta_ij^2 + sum_{k not in {i,j}} beta_ik * beta_jk = 0
    for i in range(n):
      for j in range(i + 1, n):  # Only check i < j to avoid duplicates
        # First part: U_i(beta_ji) + U_j(beta_ij) + beta_ji^2 + beta_ij^2
        part1 = (self.dbeta[j, i, i] + self.dbeta[i, j, j]
                 + self.beta[j, i]**2 + self.beta[i, j]**2)
        # Second part: sum_{k not in {i,j}} beta_ik * beta_jk
        part2 = 0.0
        for k in range(n):
          if k != i and k != j:
            part2 = part2 + self.beta[i, k] * self.beta[j, k]
        # The sum should equal zero
        total = part1 + part2
        assert jnp.allclose(total, 0.0, atol=atol, rtol=rtol), \
            f"Second Lamé equation violated for (i,j)=({i},{j}): " \
            f"U_i(β_ji) + U_j(β_ij) + β_ji² + β_ij² + Σβ_ik*β_jk = {total} (should be 0)"

def _lie_bracket_loss(U: Float[Array, "N N"], beta: Float[Array, "N N"]) -> Float[Array, ""]:
  """
  Compute the loss measuring violation of the Lie bracket equation.

  The Lie bracket of the principal frame is:
    [U_i, U_j]^a = beta[i,j] * U[a,i] - beta[j,i] * U[a,j]

  For orthonormal U, this simplifies because:
    ||[U_i, U_j]||^2 = beta[i,j]^2 * ||U_i||^2 - 2*beta[i,j]*beta[j,i]*<U_i,U_j> + beta[j,i]^2 * ||U_j||^2
                     = beta[i,j]^2 + beta[j,i]^2  (since ||U_i|| = 1 and <U_i,U_j> = 0)

  Summing over i < j gives the sum of all off-diagonal beta^2 elements.
  This uses O(N^2) memory instead of O(N^3).
  """
  n = beta.shape[0]
  off_diag_mask = 1.0 - jnp.eye(n)
  return jnp.sum(beta ** 2 * off_diag_mask)

def _flatness_loss(beta: Float[Array, "N N"], dbeta: Float[Array, "N N N"]) -> Float[Array, ""]:
  """
  Compute the flatness loss measuring violation of the Lamé equations.

  The Lamé equations are necessary conditions for an orthogonal coordinate system
  to exist in flat space. This function computes the sum of squared residuals:

  1. First Lamé equation (for k ≠ i, k ≠ j, AND i ≠ j):
     U_k(β_ij) = β_ik β_kj - β_ij β_kj
     Residual: dbeta[i,j,k] - (beta[i,k]*beta[k,j] - beta[i,j]*beta[k,j])

  2. Second Lamé equation (for i ≠ j):
     U_i(β_ji) + U_j(β_ij) + β_ji² + β_ij² + Σ_{k∉{i,j}} β_ik β_jk = 0
     Residual: dbeta[j,i,i] + dbeta[i,j,j] + beta[j,i]² + beta[i,j]² + Σ_k beta[i,k]*beta[j,k]

  Args:
    beta: Rotation coefficients, shape (N, N). beta[i,j] = U_j(log s_i).
    dbeta: Directional derivatives of rotation coefficients, shape (N, N, N).
           dbeta[i,j,k] = U_k(beta[i,j]).

  Returns:
    Scalar loss value (sum of squared residuals).
  """
  n = beta.shape[0]

  # === First Lamé equation (vectorized) ===
  # For all (i, j, k) with k ≠ i, k ≠ j, AND i ≠ j:
  # dbeta[i,j,k] = beta[i,k]*beta[k,j] - beta[i,j]*beta[k,j]

  # Create broadcast indices
  i_idx = jnp.arange(n)[:, None, None]  # (n, 1, 1)
  j_idx = jnp.arange(n)[None, :, None]  # (1, n, 1)
  k_idx = jnp.arange(n)[None, None, :]  # (1, 1, n)

  # Mask: k must differ from both i and j, AND i must differ from j
  # (First Lamé only applies when all three indices are distinct)
  first_lame_mask = (k_idx != i_idx) & (k_idx != j_idx) & (i_idx != j_idx)  # (n, n, n)

  # Compute target values using broadcasting:
  # beta[i,k]: i varies on axis 0, k on axis 2 -> beta[:, None, :] = (n, 1, n)
  # beta[k,j]: k varies on axis 2, j on axis 1 -> beta.T[None, :, :] = (1, n, n)
  #            where beta.T[b, c] = beta[c, b], so [_, j, k] = beta[k, j]
  # beta[i,j]: i on axis 0, j on axis 1 -> beta[:, :, None] = (n, n, 1)
  beta_ik = beta[:, None, :]      # (n, 1, n) -> broadcasts to (n, n, n)
  beta_kj = beta.T[None, :, :]    # (1, n, n) -> broadcasts to (n, n, n)
  beta_ij = beta[:, :, None]      # (n, n, 1) -> broadcasts to (n, n, n)

  first_lame_target = beta_ik * beta_kj - beta_ij * beta_kj
  first_lame_residual = dbeta - first_lame_target
  first_lame_loss = jnp.sum(jnp.where(first_lame_mask, first_lame_residual**2, 0.0))

  # === Second Lamé equation (vectorized) ===
  # For all i ≠ j:
  # dbeta[j,i,i] + dbeta[i,j,j] + beta[j,i]² + beta[i,j]² + Σ_{k∉{i,j}} beta[i,k]*beta[j,k] = 0

  # Create 2D index grids
  ii, jj = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing='ij')

  # Extract required dbeta entries using advanced indexing
  dbeta_jii = dbeta[jj, ii, ii]  # (n, n): entry [i,j] = dbeta[j,i,i]
  dbeta_ijj = dbeta[ii, jj, jj]  # (n, n): entry [i,j] = dbeta[i,j,j]

  # Beta terms
  beta_ji = beta[jj, ii]  # (n, n): entry [i,j] = beta[j,i]

  # First part of equation: U_i(β_ji) + U_j(β_ij) + β_ji² + β_ij²
  second_lame_part1 = dbeta_jii + dbeta_ijj + beta_ji**2 + beta**2

  # Second part: Σ_k beta[i,k]*beta[j,k] excluding k ∈ {i, j}
  # Full sum: (beta @ beta.T)[i,j] = Σ_k beta[i,k]*beta[j,k]
  full_sum = beta @ beta.T

  # Subtract k=i term: beta[i,i]*beta[j,i]
  diag_beta = jnp.diag(beta)
  k_eq_i_term = diag_beta[:, None] * beta.T  # [i,j] = beta[i,i]*beta[j,i]

  # Subtract k=j term: beta[i,j]*beta[j,j]
  k_eq_j_term = beta * diag_beta[None, :]    # [i,j] = beta[i,j]*beta[j,j]

  second_lame_part2 = full_sum - k_eq_i_term - k_eq_j_term

  # Residual: part1 + part2 should equal 0
  second_lame_residual = second_lame_part1 + second_lame_part2

  # Only count each unordered pair once (i < j), since equation is symmetric
  upper_tri_mask = ii < jj
  second_lame_loss = jnp.sum(jnp.where(upper_tri_mask, second_lame_residual**2, 0.0))

  return first_lame_loss + second_lame_loss

def fit_local_oct_to_log_det(
  local_oct: LocalOCT,
  log_det_jet: Jet,
  max_steps: int = 1000,
  solver: Optional[optimistix.AbstractMinimiser] = None,
  verbose: bool = False,
  reg_weight: float = 0.1,
  return_history: bool = False,
) -> Union[LocalOCT, Tuple[LocalOCT, List[dict]]]:
  """
  Fit a LocalOCT to match a target log-determinant jet.

  Optimizes over (U, log_s, beta, dbeta) to minimize:
    1. Integrability loss (Lamé equations)
    2. Log-det value/gradient/hessian matching loss
    3. Orthonormality constraint on U

  Args:
    local_oct: Initial LocalOCT parameters.
    log_det_jet: Target log-determinant jet to match.
    max_steps: Maximum optimization steps.
    solver: Optimistix minimizer. Defaults to BFGS.
    verbose: If True, print optimization progress.
    reg_weight: Weight for L1 regularization on beta and dbeta.
    return_history: If True, return training history list along with result.

  Returns:
    If return_history is False: Optimized LocalOCT.
    If return_history is True: Tuple of (Optimized LocalOCT, history list).
      History is a list of dicts, one per iteration, with loss components.
  """
  assert log_det_jet.shape == ()

  # Mutable container to capture loss history via callback
  history: List[dict] = []

  def record_loss(total_loss, aux):
    """Callback to record loss values at each step."""
    history.append({
      'total_loss': float(total_loss),
      **{k: float(v) for k, v in aux.items()}
    })

  if solver is None:
    verbose_set = frozenset({"step", "loss"}) if verbose else frozenset()
    solver = optimistix.BFGS(rtol=1e-6, atol=1e-6, verbose=verbose_set)

  def loss_fn(params: LocalOCT, log_det_jet: Jet):
    integrability_loss = _flatness_loss(params.beta, params.dbeta)

    U = params.U
    U, _ = jnp.linalg.qr(U)

    Gamma = params._get_christoffel_symbols_value()
    Uj_log_det_grad2 = jnp.einsum("aj,a->j", U, log_det_jet.gradient)
    Ui_Uj_log_det_hessian2_extrinsic = jnp.einsum("ai,bj,ab->ij", U, U, log_det_jet.hessian)
    Ui_Uj_log_det_hessian2_intrinsic = jnp.einsum("kjc,c->jk", Gamma, Uj_log_det_grad2)

    Uj_log_det_grad = Uj_log_det_grad2
    Ui_Uj_log_det_hessian = Ui_Uj_log_det_hessian2_extrinsic + Ui_Uj_log_det_hessian2_intrinsic

    # model_log_det_jet = params.get_log_loadings_jet()
    model_log_det_jet = params.get_log_det_jet()

    value_loss = jnp.sum((log_det_jet.value - model_log_det_jet.value)**2)
    grad_loss = jnp.sum((Uj_log_det_grad - model_log_det_jet.gradient)**2)
    hessian_loss = jnp.sum((Ui_Uj_log_det_hessian - model_log_det_jet.hessian)**2)

    log_det_jet_loss = value_loss + grad_loss + hessian_loss

    reg = jnp.abs(params.beta).sum() + jnp.abs(params.dbeta).sum()

    aux = dict(
      integrability_loss=integrability_loss,
      log_det_jet_loss=log_det_jet_loss,
      reg=reg,
      value_loss=value_loss,
      grad_loss=grad_loss,
      hessian_loss=hessian_loss,
    )

    total_loss = integrability_loss + log_det_jet_loss + reg_weight*reg

    # Record history via callback (only when requested)
    if return_history:
      jax.debug.callback(record_loss, total_loss, aux)

    return total_loss, aux

  # p is not used in loss_fn, so its gradient is zero and it won't change
  solution = optimistix.minimise(
    fn=loss_fn,
    solver=solver,
    y0=local_oct,
    args=log_det_jet,
    max_steps=max_steps,
    throw=False,
    has_aux=True
  )

  # Force U to be orthogonal (matching the projection done in loss_fn)
  result = solution.value
  U_ortho, _ = jnp.linalg.qr(result.U)
  result = eqx.tree_at(lambda x: x.U, result, U_ortho)

  if return_history:
    history = jtu.tree_map(lambda *xs: jnp.array(xs), *history)
    return result, history
  return result


def _compute_dbeta_from_beta(
    beta: Float[Array, "N N"],
    dbeta_free: Optional[Float[Array, "N N"]] = None,
) -> Float[Array, "N N N"]:
  """
  Compute dbeta from beta to exactly satisfy the Lamé equations.

  The Lamé equations constrain dbeta given beta:

  1. First Lamé (for k ≠ i, k ≠ j, AND i ≠ j): Fully determines dbeta[i,j,k].
     dbeta[i,j,k] = beta[i,k]*beta[k,j] - beta[i,j]*beta[k,j]

  2. Second Lamé (for i ≠ j): Constrains dbeta[j,i,i] + dbeta[i,j,j].
     From: dbeta[j,i,i] + dbeta[i,j,j] + beta[j,i]² + beta[i,j]² + Σ_{k∉{i,j}} beta[i,k]*beta[j,k] = 0
     So the sum must equal: -(beta[j,i]² + beta[i,j]² + Σ_{k∉{i,j}} beta[i,k]*beta[j,k])
     We split this evenly between the two entries (or use dbeta_free for asymmetric splits).

  3. Unconstrained (FREE) entries:
     - dbeta[i,i,k] for k ≠ i: N(N-1) free parameters (First Lamé requires i ≠ j)
     - dbeta[i,j,i] for i ≠ j: N(N-1) free parameters (First Lamé requires k ≠ i)
     - dbeta[i,i,i]: N free parameters (no constraint applies)
     Total free parameters: N(2N-1)

  Args:
    beta: Rotation coefficients, shape (N, N).
    dbeta_free: Optional free parameters for unconstrained entries, shape (N, N).
                If provided, dbeta_free[i,j] controls the splitting for the (i,j) pair
                in the second Lamé equation, and diagonal entries set dbeta[i,i,i].

  Returns:
    dbeta tensor of shape (N, N, N) satisfying the Lamé equations.
  """
  n = beta.shape[0]

  # Initialize output
  dbeta = jnp.zeros((n, n, n))

  # === First Lamé: dbeta[i,j,k] for k ≠ i, k ≠ j, AND i ≠ j ===
  i_idx = jnp.arange(n)[:, None, None]
  j_idx = jnp.arange(n)[None, :, None]
  k_idx = jnp.arange(n)[None, None, :]

  # Critical: First Lamé requires ALL THREE conditions: k ≠ i, k ≠ j, i ≠ j
  first_lame_mask = (k_idx != i_idx) & (k_idx != j_idx) & (i_idx != j_idx)

  beta_ik = beta[:, None, :]
  beta_kj = beta.T[None, :, :]
  beta_ij = beta[:, :, None]

  first_lame_values = beta_ik * beta_kj - beta_ij * beta_kj
  dbeta = jnp.where(first_lame_mask, first_lame_values, dbeta)

  # === Second Lamé: dbeta[j,i,i] and dbeta[i,j,j] for i ≠ j ===
  # Constraint: dbeta[j,i,i] + dbeta[i,j,j] = C[i,j]
  # where C[i,j] = -(beta[j,i]² + beta[i,j]² + Σ_{k∉{i,j}} beta[i,k]*beta[j,k])

  # Compute the excluded sum: Σ_{k∉{i,j}} beta[i,k]*beta[j,k]
  full_sum = beta @ beta.T
  diag_beta = jnp.diag(beta)
  k_eq_i_term = diag_beta[:, None] * beta.T
  k_eq_j_term = beta * diag_beta[None, :]
  excluded_sum = full_sum - k_eq_i_term - k_eq_j_term

  # C[i,j] = -(β_ji² + β_ij² + excluded_sum)
  C = -(beta.T**2 + beta**2 + excluded_sum)

  # C is symmetric: C[i,j] = C[j,i]
  # We need to set dbeta[j,i,i] and dbeta[i,j,j] such that their sum = C[i,j]
  # Default: split evenly
  half_C = C / 2.0

  # dbeta[a,b,b] for a ≠ b should be set to half_C[b,a]
  # (since dbeta[j,i,i] corresponds to C[i,j], and we want dbeta[a,b,b] = half_C[b,a])

  # Create mask for entries where third index equals second (c = b) and first ≠ second
  c_eq_b_mask = (k_idx == j_idx) & (i_idx != j_idx)

  # For dbeta[a,b,b], the value should be half_C[b,a]
  # half_C.T[a,b] = half_C[b,a], and we need this indexed by (i=a, j=b)
  half_C_ba = half_C.T[:, :, None]  # (n, n, 1) broadcasts to (n, n, n)

  dbeta = jnp.where(c_eq_b_mask, half_C_ba, dbeta)

  # === Unconstrained entries ===
  # dbeta[a,b,a] for a ≠ b (c = a, a ≠ b): leave as 0
  # dbeta[a,a,a] (diagonal): leave as 0

  # If dbeta_free is provided, use it for the free parameters
  if dbeta_free is not None:
    # Use diagonal of dbeta_free for dbeta[i,i,i]
    diag_free = jnp.diag(dbeta_free)
    diag_mask = (i_idx == j_idx) & (j_idx == k_idx)
    dbeta = jnp.where(diag_mask, diag_free[:, None, None], dbeta)

    # Could also use off-diagonal for asymmetric splitting, but keeping it simple

  return dbeta


def create_local_oct(
    p: Float[Array, "N"],
    key: PRNGKeyArray,
    beta_scale: float = 1.0,
    log_s_scale: float = 1.0,
) -> LocalOCT:
  """
  Create a valid LocalOCT object with randomly initialized parameters.

  This function creates a LocalOCT by:
  1. Generating a random orthonormal frame U via QR decomposition
  2. Generating random log-loadings
  3. Generating random rotation coefficients beta
  4. Computing dbeta analytically from beta to satisfy the Lamé equations

  The Lamé equations are the flatness constraints that ensure the parameters
  correspond to a valid orthogonal coordinate system in flat space.

  Args:
    p: The base point in the ambient space, shape (N,).
    key: JAX PRNG key for random initialization.
    beta_scale: Scale factor for the rotation coefficients (default 1.0).
    log_s_scale: Scale factor for the log-loadings (default 1.0).

  Returns:
    A LocalOCT object with valid parameters satisfying the Lamé equations.

  Example:
    >>> key = jax.random.PRNGKey(0)
    >>> p = jnp.zeros(3)
    >>> oct = create_local_oct(p, key)
    >>> oct._check_symmetries()  # Should pass without error
  """
  k1, k2, k3 = random.split(key, 3)
  dim = p.shape[0]

  # Create orthonormal principal basis via QR decomposition
  U = random.normal(k1, (dim, dim))
  U, _ = jnp.linalg.qr(U)

  # Create log-loadings (unconstrained)
  log_s = log_s_scale * random.normal(k2, (dim,))

  # Create rotation coefficients (symmetric)
  beta = beta_scale * random.normal(k3, (dim, dim))
  # beta = (beta + beta.T) / 2  # Symmetrize: β_{ij} = β_{ji}

  # Compute dbeta analytically to satisfy Lamé equations
  dbeta = _compute_dbeta_from_beta(beta)

  local_oct = LocalOCT(p, U, log_s, beta, dbeta)
  return local_oct


def plot_oct_grid(
    oct: LocalOCT,
    num: int = 21,
    span: float = 0.3,
    savepath: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = True,
    draw_grid: bool = True,
    draw_basis_vectors: bool = True,
    basis_vector_scale: float = 0.15,
    figsize: Tuple[float, float] = (8, 8),
    line_color_1: str = '#2E86AB',  # Steel blue
    line_color_2: str = '#A23B72',  # Berry
    linewidth: float = 0.8,
    alpha: float = 0.9,
    basepoint_color: str = '#1a1a2e',
    arrow_color: str = '#E94F37',  # Vermillion
    ax: Optional[Any] = None,
):
  """
  Plot the coordinate grid for a LocalOCT using its Taylor approximation.

  Creates a clean, minimalist visualization of the orthogonal coordinate
  system without axis ticks, spines, or grid lines.

  Args:
    oct: The LocalOCT object to visualize.
    num: Number of grid lines in each direction.
    span: Range of the grid in each coordinate direction [-span, span].
    savepath: If provided, save the figure to this path.
    title: Optional title for the plot.
    show: Whether to display the plot.
    draw_grid: Whether to draw the coordinate grid lines.
    draw_basis_vectors: Whether to draw the basis vectors at the origin.
    basis_vector_scale: Scale factor for the basis vector arrows.
    figsize: Figure size as (width, height).
    line_color_1: Color for the first family of coordinate lines.
    line_color_2: Color for the second family of coordinate lines.
    linewidth: Width of the grid lines.
    alpha: Transparency of the grid lines.
    basepoint_color: Color of the basepoint marker.
    arrow_color: Color of the basis vector arrows.
    ax: Optional matplotlib axes to plot on. If None, creates a new figure.

  Returns:
    fig, ax: The matplotlib figure and axes objects (fig is None if ax was provided).
  """
  import matplotlib.pyplot as plt
  import matplotlib.patches as mpatches

  # Get the Jacobian (Taylor coefficients)
  jacobian = oct.get_jacobian()
  dim = oct.p.shape[0]

  # Build a Jet for evaluation
  jet = Jet(
      value=oct.p,
      gradient=jacobian.value,
      hessian=jacobian.gradient,
  )

  uvs = jnp.linspace(-span, span, num)

  def eval_line(along_first: bool, fixed_val: float):
    """Evaluate points along a coordinate line."""
    ts = uvs
    U = jnp.zeros((ts.shape[0], dim))
    if along_first:
      # Vary first coord, fix second
      U = U.at[:, 0].set(ts)
      if dim > 1:
        U = U.at[:, 1].set(fixed_val)
    else:
      # Vary second coord, fix first
      if dim > 1:
        U = U.at[:, 1].set(ts)
      U = U.at[:, 0].set(fixed_val)

    Y = jax.vmap(lambda w: jet(w))(U)
    xs = Y[:, 0]
    ys = Y[:, 1] if Y.shape[1] > 1 else jnp.zeros_like(xs)
    return jnp.array(xs), jnp.array(ys)

  # Create figure or use provided axes
  fig = None
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)

  # Plot coordinate lines (if enabled)
  if draw_grid:
    # Plot coordinate lines - first family (varying first coordinate)
    for v in uvs:
      xs, ys = eval_line(True, float(v))
      ax.plot(xs, ys, color=line_color_1, linewidth=linewidth, alpha=alpha)

    # Plot coordinate lines - second family (varying second coordinate)
    for u in uvs:
      xs, ys = eval_line(False, float(u))
      ax.plot(xs, ys, color=line_color_2, linewidth=linewidth, alpha=alpha)

  # Mark the basepoint
  x0, y0 = float(oct.p[0]), float(oct.p[1]) if dim > 1 else 0.0
  ax.scatter([x0], [y0], c=basepoint_color, s=40, zorder=10, edgecolors='white', linewidths=1.5)

  # Draw basis vectors
  if draw_basis_vectors:
    J = jacobian.value  # shape (ambient_dim, tangent_dim)
    if J.ndim >= 2:
      v1 = J[:2, 0] if J.shape[1] >= 1 else jnp.zeros((2,))
      v2 = J[:2, 1] if J.shape[1] >= 2 else jnp.zeros((2,))

      for i, (vx, vy) in enumerate([(float(v1[0]), float(v1[1])),
                                     (float(v2[0]), float(v2[1]))]):
        dx = basis_vector_scale * vx
        dy = basis_vector_scale * vy
        arrow = mpatches.FancyArrowPatch(
            (x0, y0), (x0 + dx, y0 + dy),
            arrowstyle='-|>',
            mutation_scale=15,
            linewidth=2.0,
            color=arrow_color,
            zorder=15,
        )
        ax.add_patch(arrow)

  # Clean up the plot - minimal style with labels
  ax.set_aspect('equal', 'box')

  # Set fixed axis limits based on span (with some padding)
  limit = span * 1.5
  ax.set_xlim(-limit, limit)
  ax.set_ylim(-limit, limit)

  # Show only bottom and left spines (axis lines)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(True)
  ax.spines['left'].set_visible(True)
  ax.spines['bottom'].set_color('#888888')
  ax.spines['left'].set_color('#888888')
  ax.spines['bottom'].set_linewidth(0.8)
  ax.spines['left'].set_linewidth(0.8)

  # Add axis labels (using plain text to avoid mathtext parsing issues)
  ax.set_xlabel('z1', fontsize=12, labelpad=5)
  ax.set_ylabel('z2', fontsize=12, labelpad=5)

  # Add title if provided
  if title is not None:
    ax.set_title(title, fontsize=14, fontweight='light', pad=20)

  # Only do figure-level operations if we created the figure
  if fig is not None:
    plt.tight_layout()

    if savepath is not None:
      fig.savefig(savepath, bbox_inches='tight', dpi=200, facecolor='white', edgecolor='none')

    if show:
      plt.show()

  return fig, ax


def plot_oct_with_density(
    oct: LocalOCT,
    log_density_fn: Callable[[Array], float],
    num_grid: int = 21,
    num_heatmap: int = 100,
    span: float = 0.3,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    savepath: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = True,
    draw_heatmap: bool = True,
    draw_grid: bool = True,
    draw_basis_vectors: bool = True,
    basis_vector_scale: float = 0.15,
    figsize: Tuple[float, float] = (8, 8),
    line_color_1: str = '#2E86AB',
    line_color_2: str = '#A23B72',
    linewidth: float = 0.8,
    alpha: float = 0.9,
    basepoint_color: str = '#1a1a2e',
    arrow_color: str = '#E94F37',
    cmap: str = 'viridis',
    ax: Optional[Any] = None,
):
  """
  Plot heatmap of exp(log_density) with OCT coordinate curves overlaid.

  Args:
    oct: The LocalOCT object to visualize.
    log_density_fn: Function x -> log p(x) to evaluate for the heatmap.
    num_grid: Number of coordinate grid lines in each direction.
    num_heatmap: Resolution of the heatmap grid.
    span: Range of the grid in each coordinate direction [-span, span].
    xlim: Optional (xmin, xmax) tuple. If None, computed from coordinate curves.
    ylim: Optional (ymin, ymax) tuple. If None, computed from coordinate curves.
    savepath: If provided, save the figure to this path.
    title: Optional title for the plot.
    show: Whether to display the plot.
    draw_heatmap: Whether to draw the density heatmap.
    draw_grid: Whether to draw the coordinate grid lines.
    draw_basis_vectors: Whether to draw the basis vectors at the origin.
    basis_vector_scale: Scale factor for the basis vector arrows.
    figsize: Figure size as (width, height).
    line_color_1: Color for the first family of coordinate lines.
    line_color_2: Color for the second family of coordinate lines.
    linewidth: Width of the grid lines.
    alpha: Transparency of the grid lines.
    basepoint_color: Color of the basepoint marker.
    arrow_color: Color of the basis vector arrows.
    cmap: Colormap for the density heatmap.
    ax: Optional matplotlib axes to plot on. If None, creates a new figure.

  Returns:
    fig, ax: The matplotlib figure and axes objects (fig is None if ax was provided).
  """
  import matplotlib.pyplot as plt
  import matplotlib.patches as mpatches

  # Get the Jacobian (Taylor coefficients) for coordinate curves
  jacobian = oct.get_jacobian()
  dim = oct.p.shape[0]

  jet = Jet(
      value=oct.p,
      gradient=jacobian.value,
      hessian=jacobian.gradient,
  )

  uvs = jnp.linspace(-span, span, num_grid)

  def eval_line(along_first: bool, fixed_val: float):
    """Evaluate points along a coordinate line."""
    ts = uvs
    U = jnp.zeros((ts.shape[0], dim))
    if along_first:
      U = U.at[:, 0].set(ts)
      if dim > 1:
        U = U.at[:, 1].set(fixed_val)
    else:
      if dim > 1:
        U = U.at[:, 1].set(ts)
      U = U.at[:, 0].set(fixed_val)

    Y = jax.vmap(lambda w: jet(w))(U)
    xs = Y[:, 0]
    ys = Y[:, 1] if Y.shape[1] > 1 else jnp.zeros_like(xs)
    return jnp.array(xs), jnp.array(ys)

  # Compute all coordinate curves first to determine bounds
  all_xs = []
  all_ys = []
  curves_family1 = []
  curves_family2 = []

  for v in uvs:
    xs, ys = eval_line(True, float(v))
    curves_family1.append((xs, ys))
    all_xs.append(xs)
    all_ys.append(ys)

  for u in uvs:
    xs, ys = eval_line(False, float(u))
    curves_family2.append((xs, ys))
    all_xs.append(xs)
    all_ys.append(ys)

  # Use provided limits or compute from coordinate curves with padding
  if xlim is not None:
    x_min, x_max = xlim
  else:
    all_xs = jnp.concatenate(all_xs)
    x_min, x_max = float(all_xs.min()), float(all_xs.max())
    x_pad = 0.1 * (x_max - x_min)
    x_min -= x_pad
    x_max += x_pad

  if ylim is not None:
    y_min, y_max = ylim
  else:
    all_ys = jnp.concatenate(all_ys)
    y_min, y_max = float(all_ys.min()), float(all_ys.max())
    y_pad = 0.1 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

  # Create figure or use provided axes
  fig = None
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)

  # Draw heatmap (if enabled)
  if draw_heatmap:
    heatmap_xs = jnp.linspace(x_min, x_max, num_heatmap)
    heatmap_ys = jnp.linspace(y_min, y_max, num_heatmap)
    X, Y = jnp.meshgrid(heatmap_xs, heatmap_ys)
    points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    # Evaluate log density at each point
    log_densities = jax.vmap(log_density_fn)(points)
    densities = jnp.exp(log_densities)
    Z = densities.reshape(X.shape)

    # Plot heatmap
    im = ax.imshow(
        Z,
        extent=[x_min, x_max, y_min, y_max],
        origin='lower',
        cmap=cmap,
        aspect='equal',
        alpha=0.7,
    )

    # Add colorbar
    if fig is not None:
      cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='density')

  # Plot coordinate lines (if enabled)
  if draw_grid:
    for xs, ys in curves_family1:
      ax.plot(xs, ys, color=line_color_1, linewidth=linewidth, alpha=alpha)

    for xs, ys in curves_family2:
      ax.plot(xs, ys, color=line_color_2, linewidth=linewidth, alpha=alpha)

  # Mark the basepoint
  x0, y0 = float(oct.p[0]), float(oct.p[1]) if dim > 1 else 0.0
  ax.scatter([x0], [y0], c=basepoint_color, s=40, zorder=10, edgecolors='white', linewidths=1.5)

  # Draw basis vectors
  if draw_basis_vectors:
    J = jacobian.value
    if J.ndim >= 2:
      v1 = J[:2, 0] if J.shape[1] >= 1 else jnp.zeros((2,))
      v2 = J[:2, 1] if J.shape[1] >= 2 else jnp.zeros((2,))

      for i, (vx, vy) in enumerate([(float(v1[0]), float(v1[1])),
                                     (float(v2[0]), float(v2[1]))]):
        dx = basis_vector_scale * vx
        dy = basis_vector_scale * vy
        arrow = mpatches.FancyArrowPatch(
            (x0, y0), (x0 + dx, y0 + dy),
            arrowstyle='-|>',
            mutation_scale=15,
            linewidth=2.0,
            color=arrow_color,
            zorder=15,
        )
        ax.add_patch(arrow)

  # Clean up the plot
  ax.set_aspect('equal', 'box')
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(y_min, y_max)

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(True)
  ax.spines['left'].set_visible(True)
  ax.spines['bottom'].set_color('#888888')
  ax.spines['left'].set_color('#888888')
  ax.spines['bottom'].set_linewidth(0.8)
  ax.spines['left'].set_linewidth(0.8)

  ax.set_xlabel('x1', fontsize=12, labelpad=5)
  ax.set_ylabel('x2', fontsize=12, labelpad=5)

  if title is not None:
    ax.set_title(title, fontsize=14, fontweight='light', pad=20)

  if fig is not None:
    plt.tight_layout()

    if savepath is not None:
      fig.savefig(savepath, bbox_inches='tight', dpi=200, facecolor='white', edgecolor='none')

    if show:
      plt.show()

  return fig, ax


def _create_example_octs():
  """Create example OCT objects for visualization and analysis."""
  dim = 2
  p = jnp.zeros(dim)
  U = jnp.eye(dim)
  log_s = jnp.array([1.0, 0.0])

  octs = {}

  # OCT 1: positive beta_12
  beta_1 = jnp.array([[1.0, 2.0], [0.0, 0.0]])
  octs['oct_1'] = LocalOCT(p, U, log_s, beta_1, _compute_dbeta_from_beta(beta_1))

  # OCT 2: negative beta_12
  beta_2 = jnp.array([[1.0, -2.0], [0.0, 0.0]])
  octs['oct_2'] = LocalOCT(p, U, log_s, beta_2, _compute_dbeta_from_beta(beta_2))

  # OCT 3: beta_11 = 2 only
  beta_3 = jnp.array([[2.0, 0.0], [0.0, 0.0]])
  octs['oct_3'] = LocalOCT(p, U, log_s, beta_3, _compute_dbeta_from_beta(beta_3))

  # OCT 4: beta_22 = 2 only
  beta_4 = jnp.array([[0.0, 0.0], [0.0, 2.0]])
  octs['oct_4'] = LocalOCT(p, U, log_s, beta_4, _compute_dbeta_from_beta(beta_4))

  # OCT 5: symmetric off-diagonal beta
  beta_5 = jnp.array([[0.0, 1.0], [1.0, 0.0]])
  octs['oct_5'] = LocalOCT(p, U, log_s, beta_5, _compute_dbeta_from_beta(beta_5))

  # OCT 6: mixed diagonal + off-diagonal
  beta_6 = jnp.array([[2.0, 1.0], [1.0, 0.0]])
  octs['oct_6'] = LocalOCT(p, U, log_s, beta_6, _compute_dbeta_from_beta(beta_6))

  return octs, p, dim


def _get_z2_zero_curve(oct: LocalOCT, span: float, num: int = 100):
  """Get the coordinate curve where z2=0 (varying z1)."""
  jacobian = oct.get_jacobian()
  jet = Jet(value=oct.p, gradient=jacobian.value, hessian=jacobian.gradient)
  z1_vals = jnp.linspace(-span, span, num)
  z_coords = jnp.stack([z1_vals, jnp.zeros_like(z1_vals)], axis=-1)
  x_coords = jax.vmap(lambda w: jet(w))(z_coords)
  return x_coords[:, 0], x_coords[:, 1]


def _map_z_to_x(oct: LocalOCT, z: Float[Array, "M N"]) -> Float[Array, "M N"]:
  """Map points z in coordinate space to x in ambient space via Taylor expansion."""
  jac = oct.get_jacobian()
  J, H, T = jac.value, jac.gradient, jac.hessian
  linear = jnp.einsum('ij,mj->mi', J, z)
  quadratic = 0.5 * jnp.einsum('ijk,mj,mk->mi', H, z, z)
  cubic = (1/6) * jnp.einsum('ijkl,mj,mk,ml->mi', T, z, z, z)
  return oct.p + linear + quadratic + cubic


def run_plots():
  """Generate all OCT visualization plots."""
  import matplotlib.pyplot as plt

  octs, p, dim = _create_example_octs()
  oct_1, oct_2 = octs['oct_1'], octs['oct_2']
  oct_3, oct_4 = octs['oct_3'], octs['oct_4']
  oct_5, oct_6 = octs['oct_5'], octs['oct_6']

  span = 0.3

  # === Figure 1: Coordinate grids with basis vectors ===
  fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))

  plot_oct_grid(oct_1, span=span, num=25, basis_vector_scale=0.12,
                show=False, ax=axes1[0], title=r'$\beta_{12} = +2$')
  plot_oct_grid(oct_2, span=span, num=25, basis_vector_scale=0.12,
                show=False, ax=axes1[1], title=r'$\beta_{12} = -2$')

  for ax, oct in [(axes1[0], oct_1), (axes1[1], oct_2)]:
    xs, ys = _get_z2_zero_curve(oct, span, num=100)
    ax.plot(xs, ys, color='#E67E22', linewidth=2.5, alpha=1.0, zorder=5)

  plt.tight_layout()
  fig1.savefig('oct_side_by_side.pdf', bbox_inches='tight', facecolor='white')
  print('Saved oct_side_by_side.pdf')

  # === Figure 2: Coordinate grids with samples ===
  sample_std = 0.12
  n_samples = 5000

  key = random.PRNGKey(42)
  z_samples = random.normal(key, shape=(n_samples, dim)) * sample_std

  x_samples_1 = _map_z_to_x(oct_1, z_samples)
  x_samples_2 = _map_z_to_x(oct_2, z_samples)

  fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

  plot_oct_grid(oct_1, span=span, num=25, draw_basis_vectors=False,
                show=False, ax=axes2[0], title=r'$\beta_{12} = +2$')
  xlim1, ylim1 = axes2[0].get_xlim(), axes2[0].get_ylim()

  plot_oct_grid(oct_2, span=span, num=25, draw_basis_vectors=False,
                show=False, ax=axes2[1], title=r'$\beta_{12} = -2$')
  xlim2, ylim2 = axes2[1].get_xlim(), axes2[1].get_ylim()

  sample_color = '#F4A024'
  axes2[0].scatter(x_samples_1[:, 0], x_samples_1[:, 1],
                   c=sample_color, s=5, alpha=0.85, edgecolors='#1a1a2e', linewidths=0.5, zorder=20)
  axes2[0].set_xlim(xlim1)
  axes2[0].set_ylim(ylim1)

  axes2[1].scatter(x_samples_2[:, 0], x_samples_2[:, 1],
                   c=sample_color, s=5, alpha=0.85, edgecolors='#1a1a2e', linewidths=0.5, zorder=20)
  axes2[1].set_xlim(xlim2)
  axes2[1].set_ylim(ylim2)

  plt.tight_layout()
  plt.show()

  # === Figure 3: Comparing beta_11 vs beta_22 ===
  fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

  plot_oct_grid(oct_3, span=span, num=25, basis_vector_scale=0.12,
                show=False, ax=axes3[0], title=r'$\beta_{11} = 2$')
  plot_oct_grid(oct_4, span=span, num=25, basis_vector_scale=0.12,
                show=False, ax=axes3[1], title=r'$\beta_{22} = 2$')

  for ax, oct in [(axes3[0], oct_3), (axes3[1], oct_4)]:
    xs, ys = _get_z2_zero_curve(oct, span, num=100)
    ax.plot(xs, ys, color='#E67E22', linewidth=2.5, alpha=1.0, zorder=5)

  plt.tight_layout()
  fig3.savefig('oct_diagonal_beta.pdf', bbox_inches='tight', facecolor='white')
  print('Saved oct_diagonal_beta.pdf')
  plt.show()

  # === Figure 4: Symmetric off-diagonal vs mixed ===
  fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))

  plot_oct_grid(oct_5, span=span, num=25, basis_vector_scale=0.12,
                show=False, ax=axes4[0], title=r'$\beta_{12} = \beta_{21} = 1$')
  plot_oct_grid(oct_6, span=span, num=25, basis_vector_scale=0.12,
                show=False, ax=axes4[1], title=r'$\beta_{11} = 2, \beta_{12} = \beta_{21} = 1$')

  for ax, oct in [(axes4[0], oct_5), (axes4[1], oct_6)]:
    xs, ys = _get_z2_zero_curve(oct, span, num=100)
    ax.plot(xs, ys, color='#E67E22', linewidth=2.5, alpha=1.0, zorder=5)

  plt.tight_layout()
  fig4.savefig('oct_symmetric_beta.pdf', bbox_inches='tight', facecolor='white')
  print('Saved oct_symmetric_beta.pdf')
  plt.show()


def run_debug():
  """Run coordinate frame analysis and debugging."""
  octs, p, dim = _create_example_octs()
  oct_6 = octs['oct_6']

  jac = oct_6.get_jacobian()
  standard_basis = get_standard_basis(p)
  coord_frame = basis_to_frame(standard_basis)
  coord_frame = change_coordinates(coord_frame, jac)

  # Check that this is a coordinate frame
  lb = get_lie_bracket_between_frame_pairs(coord_frame)
  # Check that the metric is diagonalized
  identity_jet = get_identity_jet(dim)
  g = RiemannianMetric(basis=standard_basis, components=identity_jet)
  g_in_frame = change_basis(g, coord_frame.basis)
  connection = get_levi_civita_connection(g_in_frame)
  riemann_curvature_tensor = get_riemann_curvature_tensor(connection)
  R: Jet = riemann_curvature_tensor.components

  import pdb; pdb.set_trace()


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description='OCT visualization and analysis')
  parser.add_argument('mode', choices=['plot', 'debug'],
                      help='Mode to run: "plot" generates visualization plots, "debug" runs coordinate frame analysis')

  args = parser.parse_args()


  p = random.normal(random.PRNGKey(42), (2,))
  key = random.PRNGKey(42)
  oct = create_local_oct(p, key)


  frame: Frame = oct.get_coordinate_frame()
  lb = get_lie_bracket_between_frame_pairs(frame)
  import pdb; pdb.set_trace()
  assert jnp.allclose(lb.components.value, 0.0)
  assert jnp.allclose(lb.components.gradient, 0.0)
  exit()

  if args.mode == 'plot':
    run_plots()
  elif args.mode == 'debug':
    run_debug()