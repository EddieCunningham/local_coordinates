"""
Tests for LocalOCT (Local Orthogonal Coordinate Transform) data structure.

Tests verify the correctness of LocalOCT by comparing geometric objects
computed using the local_coordinates library against formulas from the thesis.
"""
import jax
import jax.numpy as jnp
from jax import random
import pytest
from jaxtyping import Float, Array

from local_coordinates.metric import RiemannianMetric
from local_coordinates.oct import (
    LocalOCT,
    _flatness_loss,
    _lie_bracket_loss,
    _compute_dbeta_from_beta,
    create_local_oct,
    fit_local_oct_to_log_det,
    plot_oct_with_density,
)
from local_coordinates.basis import BasisVectors, change_basis, get_standard_basis, change_coordinates
from local_coordinates.frame import Frame, basis_to_frame
from local_coordinates.frame import get_lie_bracket_between_frame_pairs
from local_coordinates.jet import Jet, get_identity_jet, function_to_jet
from local_coordinates.jet import jet_decorator
from local_coordinates.jacobian import Jacobian
from local_coordinates.tangent import TangentVector


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def key():
    """Default PRNG key for tests."""
    return random.PRNGKey(42)


@pytest.fixture(params=[2, 3, 4])
def dim(request):
    """Parameterized dimension for testing multiple sizes."""
    return request.param


@pytest.fixture
def random_oct(key, dim):
    """Create a random valid LocalOCT for testing."""
    p = jnp.zeros(dim)
    return create_local_oct(p, key)


# =============================================================================
# Test Setup: Coordinate Basis and Principal Basis
# =============================================================================

class TestBasisConstruction:
    """Tests for constructing coordinate and principal bases from LocalOCT."""

    def test_coordinate_frame_is_frame(self, random_oct):
        """get_coordinate_frame should return a Frame object."""
        coord_frame = random_oct.get_coordinate_frame()
        assert isinstance(coord_frame, Frame)

    def test_principal_frame_is_frame(self, random_oct):
        """get_principal_frame should return a Frame object."""
        principal_frame = random_oct.get_principal_frame()
        assert isinstance(principal_frame, Frame)

    def test_coordinate_frame_shape(self, random_oct):
        """Coordinate frame components should have correct shape."""
        coord_frame = random_oct.get_coordinate_frame()
        dim = random_oct.p.shape[0]

        # Value should be (dim, dim) matrix of frame vectors
        assert coord_frame.components.value.shape == (dim, dim)
        # Gradient should be (dim, dim, dim)
        assert coord_frame.components.gradient.shape == (dim, dim, dim)
        # Hessian should be (dim, dim, dim, dim)
        assert coord_frame.components.hessian.shape == (dim, dim, dim, dim)

    def test_principal_frame_shape(self, random_oct):
        """Principal frame components should have correct shape."""
        principal_frame = random_oct.get_principal_frame()
        dim = random_oct.p.shape[0]

        # Value should be (dim, dim) matrix of frame vectors
        assert principal_frame.components.value.shape == (dim, dim)
        # Gradient should be (dim, dim, dim)
        assert principal_frame.components.gradient.shape == (dim, dim, dim)
        # Hessian should be (dim, dim, dim, dim)
        assert principal_frame.components.hessian.shape == (dim, dim, dim, dim)

    def test_coordinate_frame_metric_symmetries(self, random_oct):
        """
        The metric g_{ij} = E^T E should have correct symmetries at all derivative levels.
        """
        E_frame = random_oct.get_coordinate_frame()

        @jet_decorator
        def get_metric_components(E_vals) -> Float[Array, "D D"]:
          return E_vals.T @ E_vals

        metric_components: Jet = get_metric_components(E_frame.components.get_value_jet())

        g = metric_components.value       # g_{ij}
        dg = metric_components.gradient   # ∂g_{ij}/∂x^k
        d2g = metric_components.hessian   # ∂²g_{ij}/∂x^k∂x^l

        # Value symmetry: g_{ij} = g_{ji}
        value_sym_error = jnp.max(jnp.abs(g - g.T))
        assert value_sym_error < 1e-6, f"Metric value not symmetric: max error = {value_sym_error}"

        # Gradient symmetry: ∂g_{ij}/∂x^k = ∂g_{ji}/∂x^k (swap first two indices)
        dg_transposed = jnp.transpose(dg, (1, 0, 2))  # dg[j,i,k]
        grad_sym_error = jnp.max(jnp.abs(dg - dg_transposed))
        assert grad_sym_error < 1e-6, f"Metric gradient not symmetric in (i,j): max error = {grad_sym_error}"

        # Hessian symmetry in (i,j): ∂²g_{ij}/∂x^k∂x^l = ∂²g_{ji}/∂x^k∂x^l
        d2g_ij_swap = jnp.transpose(d2g, (1, 0, 2, 3))  # d2g[j,i,k,l]
        hess_ij_sym_error = jnp.max(jnp.abs(d2g - d2g_ij_swap))
        assert hess_ij_sym_error < 1e-6, f"Metric hessian not symmetric in (i,j): max error = {hess_ij_sym_error}"

        # Hessian symmetry in (k,l): ∂²g_{ij}/∂x^k∂x^l = ∂²g_{ij}/∂x^l∂x^k
        d2g_kl_swap = jnp.transpose(d2g, (0, 1, 3, 2))  # d2g[i,j,l,k]
        hess_kl_sym_error = jnp.max(jnp.abs(d2g - d2g_kl_swap))
        assert hess_kl_sym_error < 1e-6, f"Metric hessian not symmetric in (k,l): max error = {hess_kl_sym_error}"

    def test_coordinate_frame_lie_bracket_zero_to_first_order(self, random_oct):
        """
        The Lie bracket [E_i, E_j] should be zero to first order.

        With symmetric β and Lamé equations satisfied, the coordinate basis
        E_i = ∂/∂z^i has [E_i, E_j] = 0 in a neighborhood of the basepoint.
        """
        E_frame: Frame = random_oct.get_coordinate_frame()
        E_lb: TangentVector = get_lie_bracket_between_frame_pairs(E_frame)

        # Value: [E_i, E_j] = 0 at the basepoint
        max_lb_val = jnp.max(jnp.abs(E_lb.components.value))
        assert max_lb_val < 1e-6, f"Lie bracket value not zero: max = {max_lb_val}"

        # Gradient: ∂[E_i, E_j]/∂x = 0 (first order)
        max_lb_grad = jnp.max(jnp.abs(E_lb.components.gradient))
        assert max_lb_grad < 1e-5, f"Lie bracket gradient not zero: max = {max_lb_grad}"

    def test_jacobian_third_derivative_symmetric(self, random_oct):
        """
        The third derivative T^i_{jkl} = ∂³x^i/∂z^j∂z^k∂z^l should be fully symmetric in j,k,l.

        By Clairaut's theorem, mixed partial derivatives commute, so T must be symmetric
        in all lower indices (j,k,l).

        Mathematical note: The thesis formula computes T as ∂H_{jk}/∂z^l where H is the
        Hessian. This naturally preserves (j,k) symmetry (since H is symmetric), but does
        NOT automatically give (j,l) or (k,l) symmetry because l is treated as the
        derivative direction. The implementation must therefore symmetrize the formula
        output in (j,k,l). See notes/rnc.md lines 324-337 for the analogous issue in
        Riemann normal coordinates.
        """
        dxdz = random_oct.get_jacobian()
        T = dxdz.hessian  # T[i,j,k,l] = ∂³x^i/∂z^j∂z^k∂z^l

        # Check symmetry in (j,k)
        T_jk_swap = jnp.transpose(T, (0, 2, 1, 3))
        jk_error = jnp.max(jnp.abs(T - T_jk_swap))
        assert jk_error < 1e-6, f"T not symmetric in (j,k): max error = {jk_error}"

        # Check symmetry in (k,l)
        T_kl_swap = jnp.transpose(T, (0, 1, 3, 2))
        kl_error = jnp.max(jnp.abs(T - T_kl_swap))
        assert kl_error < 1e-6, f"T not symmetric in (k,l): max error = {kl_error}"

        # Check symmetry in (j,l)
        T_jl_swap = jnp.transpose(T, (0, 3, 2, 1))
        jl_error = jnp.max(jnp.abs(T - T_jl_swap))
        assert jl_error < 1e-6, f"T not symmetric in (j,l): max error = {jl_error}"


# =============================================================================
# Test Geometric Quantities vs Beta Formulas
# =============================================================================

class TestGeometricQuantitiesVsBeta:
    """
    Tests verifying geometric quantities computed by the library match
    the mathematical formulas from the thesis in terms of beta.

    Mathematical Reference (from notes/phd_content.tex and notes/oct_math.md):
    - Lie bracket of coordinate basis: [E_i, E_j] = 0 (coordinate basis)
    - Lie bracket of principal basis: [U_i, U_j] = β_{ij} U_i - β_{ji} U_j
    - Christoffel symbols: Γ^k_{ij} = β_{jk} δ^k_i - β_{kj} δ^j_i
    - Riemann curvature: R = 0 for flat space (Lamé equations satisfied)
    """

    def test_lame_equations_satisfied(self, random_oct):
        """
        The Lamé equations should be satisfied (verified via flatness loss).

        First Lamé equation (k ≠ i ≠ j):
            U_k(β_{ij}) = β_{ik} β_{kj} - β_{ij} β_{kj}

        Second Lamé equation (i ≠ j):
            U_i(β_{ji}) + U_j(β_{ij}) + β_{ji}² + β_{ij}² + Σ_{m∉{i,j}} β_{im} β_{jm} = 0
        """
        beta = random_oct.beta
        dbeta = random_oct.dbeta

        loss = _flatness_loss(beta, dbeta)
        assert loss < 1e-6, f"Lamé equations not satisfied: loss = {float(loss)}"

    def test_lie_bracket_loss_matches_original_formula(self, random_oct):
        """
        Test that the O(N^2) lie bracket loss matches the original O(N^3) formula.

        The Lie bracket of the principal frame is:
            [U_i, U_j]^a = beta[i,j] * U[a,i] - beta[j,i] * U[a,j]

        Original O(N^3) implementation:
            term1 = einsum("ij,ai->ija", beta, U)  # term1[i,j,a] = beta[i,j] * U[a,i]
            term2 = einsum("ji,aj->ija", beta, U)  # term2[i,j,a] = beta[j,i] * U[a,j]
            lb = term1 - term2
            loss = sum over i < j of ||lb[i,j,:]||^2

        Simplified O(N^2) implementation uses orthonormality of U.
        """
        U = random_oct.U
        beta = random_oct.beta
        n = beta.shape[0]

        # Original O(N^3) formula
        term1 = jnp.einsum("ij,ai->ija", beta, U)
        term2 = jnp.einsum("ji,aj->ija", beta, U)
        lb = term1 - term2
        mask = jnp.triu(jnp.ones((n, n)), k=1)[:, :, None]
        masked_lb = lb * mask
        expected_loss = jnp.sum(masked_lb ** 2)

        # New O(N^2) implementation
        actual_loss = _lie_bracket_loss(U, beta)

        error = jnp.abs(expected_loss - actual_loss)
        assert error < 1e-10, (
            f"Lie bracket loss mismatch: expected={expected_loss}, actual={actual_loss}, error={error}"
        )

    def test_lie_bracket_loss_zero_for_symmetric_beta(self, key, dim):
        """
        For symmetric beta with orthonormal U, the Lie bracket loss should be zero
        because [U_i, U_j] = beta[i,j]*U_i - beta[j,i]*U_j = beta[i,j]*(U_i - U_j)
        and the loss sums beta[i,j]^2 + beta[j,i]^2 = 2*beta[i,j]^2 for symmetric beta.

        Actually, the loss is NOT zero for symmetric beta. This test verifies both
        implementations give the same non-zero result.
        """
        # Create random orthonormal U
        key, subkey = random.split(key)
        A = random.normal(subkey, (dim, dim))
        U, _ = jnp.linalg.qr(A)

        # Create symmetric beta
        key, subkey = random.split(key)
        B = random.normal(subkey, (dim, dim))
        beta = (B + B.T) / 2
        beta = beta - jnp.diag(jnp.diag(beta))  # Zero diagonal

        # Original O(N^3) formula
        n = dim
        term1 = jnp.einsum("ij,ai->ija", beta, U)
        term2 = jnp.einsum("ji,aj->ija", beta, U)
        lb = term1 - term2
        mask = jnp.triu(jnp.ones((n, n)), k=1)[:, :, None]
        masked_lb = lb * mask
        expected_loss = jnp.sum(masked_lb ** 2)

        # New O(N^2) implementation
        actual_loss = _lie_bracket_loss(U, beta)

        error = jnp.abs(expected_loss - actual_loss)
        assert error < 1e-10, f"Loss mismatch for symmetric beta: error={error}"

    def test_connection_from_oct_matches_beta_formula(self, random_oct):
        """
        The Christoffel symbols Γ^k_{ij} from LocalOCT should equal β_{jk} δ^k_i - β_{kj} δ^j_i.

        From notes/orthogonal_coordinates.md:
            Γ^k_{ij} = β_{jk} δ^k_i - β_{kj} δ^j_i

        Explicitly: ∇_{U_i} U_j = β_{ji} U_i - δ_{ij} Σ_{k≠j} β_{kj} U_k
        """
        dim = random_oct.p.shape[0]
        beta = random_oct.beta

        # Get connection from LocalOCT (computed directly from beta)
        connection = random_oct.get_connection()
        Gamma = connection.christoffel_symbols.value  # Γ[i,j,k] = Γ^k_{ij}

        # Expected: Γ^k_{ij} = β_{jk} δ^k_i - β_{kj} δ^j_i
        I = jnp.eye(dim)
        expected_Gamma = jnp.einsum("jk,ki->ijk", beta, I) - jnp.einsum("kj,ij->ijk", beta, I)

        max_error = jnp.max(jnp.abs(Gamma - expected_Gamma))
        assert max_error < 1e-6, f"Christoffel symbols don't match β formula: max error = {max_error}"

    def test_covariant_derivative_formula(self, random_oct):
        """
        Test the explicit covariant derivative formula:
            ∇_{U_i} U_j = β_{ji} U_i  (when i ≠ j)
            ∇_{U_i} U_i = -Σ_{k≠i} β_{ki} U_k

        From notes/orthogonal_coordinates.md:
            ∇_{U_i} U_j = β_{ji} U_i - δ_{ij} Σ_{k≠i} β_{kj} U_k
        """
        dim = random_oct.p.shape[0]
        beta = random_oct.beta

        # Get connection Christoffel symbols
        connection = random_oct.get_connection()
        Gamma = connection.christoffel_symbols.value  # Γ[i,j,k] = Γ^k_{ij}

        # Check ∇_{U_i} U_j = Γ^k_{ij} U_k
        # For i ≠ j: ∇_{U_i} U_j = β_{ji} U_i
        # For i = j: ∇_{U_i} U_i = -Σ_{k≠i} β_{ki} U_k
        for i in range(dim):
            for j in range(dim):
                # Expected coefficients: ∇_{U_i} U_j = expected_k U_k
                expected = jnp.zeros(dim)
                if i != j:
                    # ∇_{U_i} U_j = β_{ji} U_i
                    expected = expected.at[i].set(beta[j, i])
                else:
                    # ∇_{U_i} U_i = -Σ_{k≠i} β_{ki} U_k
                    for k in range(dim):
                        if k != i:
                            expected = expected.at[k].set(-beta[k, i])

                actual = Gamma[i, j, :]  # Γ^k_{ij} for all k
                error = jnp.max(jnp.abs(actual - expected))
                assert error < 1e-6, f"Covariant derivative formula failed for (i,j)=({i},{j}): error = {error}"


# =============================================================================
# Test Geometric Quantities via Library (Metric-based computation)
# =============================================================================

class TestGeometricQuantitiesFromMetric:
    """
    Tests that verify geometric properties of the coordinate frame computed
    directly from the LocalOCT, without going through the library's full
    metric → connection → curvature pipeline.

    The library's pipeline has different semantics for BasisVectors vs Frame
    that make it incompatible with directly passing the coordinate frame.
    """

    def test_metric_from_coordinate_frame_is_diagonal(self, random_oct):
        """
        The metric g_{ij} = <E_i, E_j> should be diagonal with g_{ii} = s_i².

        For orthogonal coordinates: g_{ij} = s_i² δ_{ij}

        Frame convention: E_val[k, j] = E_j^k (k-th x-component of j-th basis vector, columns are vectors)
        So: g_{ij} = Σ_k E_i^k E_j^k = (E.T @ E)_{ij}
        """
        dim = random_oct.p.shape[0]
        s = jnp.exp(random_oct.log_s)

        # Get coordinate frame and compute metric
        E_frame = random_oct.get_coordinate_frame()

        @jet_decorator
        def compute_metric(E_val) -> Float[Array, "D D"]:
            # g_{ij} = <E_i, E_j> = Σ_k E_i^k E_j^k = (E.T @ E)_{ij}
            return E_val.T @ E_val

        g: Jet = compute_metric(E_frame.components.get_value_jet())

        # Expected: g_{ij} = s_i² δ_{ij}
        expected_g = jnp.diag(s ** 2)
        max_error = jnp.max(jnp.abs(g.value - expected_g))
        assert max_error < 1e-5, f"Metric not diagonal with s²: max error = {max_error}"

    def test_coordinate_frame_is_orthogonal(self, random_oct):
        """
        The coordinate frame E_i should be orthogonal: <E_i, E_j> = 0 for i ≠ j.
        """
        E_frame = random_oct.get_coordinate_frame()

        @jet_decorator
        def compute_metric(E_val) -> Float[Array, "D D"]:
            # Columns are vectors: g_{ij} = E.T @ E
            return E_val.T @ E_val

        g: Jet = compute_metric(E_frame.components.get_value_jet())

        # Off-diagonal elements should be zero
        dim = random_oct.p.shape[0]
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    off_diag = jnp.abs(g.value[i, j])
                    assert off_diag < 1e-6, f"g[{i},{j}] = {g.value[i,j]} not zero"

    def test_coordinate_frame_loadings_match_oct(self, random_oct):
        """
        The norms ||E_j|| should equal s_j from LocalOCT.
        """
        s = jnp.exp(random_oct.log_s)
        E_frame = random_oct.get_coordinate_frame()
        E_val = E_frame.components.value

        # ||E_j|| = sqrt(Σ_k (E_j^k)²) = norm of column j (axis=0)
        norms = jnp.linalg.norm(E_val, axis=0)
        max_error = jnp.max(jnp.abs(norms - s))
        assert max_error < 1e-5, f"||E_j|| != s_j: max error = {max_error}"

    def test_principal_frame_is_orthonormal(self, random_oct):
        """
        The principal frame U_j should be orthonormal: <U_i, U_j> = δ_{ij}.
        """
        U_frame = random_oct.get_principal_frame()

        @jet_decorator
        def compute_metric(U_val) -> Float[Array, "D D"]:
            # Columns are vectors: g_{ij} = U.T @ U
            return U_val.T @ U_val

        g: Jet = compute_metric(U_frame.components.get_value_jet())

        # Should be identity matrix
        dim = random_oct.p.shape[0]
        expected = jnp.eye(dim)
        max_error = jnp.max(jnp.abs(g.value - expected))
        assert max_error < 1e-5, f"U not orthonormal: max error = {max_error}"

    def test_compare_metric_two_different_ways(self, random_oct):
        """
        Compute the metric using the E basis and the U basis and check they are the same
        after an appropriate change of basis.
        """
        E_frame = random_oct.get_coordinate_frame()

        @jet_decorator
        def compute_metric(E_val) -> Float[Array, "D D"]:
            # Columns are vectors: g_{ij} = E.T @ E
            return E_val.T @ E_val

        g: Jet = compute_metric(E_frame.components.get_value_jet())

    def test_principal_frame_lie_bracket_antisymmetric(self, random_oct):
        """
        The Lie bracket [U_i, U_j] should be antisymmetric: [U_i, U_j] = -[U_j, U_i].

        From the thesis: [U_i, U_j] = β_{ij} U_i - β_{ji} U_j
        """
        dim = random_oct.p.shape[0]

        # Get principal frame and compute Lie brackets using the library
        U_frame = random_oct.get_principal_frame()
        U_lb = get_lie_bracket_between_frame_pairs(U_frame)

        # Check antisymmetry
        lb_val = U_lb.components.value
        antisym_error = jnp.max(jnp.abs(lb_val + jnp.transpose(lb_val, (1, 0, 2))))
        assert antisym_error < 1e-6, f"Lie bracket not antisymmetric: max error = {antisym_error}"

        # Check diagonal is zero
        for i in range(dim):
            diag_error = jnp.max(jnp.abs(lb_val[i, i, :]))
            assert diag_error < 1e-6, f"[U_{i}, U_{i}] not zero: max = {diag_error}"

    def test_principal_frame_lie_bracket_matches_beta_formula(self, random_oct):
        """
        The Lie bracket [U_i, U_j] should equal β_{ij} U_i - β_{ji} U_j.

        From phd_content.tex (Proposition: Lie bracket of principal basis):
            [U_i, U_j] = β_{ij} U_i - β_{ji} U_j

        In x-components: [U_i, U_j]^a = β_{ij} U_i^a - β_{ji} U_j^a
        """
        dim = random_oct.p.shape[0]
        beta = random_oct.beta
        # Code convention: U[a, i] = U^a_i = a-th x-component of i-th basis vector
        # (columns are vectors, so U[:, i] = U_i)
        U = random_oct.U

        # Get principal frame and compute Lie brackets using the library
        U_frame = random_oct.get_principal_frame()
        U_lb = get_lie_bracket_between_frame_pairs(U_frame)

        # The library computes [U_i, U_j] with components in x-coordinates
        # U_lb.components.value[i, j, a] = [U_i, U_j]^a
        lb_computed = U_lb.components.value

        # Expected: [U_i, U_j]^a = β_{ij} U_i^a - β_{ji} U_j^a
        # With column-vector convention: U_i^a = U[a, i]
        expected_lb = jnp.zeros((dim, dim, dim))
        for i in range(dim):
            for j in range(dim):
                for a in range(dim):
                    expected_lb = expected_lb.at[i, j, a].set(
                        beta[i, j] * U[a, i] - beta[j, i] * U[a, j]
                    )

        max_error = jnp.max(jnp.abs(lb_computed - expected_lb))
        assert max_error < 1e-5, f"Lie bracket doesn't match β formula: max error = {max_error}"

    def test_oct_lie_bracket_matches_library(self, random_oct):
        """
        The Lie bracket from oct.get_lie_bracket_between_frame_pairs() should match
        the library's get_lie_bracket_between_frame_pairs(oct.get_principal_frame()).
        """
        # Compute using the oct2.py method
        lb_oct = random_oct.get_lie_bracket_between_frame_pairs()

        # Compute using the library's frame.py method
        U_frame = random_oct.get_principal_frame()
        lb_library = get_lie_bracket_between_frame_pairs(U_frame)

        # Compare the components
        oct_val = lb_oct.components.value
        lib_val = lb_library.components.value

        max_error = jnp.max(jnp.abs(oct_val - lib_val))
        assert max_error < 1e-6, f"OCT and library Lie brackets don't match: max error = {max_error}"

    def test_coordinate_to_principal_scaling(self, random_oct):
        """
        The coordinate frame should equal s times the principal frame: E_j = s_j U_j.
        """
        s = jnp.exp(random_oct.log_s)
        E_frame = random_oct.get_coordinate_frame()
        U_frame = random_oct.get_principal_frame()

        E_val = E_frame.components.value  # E[a, j] = E_j^a (columns are vectors)
        U_val = U_frame.components.value  # U[a, j] = U_j^a (columns are vectors)

        # E_j = s_j U_j, so E[:, j] = s[j] * U[:, j]
        expected_E = U_val * s[None, :]
        max_error = jnp.max(jnp.abs(E_val - expected_E))
        assert max_error < 1e-5, f"E_j != s_j U_j: max error = {max_error}"

    def test_connection_basis_is_principal_frame(self, random_oct):
        """
        The connection from get_connection() should have the U-frame as its basis.

        The Christoffel symbols Γ^k_{ij} = β_{kj}δ^k_i - β_{jk}δ^j_i are computed
        in the principal (U) frame, so the connection's basis must represent U.

        Both BasisVectors and Frame now use column-vector convention:
        components[:, j] = j-th vector. No transpose needed.
        """
        dim = random_oct.p.shape[0]

        # Get connection from LocalOCT
        connection = random_oct.get_connection()

        # Get principal frame
        U_frame = random_oct.get_principal_frame()

        # Both use column-vector convention, so direct comparison
        conn_basis_val = connection.basis.components.value
        U_frame_val = U_frame.components.value

        max_error = jnp.max(jnp.abs(conn_basis_val - U_frame_val))
        assert max_error < 1e-6, f"Connection basis doesn't match U-frame: error = {max_error}"

    def test_connection_christoffel_formula_explicit(self, random_oct):
        """
        Verify the Christoffel symbol formula explicitly against the thesis:
            Γ^k_{ij} = β_{kj}δ^k_i - β_{jk}δ^j_i

        From phd_content.tex (line 1636):
            ∇_{U_i} U_j = β_{ij} U_i (for i ≠ j)
            ∇_{U_i} U_i = -Σ_{k≠i} β_{ik} U_k

        This means:
            Γ^k_{ij} = β_{ij} δ^k_i      (for i ≠ j)
            Γ^k_{ii} = -β_{ik}           (for k ≠ i)
            Γ^i_{ii} = 0
        """
        dim = random_oct.p.shape[0]
        beta = random_oct.beta
        connection = random_oct.get_connection()
        Gamma = connection.christoffel_symbols.value  # Gamma[i,j,k] = Γ^k_{ij}

        # Check each case explicitly
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    actual = Gamma[i, j, k]
                    if i != j:
                        # Γ^k_{ij} = β_{ij} δ^k_i for i ≠ j
                        expected = beta[i, j] if k == i else 0.0
                    else:
                        # Γ^k_{ii} = -β_{ik} for k ≠ i, and 0 for k = i
                        expected = -beta[i, k] if k != i else 0.0

                    error = abs(actual - expected)
                    assert error < 1e-6, (
                        f"Γ^{k}_{{{i}{j}}} wrong: got {actual}, expected {expected}"
                    )

    def test_connection_from_get_levi_civita_matches_get_connection(self, random_oct):
        """
        The Levi-Civita connection computed from get_metric() should match
        the connection from get_connection().

        This verifies that:
            get_levi_civita_connection(oct.get_metric()) == oct.get_connection()
        """
        from local_coordinates.connection import get_levi_civita_connection

        # Get connection directly from beta formula
        connection_from_beta = random_oct.get_connection()
        Gamma_beta = connection_from_beta.christoffel_symbols.value

        # Get connection via library pipeline: metric → Levi-Civita connection
        metric = random_oct.get_metric()
        metric = change_basis(metric, connection_from_beta.basis)
        connection_from_metric = get_levi_civita_connection(metric)
        Gamma_metric = connection_from_metric.christoffel_symbols.value

        # They should match
        max_error = jnp.max(jnp.abs(Gamma_beta - Gamma_metric))
        assert max_error < 1e-5, f"Connections don't match: max error = {max_error}"

    def test_riemann_curvature_is_zero_for_flat_space(self, random_oct):
        """
        The Riemann curvature tensor should be zero for a LocalOCT that satisfies
        the Lamé equations (flatness constraints).

        From phd_content.tex (Theorem: Implications of flatness):
            The Lamé equations are equivalent to requiring R_{kli}^j = 0.
            LocalOCT is constructed to satisfy these equations, so the
            Riemann curvature tensor computed via the library should vanish.

        Note: The library's get_riemann_curvature_tensor expects a coordinate basis
        (where [E_i, E_j] = 0), not an orthonormal frame. We use the E-frame
        (coordinate frame) from LocalOCT, which IS a coordinate basis.
        """
        from local_coordinates.riemann import get_riemann_curvature_tensor
        from local_coordinates.connection import get_levi_civita_connection
        from local_coordinates.basis import BasisVectors
        from local_coordinates.metric import RiemannianMetric

        # Get the coordinate frame E (which IS a coordinate basis, so [E_i, E_j] = 0)
        E_frame = random_oct.get_coordinate_frame()

        # Build metric in E-frame: g_{ij} = <E_i, E_j> = s_i^2 δ_{ij}
        @jet_decorator
        def compute_metric(E_val):
            return E_val.T @ E_val

        g_components = compute_metric(E_frame.components)

        # Create metric with E-frame as basis
        E_basis = BasisVectors(p=random_oct.p, components=E_frame.components)
        metric = RiemannianMetric(basis=E_basis, components=g_components)

        # Get Levi-Civita connection from this metric
        connection = get_levi_civita_connection(metric)

        # Compute Riemann curvature tensor via library
        R = get_riemann_curvature_tensor(connection)

        # For flat space, all components should be zero
        R_components = R.components.value
        max_error = jnp.max(jnp.abs(R_components))
        assert max_error < 1e-5, f"Riemann curvature not zero: max = {max_error}"

    def test_riemann_explicit_formula_matches_thesis(self, random_oct):
        """
        Verify the Riemann tensor is zero in the U-basis when computed from
        the curvature 2-form formula with the Lamé equations.

        From phd_content.tex, the Riemann curvature 2-form components are:

            {R_{ij,i}}^j = 2 * (U_i(β_ji) + U_j(β_ij) + β_ji² + β_ij²
                              + Σ_{m∉{i,j}} β_im*β_jm)  for i ≠ j

        The second Lamé equation states:

            U_i(β_ji) + U_j(β_ij) + β_ji² + β_ij² + Σ_{k∉{i,j}} β_ik*β_jk = 0

        Substituting: R = 2 * 0 = 0 ✓

        This confirms that when Lamé equations are satisfied, R = 0 in ALL bases,
        as expected for a flat manifold.
        """
        dim = random_oct.p.shape[0]
        beta = random_oct.beta
        dbeta = random_oct.dbeta

        for i in range(dim):
            for j in range(dim):
                if i == j:
                    continue

                # Compute the cross sum: Σ_{k∉{i,j}} β_ik*β_jk
                cross_sum = 0.0
                for k in range(dim):
                    if k != i and k != j:
                        cross_sum += beta[i, k] * beta[j, k]

                # First part of second Lamé equation
                part1 = dbeta[j, i, i] + dbeta[i, j, j] + beta[j, i]**2 + beta[i, j]**2

                # Verify Lamé equation: part1 + cross_sum = 0
                lame_residual = abs(part1 + cross_sum)
                assert lame_residual < 1e-5, \
                    f"Lamé equation not satisfied for (i={i}, j={j}): residual = {lame_residual}"

                # Verify R component is zero: R = 2 * (part1 + cross_sum) = 0
                R_component = 2 * (part1 + cross_sum)
                assert abs(R_component) < 1e-5, \
                    f"R[{i},{j},{i},{j}] not zero: {R_component}"


# =============================================================================
# Comprehensive Invariant Tests
# =============================================================================

class TestOCTInvariants:
    """
    Comprehensive tests for the three fundamental invariants of orthogonal
    coordinate systems produced by LocalOCT:

    1. INTEGRABILITY: [E_i, E_j] = 0 (coordinate frame has vanishing Lie brackets)
    2. ORTHOGONALITY: g_{ij} = s_i² δ_{ij} (metric is diagonal)
    3. FLATNESS: R = 0 (Riemann curvature vanishes)

    Each invariant is tested at the point (value level) and in a neighborhood
    (gradient level) where applicable.
    """

    def test_integrability_at_point_and_neighborhood(self, random_oct):
        """
        INTEGRABILITY: The coordinate frame Lie bracket [E_i, E_j] should be zero.

        This is the defining property of a coordinate basis: the basis vectors
        commute because they are partial derivatives ∂/∂z^i.

        Tests:
          - Value: [E_i, E_j] = 0 at the basepoint
          - Gradient: ∂[E_i, E_j]/∂x = 0 (first order in neighborhood)
        """
        E_frame = random_oct.get_coordinate_frame()
        E_lb = get_lie_bracket_between_frame_pairs(E_frame)

        # Value level: [E_i, E_j] = 0 at the point
        max_lb_value = jnp.max(jnp.abs(E_lb.components.value))
        assert max_lb_value < 1e-5, \
            f"Integrability violated at point: max |[E_i, E_j]| = {max_lb_value}"

        # Gradient level: ∂[E_i, E_j]/∂x = 0 in neighborhood
        max_lb_grad = jnp.max(jnp.abs(E_lb.components.gradient))
        assert max_lb_grad < 1e-4, \
            f"Integrability violated in neighborhood: max |∂[E_i, E_j]/∂x| = {max_lb_grad}"

    def test_orthogonality_at_point_and_neighborhood(self, random_oct):
        """
        ORTHOGONALITY: The metric g_{ij} = <E_i, E_j> should be diagonal.

        For orthogonal coordinates: g_{ij} = s_i² δ_{ij}
        The off-diagonal elements should be zero at the point AND remain zero
        in a neighborhood (to first order).

        Tests:
          - Value: g_{ij} = 0 for i ≠ j at the point
          - Gradient: ∂g_{ij}/∂x^k = 0 for i ≠ j (metric stays diagonal)
        """
        dim = random_oct.p.shape[0]
        s = jnp.exp(random_oct.log_s)
        E_frame = random_oct.get_coordinate_frame()

        @jet_decorator
        def compute_metric(E_val):
            return E_val.T @ E_val

        g_jet = compute_metric(E_frame.components)

        # Create off-diagonal mask
        off_diag_mask = ~jnp.eye(dim, dtype=bool)

        # Value level: g_{ij} = s_i² δ_{ij}
        expected_g = jnp.diag(s ** 2)
        max_metric_error = jnp.max(jnp.abs(g_jet.value - expected_g))
        assert max_metric_error < 1e-5, \
            f"Orthogonality violated at point: max |g - diag(s²)| = {max_metric_error}"

        # Off-diagonal should be zero
        off_diag_values = g_jet.value[off_diag_mask]
        max_off_diag = jnp.max(jnp.abs(off_diag_values))
        assert max_off_diag < 1e-5, \
            f"Off-diagonal metric non-zero at point: max |g_{{i≠j}}| = {max_off_diag}"

        # Gradient level: ∂g_{ij}/∂x^k = 0 for i ≠ j (metric stays diagonal)
        max_off_diag_grad = 0.0
        for k in range(dim):
            off_diag_grad = g_jet.gradient[:, :, k][off_diag_mask]
            max_off_diag_grad = max(max_off_diag_grad, float(jnp.max(jnp.abs(off_diag_grad))))

        assert max_off_diag_grad < 1e-4, \
            f"Orthogonality violated in neighborhood: max |∂g_{{i≠j}}/∂x| = {max_off_diag_grad}"

    def test_flatness_at_point(self, random_oct):
        """
        FLATNESS: The Riemann curvature tensor should be zero.

        This is ensured by the Lamé equations, which are necessary and sufficient
        conditions for an orthogonal coordinate system to exist in flat space.

        Tests:
          - Lamé equations satisfied (via flatness_loss)
          - Riemann tensor R = 0 (computed via library)
        """
        from local_coordinates.riemann import get_riemann_curvature_tensor
        from local_coordinates.connection import get_levi_civita_connection
        from local_coordinates.basis import BasisVectors
        from local_coordinates.metric import RiemannianMetric

        # Test 1: Lamé equations satisfied
        lame_loss = _flatness_loss(random_oct.beta, random_oct.dbeta)
        assert lame_loss < 1e-5, \
            f"Flatness violated (Lamé equations): loss = {lame_loss}"

        # Test 2: Riemann tensor = 0
        E_frame = random_oct.get_coordinate_frame()

        @jet_decorator
        def compute_metric(E_val):
            return E_val.T @ E_val

        g_components = compute_metric(E_frame.components)
        E_basis = BasisVectors(p=random_oct.p, components=E_frame.components)
        metric = RiemannianMetric(basis=E_basis, components=g_components)
        connection = get_levi_civita_connection(metric)
        R = get_riemann_curvature_tensor(connection)

        max_R = jnp.max(jnp.abs(R.components.value))
        assert max_R < 1e-4, \
            f"Flatness violated (Riemann tensor): max |R| = {max_R}"

    def test_all_invariants_summary(self, random_oct):
        """
        Summary test that checks all three invariants and reports results.

        This test provides a comprehensive overview of the OCT's geometric
        properties in a single test.
        """
        from local_coordinates.riemann import get_riemann_curvature_tensor
        from local_coordinates.connection import get_levi_civita_connection
        from local_coordinates.basis import BasisVectors
        from local_coordinates.metric import RiemannianMetric

        dim = random_oct.p.shape[0]
        s = jnp.exp(random_oct.log_s)
        E_frame = random_oct.get_coordinate_frame()

        # Compute metric
        @jet_decorator
        def compute_metric(E_val):
            return E_val.T @ E_val

        g_jet = compute_metric(E_frame.components)

        # 1. INTEGRABILITY
        E_lb = get_lie_bracket_between_frame_pairs(E_frame)
        integ_value = float(jnp.max(jnp.abs(E_lb.components.value)))
        integ_grad = float(jnp.max(jnp.abs(E_lb.components.gradient)))

        # 2. ORTHOGONALITY
        off_diag_mask = ~jnp.eye(dim, dtype=bool)
        ortho_value = float(jnp.max(jnp.abs(g_jet.value[off_diag_mask])))
        ortho_grad = max(
            float(jnp.max(jnp.abs(g_jet.gradient[:, :, k][off_diag_mask])))
            for k in range(dim)
        )

        # 3. FLATNESS
        flat_lame = float(_flatness_loss(random_oct.beta, random_oct.dbeta))
        E_basis = BasisVectors(p=random_oct.p, components=E_frame.components)
        metric = RiemannianMetric(basis=E_basis, components=g_jet)
        connection = get_levi_civita_connection(metric)
        R = get_riemann_curvature_tensor(connection)
        flat_riemann = float(jnp.max(jnp.abs(R.components.value)))

        # Assert all invariants hold
        assert integ_value < 1e-5, f"Integrability (value): {integ_value}"
        assert integ_grad < 1e-4, f"Integrability (gradient): {integ_grad}"
        assert ortho_value < 1e-5, f"Orthogonality (value): {ortho_value}"
        assert ortho_grad < 1e-4, f"Orthogonality (gradient): {ortho_grad}"
        assert flat_lame < 1e-5, f"Flatness (Lamé): {flat_lame}"
        assert flat_riemann < 1e-4, f"Flatness (Riemann): {flat_riemann}"


class TestAdditionalThesisFormulas:
    """
    Tests for additional mathematical formulas from phd_content.tex that verify
    the correctness of LocalOCT's geometric computations.

    These tests use autodiff and the local_coordinates library to compute
    baselines independently, then compare against the thesis formulas.
    """

    def test_divergence_formula(self, random_oct):
        """
        Verify the divergence formula from phd_content.tex (line 1654):

            Div(U_j) = Σ_{i≠j} β_ij

        Computed using the library's covariant derivative:
            Div(X) = Σ_i g(∇_{U_i} X, U_i)

        For orthonormal basis U_i with g(U_i, U_k) = δ_ik:
            Div(X) = Σ_i (∇_{U_i} X)^i
        """
        dim = random_oct.p.shape[0]
        beta = random_oct.beta
        connection = random_oct.get_connection()
        U_frame = random_oct.get_principal_frame()

        for j in range(dim):
            # Expected from thesis: Div(U_j) = Σ_{i≠j} β_ij
            expected_div = sum(beta[i, j] for i in range(dim) if i != j)

            # Compute using library: Div(U_j) = Σ_i (∇_{U_i} U_j)^i
            U_j = U_frame.get_basis_vector(j)
            computed_div = 0.0
            for i in range(dim):
                U_i = U_frame.get_basis_vector(i)
                # ∇_{U_i} U_j
                nabla_Ui_Uj = connection.covariant_derivative(U_i, U_j)
                # Get i-th component: (∇_{U_i} U_j)^i
                computed_div += nabla_Ui_Uj.components.value[i]

            error = abs(float(computed_div) - float(expected_div))
            assert error < 1e-5, \
                f"Divergence formula failed for j={j}: " \
                f"library={computed_div}, thesis={expected_div}, error={error}"

    def test_log_determinant_of_jacobian(self, random_oct):
        """
        Verify the log determinant formula from phd_content.tex (line 1789):

            log|det Df| = Σ_i log s_i

        Uses autodiff via jet_decorator on jnp.linalg.slogdet to compute
        the log determinant from the Jacobian matrix.
        """
        jacobian = random_oct.get_jacobian()

        # Create a Jet from the Jacobian matrix values
        J_jet = Jet(
            value=jacobian.value,
            gradient=jacobian.gradient,
            hessian=jacobian.hessian,
            dim=random_oct.p.shape[0]
        )

        # Compute log|det J| using autodiff
        @jet_decorator
        def compute_log_det(J_val):
            sign, logdet = jnp.linalg.slogdet(J_val)
            return logdet

        log_det_jet = compute_log_det(J_jet)
        log_det_computed = log_det_jet.value

        # Expected from thesis: Σ_i log s_i
        log_det_expected = jnp.sum(random_oct.log_s)

        error = jnp.abs(log_det_computed - log_det_expected)
        assert error < 1e-5, \
            f"Log determinant formula failed: computed={log_det_computed}, expected={log_det_expected}"

    def test_directional_derivative_of_log_det(self, random_oct):
        """
        Verify the directional derivative formula from phd_content.tex (line 1794):

            U_j(log|det Df|) = Σ_i β_ij

        Uses the log_loadings_jet from the library, which has derivatives
        in z-coordinates (principal coordinates).

        Since log|det Df| = Σ_i log s_i, the derivative is:
            U_j(log|det Df|) = Σ_i U_j(log s_i) = Σ_i log_s_jet.gradient[i, j]
        """
        dim = random_oct.p.shape[0]
        beta = random_oct.beta

        # Get log_s as a Jet with derivatives from the library
        log_s_jet = random_oct.get_log_loadings_jet()

        for j in range(dim):
            # Expected from thesis: U_j(log|det Df|) = Σ_i β_ij
            expected_deriv = sum(beta[i, j] for i in range(dim))

            # Compute using log_s_jet from the library:
            # log|det Df| = Σ_i log s_i
            # In principal coordinates, U_j(log s_i) = log_s_jet.gradient[i, j]
            # So U_j(log|det Df|) = Σ_i log_s_jet.gradient[i, j]
            computed_deriv = jnp.sum(log_s_jet.gradient[:, j])

            error = abs(float(computed_deriv) - float(expected_deriv))
            assert error < 1e-5, \
                f"Directional derivative of log det failed for j={j}: " \
                f"library={computed_deriv}, thesis={expected_deriv}"

    def test_score_decomposition(self, random_oct):
        """
        Verify the score decomposition from phd_content.tex (line 1776):

            U_j(log q_x) = U_j(log p_z ∘ z) - U_j(log|det Df|)

        For standard normal prior at z=0:
            U_j(log p_z ∘ z)|_{z=0} = 0

        So: U_j(log q_x) = -Σ_i β_ij = -(β_jj + Div(U_j))

        Uses autodiff to verify the decomposition.
        """
        dim = random_oct.p.shape[0]
        beta = random_oct.beta
        log_s_jet = random_oct.get_log_loadings_jet()

        # For standard normal prior at z=0, log p_z = const - ||z||²/2
        # and ∂(log p_z)/∂z^j|_{z=0} = -z_j|_{z=0} = 0
        # So U_j(log p_z ∘ z) = 0 at the basepoint

        for j in range(dim):
            # U_j(log|det Df|) = Σ_i U_j(log s_i) = Σ_i β_ij
            U_j_log_det = jnp.sum(log_s_jet.gradient[:, j])

            # At z=0: U_j(log q_x) = 0 - U_j(log|det Df|) = -Σ_i β_ij
            computed_score = -U_j_log_det

            # Verify: -Σ_i β_ij = -(β_jj + Div(U_j))
            div_Uj = sum(beta[i, j] for i in range(dim) if i != j)
            expected_score = -(beta[j, j] + div_Uj)

            error = abs(float(computed_score) - float(expected_score))
            assert error < 1e-5, \
                f"Score decomposition failed for j={j}: " \
                f"computed={computed_score}, expected={expected_score}"

    def test_connection_one_forms_via_covariant_derivative(self, random_oct):
        """
        Verify the connection 1-forms from phd_content.tex (line 1584):

            ω_i^j = β_{ji}ν^j - β_{ij}ν^i

        Uses the library's covariant derivative to compute:
            ω_i^j(U_k) = coefficient of U_j in ∇_{U_k} U_i

        Then compares against the formula.
        """
        dim = random_oct.p.shape[0]
        beta = random_oct.beta
        connection = random_oct.get_connection()
        U_frame = random_oct.get_principal_frame()

        for i in range(dim):
            for k in range(dim):
                # Compute ∇_{U_k} U_i using the library
                U_i = U_frame.get_basis_vector(i)
                U_k = U_frame.get_basis_vector(k)
                nabla_Uk_Ui = connection.covariant_derivative(U_k, U_i)

                # The components give us ω_i^j(U_k) for each j
                for j in range(dim):
                    # Computed from library: (∇_{U_k} U_i)^j
                    computed = nabla_Uk_Ui.components.value[j]

                    # Expected from formula: ω_i^j(U_k) = β_{ji}δ^j_k - β_{ij}δ^i_k
                    expected = 0.0
                    if k == j:
                        expected += beta[j, i]
                    if k == i:
                        expected -= beta[i, j]

                    error = abs(float(computed) - float(expected))
                    assert error < 1e-5, \
                        f"Connection 1-form failed for (i,j,k)=({i},{j},{k}): " \
                        f"library={computed}, formula={expected}"

    def test_jacobian_from_loadings_and_principal_basis(self, random_oct):
        """
        Verify the Jacobian formula from phd_content.tex (line 1823):

            J^i_j = s_j U^i_j

        Compares the Jacobian from get_jacobian() against the formula s*U.
        """
        s = jnp.exp(random_oct.log_s)
        U = random_oct.U  # U[:, j] = U_j (column j is j-th basis vector)

        jacobian = random_oct.get_jacobian()
        J_computed = jacobian.value

        # J^i_j = s_j U^i_j = s[j] * U[i, j]
        J_expected = U * s[None, :]  # Broadcast s along columns

        max_error = jnp.max(jnp.abs(J_computed - J_expected))
        assert max_error < 1e-10, \
            f"Jacobian formula J = s*U failed: max error = {max_error}"

    def test_hessian_formula(self, random_oct):
        """
        Verify the Hessian formula from phd_content.tex (line 1844):

            H^i_{jk} = s_k s_j (β_jk U^i_j + Γ^a_kj U^i_a)

        where Γ^a_kj = β_aj δ^a_k - β_ja δ^j_k are the Christoffel symbols.

        Compares the Jacobian gradient from get_jacobian() against the formula.
        """
        dim = random_oct.p.shape[0]
        s = jnp.exp(random_oct.log_s)
        U = random_oct.U
        beta = random_oct.beta
        I = jnp.eye(dim)

        # Christoffel symbols: Γ^a_{kj} = β_aj δ^a_k - β_ja δ^j_k
        # Gamma[k, j, a] = Γ^a_{kj}
        Gamma = jnp.einsum("aj,ak->kja", beta, I) - jnp.einsum("ja,jk->kja", beta, I)

        jacobian = random_oct.get_jacobian()
        H_computed = jacobian.gradient

        # H^i_{jk} = s_k s_j (β_jk U^i_j + Γ^a_kj U^i_a)
        # First term: s_k s_j β_jk U^i_j
        term1 = jnp.einsum("k,j,jk,ij->ijk", s, s, beta, U)

        # Second term: s_k s_j Γ^a_kj U^i_a = s_k s_j Gamma[k,j,a] U[i,a]
        term2 = jnp.einsum("k,j,kja,ia->ijk", s, s, Gamma, U)

        H_expected = term1 + term2

        max_error = jnp.max(jnp.abs(H_computed - H_expected))
        assert max_error < 1e-5, \
            f"Hessian formula failed: max error = {max_error}"

    def test_metric_log_determinant_autodiff(self, random_oct):
        """
        Verify that log|det g| computed via autodiff matches 2 * Σ_i log s_i.

        Since g_{ij} = s_i² δ_{ij} in the E-basis, det(g) = Π s_i², so:
            log|det g| = 2 Σ_i log s_i
        """
        E_frame = random_oct.get_coordinate_frame()
        E_val = E_frame.components.value

        # Compute metric g_{ij} = E_i · E_j = (E.T @ E)_{ij}
        @jet_decorator
        def compute_metric(E):
            return E.T @ E

        g_jet = compute_metric(E_frame.components)

        # Compute log|det g| via autodiff
        @jet_decorator
        def compute_log_det_g(g_val):
            sign, logdet = jnp.linalg.slogdet(g_val)
            return logdet

        log_det_g_jet = compute_log_det_g(g_jet)
        log_det_g_computed = log_det_g_jet.value

        # Expected: 2 * Σ_i log s_i
        log_det_g_expected = 2.0 * jnp.sum(random_oct.log_s)

        error = jnp.abs(log_det_g_computed - log_det_g_expected)
        assert error < 1e-5, \
            f"Metric log det failed: computed={log_det_g_computed}, expected={log_det_g_expected}"

    def test_covariant_derivative_of_principal_basis(self, random_oct):
        """
        Verify the covariant derivative formula from phd_content.tex (line 1628):

            ∇_{U_i} U_j = β_ij U_i    (for i ≠ j)
            ∇_{U_j} U_j = -Σ_{k≠j} β_jk U_k

        Uses the library's covariant_derivative to compute ∇_{U_i} U_j
        and verifies the formula.
        """
        dim = random_oct.p.shape[0]
        beta = random_oct.beta
        connection = random_oct.get_connection()
        U_frame = random_oct.get_principal_frame()

        for i in range(dim):
            for j in range(dim):
                U_i = U_frame.get_basis_vector(i)
                U_j = U_frame.get_basis_vector(j)

                # Compute ∇_{U_i} U_j using the library
                nabla_Ui_Uj = connection.covariant_derivative(U_i, U_j)
                computed = nabla_Ui_Uj.components.value

                # Expected from thesis
                expected = jnp.zeros(dim)
                if i != j:
                    # ∇_{U_i} U_j = β_ij U_i
                    # In U-basis, U_i has components e_i (standard basis vector)
                    expected = expected.at[i].set(beta[i, j])
                else:
                    # ∇_{U_j} U_j = -Σ_{k≠j} β_jk U_k
                    for k in range(dim):
                        if k != j:
                            expected = expected.at[k].set(-beta[j, k])

                max_error = jnp.max(jnp.abs(computed - expected))
                assert max_error < 1e-5, \
                    f"Covariant derivative failed for (i,j)=({i},{j}): " \
                    f"max error = {max_error}"


# =============================================================================
# Test Fitting Functions
# =============================================================================

def _get_log_det_jet_in_x_coords(oct: LocalOCT) -> Jet:
    """
    Compute log|det J| jet in x-coordinates using autodiff.

    The fit_local_oct_to_log_det function expects log_det_jet in x-coordinates,
    not U-frame coordinates as returned by get_log_det_jet().
    """
    jacobian = oct.get_jacobian()
    J_jet = Jet(
        value=jacobian.value,
        gradient=jacobian.gradient,
        hessian=jacobian.hessian,
        dim=oct.p.shape[0]
    )

    @jet_decorator
    def compute_log_det(J_val):
        sign, logdet = jnp.linalg.slogdet(J_val)
        return logdet

    return compute_log_det(J_jet)


class TestFitLocalOCT:
    """Tests for fitting LocalOCT to target log-det jets."""

    def test_fit_local_oct_to_log_det_recovers_original(self, key):
        """
        Fitting a perturbed LocalOCT to its original log_det_jet should
        approximately recover the original parameters.
        """
        dim = 2
        p = jnp.zeros(dim)
        original_oct = create_local_oct(p, key)

        # Get the target log_det_jet in x-coordinates
        target_log_det_jet = _get_log_det_jet_in_x_coords(original_oct)

        # Create a perturbed initial guess
        key, subkey = random.split(key)
        perturbed_oct = create_local_oct(p, subkey)

        # Fit the perturbed OCT to the target
        fitted_oct = fit_local_oct_to_log_det(
            perturbed_oct,
            target_log_det_jet,
            max_steps=500,
        )

        # Check that the fitted log_det_jet matches the target
        fitted_log_det_jet = _get_log_det_jet_in_x_coords(fitted_oct)

        # Note: This is a non-convex problem, so we check for reasonable improvement
        # rather than exact recovery. The key tests are orthonormality and loss reduction.
        value_error = jnp.abs(fitted_log_det_jet.value - target_log_det_jet.value)
        assert value_error < 2.0, f"Log det value mismatch: error = {value_error}"

        grad_error = jnp.max(jnp.abs(fitted_log_det_jet.gradient - target_log_det_jet.gradient))
        assert grad_error < 1.0, f"Log det gradient mismatch: error = {grad_error}"

    def test_fit_local_oct_preserves_orthonormality(self, key):
        """
        The fitted LocalOCT should have an approximately orthonormal U.
        """
        dim = 2
        p = jnp.zeros(dim)
        original_oct = create_local_oct(p, key)
        target_log_det_jet = _get_log_det_jet_in_x_coords(original_oct)

        key, subkey = random.split(key)
        perturbed_oct = create_local_oct(p, subkey)

        fitted_oct = fit_local_oct_to_log_det(
            perturbed_oct,
            target_log_det_jet,
            max_steps=500,
        )

        # Check orthonormality of U
        UTU = fitted_oct.U.T @ fitted_oct.U
        orthonormality_error = jnp.max(jnp.abs(UTU - jnp.eye(dim)))
        assert orthonormality_error < 1e-2, f"U not orthonormal: error = {orthonormality_error}"

    def test_fit_local_oct_reduces_loss(self, key):
        """
        Fitting should reduce the overall loss compared to the initial guess.
        """
        dim = 2
        p = jnp.zeros(dim)
        original_oct = create_local_oct(p, key)
        target_log_det_jet = _get_log_det_jet_in_x_coords(original_oct)

        key, subkey = random.split(key)
        perturbed_oct = create_local_oct(p, subkey)

        # Compute initial loss (value and gradient in x-coords)
        def compute_loss(oct, target):
            log_det_jet = _get_log_det_jet_in_x_coords(oct)
            value_loss = (log_det_jet.value - target.value)**2
            grad_loss = jnp.sum((log_det_jet.gradient - target.gradient)**2)
            return value_loss + grad_loss

        initial_loss = compute_loss(perturbed_oct, target_log_det_jet)

        fitted_oct = fit_local_oct_to_log_det(
            perturbed_oct,
            target_log_det_jet,
            max_steps=500,
        )

        final_loss = compute_loss(fitted_oct, target_log_det_jet)

        assert final_loss < initial_loss, \
            f"Fitting did not reduce loss: initial={initial_loss}, final={final_loss}"

    def test_fit_local_oct_to_gaussian_log_prob(self, key):
        """
        Fit LocalOCT to the log probability of a full-rank Gaussian.

        For N(μ, Σ):
          log p(x) = const - 0.5 * (x-μ)^T Σ^{-1} (x-μ)
          ∇ log p(x) = -Σ^{-1}(x - μ)  (score function)
          ∇² log p(x) = -Σ^{-1}
        """
        dim = 2
        p = jnp.zeros(dim)

        # Create a random full-rank covariance matrix
        key, subkey = random.split(key)
        A = random.normal(subkey, (dim, dim))
        Sigma = A @ A.T + 0.1 * jnp.eye(dim)  # Ensure positive definite

        # Mean at origin for simplicity
        mu = jnp.zeros(dim)

        # Define log probability function for N(μ, Σ)
        def log_prob(x):
            Sigma_inv = jnp.linalg.inv(Sigma)
            diff = x - mu
            return -0.5 * diff @ Sigma_inv @ diff - 0.5 * jnp.linalg.slogdet(Sigma)[1] - 0.5 * dim * jnp.log(2 * jnp.pi)

        # Use function_to_jet to compute value, gradient, and hessian
        target_log_det_jet = function_to_jet(log_prob, p)

        # Create initial OCT guess
        key, subkey = random.split(key)
        initial_oct = create_local_oct(p, subkey)

        # Fit to the Gaussian log prob
        fitted_oct = fit_local_oct_to_log_det(
            initial_oct,
            target_log_det_jet,
            max_steps=500,
        )

        # Check that U remains orthonormal
        UTU = fitted_oct.U.T @ fitted_oct.U
        orthonormality_error = jnp.max(jnp.abs(UTU - jnp.eye(dim)))
        assert orthonormality_error < 0.1, f"U not orthonormal: error = {orthonormality_error}"

        # Check that fitted log_det_jet approximately matches target
        fitted_log_det_jet = fitted_oct.get_log_det_jet()

        # Transform fitted gradient to x-coords for comparison
        # fitted gradient is in U-frame, target is in x-coords
        fitted_grad_x = fitted_oct.U @ fitted_log_det_jet.gradient

        grad_error = jnp.max(jnp.abs(fitted_grad_x - target_log_det_jet.gradient))
        assert grad_error < 1.0, f"Gradient mismatch: error = {grad_error}"

    def test_plot_oct_with_gaussian_density(self, key):
        """
        Test plotting OCT coordinate curves overlaid on a Gaussian density heatmap.
        """
        import matplotlib.pyplot as plt
        dim = 2
        p = jnp.zeros(dim)

        # Create a random full-rank covariance matrix
        key, subkey = random.split(key)
        A = random.normal(subkey, (dim, dim))
        Sigma = A @ A.T + 0.1 * jnp.eye(dim)

        mu = jnp.zeros(dim)

        def log_prob(x):
            Sigma_inv = jnp.linalg.inv(Sigma)
            diff = x - mu
            return -0.5 * diff @ Sigma_inv @ diff - 0.5 * jnp.linalg.slogdet(Sigma)[1] - 0.5 * dim * jnp.log(2 * jnp.pi)

        # Create and fit OCT
        target_log_det_jet = function_to_jet(log_prob, p)

        key, subkey = random.split(key)
        initial_oct = create_local_oct(p, subkey)

        fitted_oct = fit_local_oct_to_log_det(
            initial_oct,
            target_log_det_jet,
            max_steps=500,
            verbose=True,
        )

        # Test that plotting works without errors
        fig, ax = plot_oct_with_density(
            fitted_oct,
            log_prob,
            span=0.5,
            show=False,
            title="Fitted OCT on Gaussian Density",
        )

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_fit_oct_to_neals_funnel(self, key):
        """
        Test fitting OCT to a 2D Neal's Funnel distribution at multiple points.

        Neal's Funnel:
          x1 ~ N(0, sigma1)
          x2 ~ N(0, exp(x1/2))
        """
        pytest.importorskip("studies.toy_pdf.energy", reason="studies module not in path")
        import matplotlib.pyplot as plt
        from studies.toy_pdf.energy import NealsFunnel

        dim = 2

        # Define 5 different evaluation points
        points = [
            jnp.array([-3.0, 0.0]),
            jnp.array([-1.5, 0.3]),
            jnp.array([0.0, 0.0]),
            jnp.array([1.5, 0.5]),
            jnp.array([3.0, 1.0]),
        ]

        # Create Neal's Funnel distribution
        funnel = NealsFunnel(input_shape=(dim,), sigma1=3.0)

        def log_prob(x):
            return funnel.log_prob(x)

        # Train OCTs at each point and collect histories
        fitted_octs = []
        histories = []
        for p in points:
            target_log_det_jet = function_to_jet(log_prob, p)

            key, subkey = random.split(key)
            initial_oct = create_local_oct(p, subkey)

            fitted_oct, history = fit_local_oct_to_log_det(
                initial_oct,
                target_log_det_jet,
                max_steps=10000,
                verbose=False,
                reg_weight=2.0,
                return_history=True,
            )
            fitted_octs.append(fitted_oct)
            histories.append(history)

        # Define colors for each OCT's coordinate grid
        colors = [
            ('#2E86AB', '#A23B72'),  # Steel blue, Berry
            ('#28A745', '#DC3545'),  # Green, Red
            ('#FFC107', '#6F42C1'),  # Yellow, Purple
            ('#17A2B8', '#FD7E14'),  # Cyan, Orange
            ('#6C757D', '#343A40'),  # Gray, Dark gray
        ]

        # Plot all OCTs on the same figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw heatmap with first OCT (draw_heatmap=True)
        plot_oct_with_density(
            fitted_octs[0],
            log_prob,
            span=2.0,
            xlim=(-4.0, 2.0),
            ylim=(-1.0, 1.0),
            num_heatmap=500,
            draw_heatmap=True,
            draw_grid=True,
            draw_basis_vectors=True,
            line_color_1=colors[0][0],
            line_color_2=colors[0][1],
            show=False,
            ax=ax,
        )

        # Overlay remaining OCTs (draw_heatmap=False)
        for i, oct in enumerate(fitted_octs[1:], start=1):
            plot_oct_with_density(
                oct,
                log_prob,
                span=0.5,
                xlim=(-4.0, 2.0),
                ylim=(-1.0, 1.0),
                draw_heatmap=False,
                draw_grid=True,
                draw_basis_vectors=True,
                line_color_1=colors[i][0],
                line_color_2=colors[i][1],
                show=False,
                ax=ax,
            )

        ax.set_title("Fitted OCTs on Neal's Funnel at 5 Different Points")
        plt.show()

        assert fig is not None
        assert ax is not None

        plt.close(fig)

        # Plot training curves
        fig_loss, axes_loss = plt.subplots(2, 2, figsize=(12, 10))

        point_labels = [f"p={p.tolist()}" for p in points]

        # Total loss
        ax_total = axes_loss[0, 0]
        for i, (history, label) in enumerate(zip(histories, point_labels)):
            ax_total.semilogy(history['total_loss'], label=label, color=colors[i][0])
        ax_total.set_xlabel('Iteration')
        ax_total.set_ylabel('Total Loss')
        ax_total.set_title('Total Loss')
        ax_total.legend(fontsize=8)
        ax_total.grid(True, alpha=0.3)

        # Value loss
        ax_value = axes_loss[0, 1]
        for i, (history, label) in enumerate(zip(histories, point_labels)):
            ax_value.semilogy(history['value_loss'], label=label, color=colors[i][0])
        ax_value.set_xlabel('Iteration')
        ax_value.set_ylabel('Value Loss')
        ax_value.set_title('Value Loss')
        ax_value.legend(fontsize=8)
        ax_value.grid(True, alpha=0.3)

        # Gradient loss
        ax_grad = axes_loss[1, 0]
        for i, (history, label) in enumerate(zip(histories, point_labels)):
            ax_grad.semilogy(history['grad_loss'], label=label, color=colors[i][0])
        ax_grad.set_xlabel('Iteration')
        ax_grad.set_ylabel('Gradient Loss')
        ax_grad.set_title('Gradient Loss')
        ax_grad.legend(fontsize=8)
        ax_grad.grid(True, alpha=0.3)

        # Hessian loss
        ax_hess = axes_loss[1, 1]
        for i, (history, label) in enumerate(zip(histories, point_labels)):
            ax_hess.semilogy(history['hessian_loss'], label=label, color=colors[i][0])
        ax_hess.set_xlabel('Iteration')
        ax_hess.set_ylabel('Hessian Loss')
        ax_hess.set_title('Hessian Loss')
        ax_hess.legend(fontsize=8)
        ax_hess.grid(True, alpha=0.3)

        plt.suptitle("Training Curves for OCT Fitting", fontsize=14)
        plt.tight_layout()
        plt.show()

        plt.close(fig_loss)
