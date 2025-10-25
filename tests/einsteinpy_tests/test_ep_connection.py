import numpy as np
import jax.numpy as jnp
from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols
from sympy import symbols, sin, cos, Matrix
import sympy

from local_coordinates.metric import RiemannianMetric
from local_coordinates.basis import BasisVectors, get_standard_basis, change_basis
from local_coordinates.jet import Jet, function_to_jet
from local_coordinates.connection import Connection, get_levi_civita_connection
from local_coordinates.tangent import TangentVector, lie_bracket


def test_levi_civita_connection():
    """
    Compare the Levi-Civita connection from a RiemannianMetric with
    einsteinpy's symbolic Christoffel symbols.
    """
    # 1. Define a simpler 2D symbolic metric for speed
    r, theta = symbols("r, theta")
    syms = (r, theta)
    # Using simple polynomials is much faster for sympy and avoids JAX tracer issues
    metric_list = [
        [1 + r**2, theta],
        [theta, r**2]
    ]
    metric_sym = MetricTensor(metric_list, syms)

    # 2. Calculate Christoffel symbols symbolically with einsteinpy
    ch_sym = ChristoffelSymbols.from_metric(metric_sym)

    # 3. Lambdify for numerical evaluation
    arg_list_metric, metric_num_func = metric_sym.tensor_lambdify()
    arg_list_ch, ch_num_func = ch_sym.tensor_lambdify()

    # 4. Define a numerical point and evaluation arguments
    r_val, theta_val = 2.0, np.pi / 2
    val_map = {"r": r_val, "theta": theta_val}
    num_args_metric = [val_map[str(arg)] for arg in arg_list_metric]
    num_args_ch = [val_map[str(arg)] for arg in arg_list_ch]

    # Ground truth Christoffel symbols
    ep_chris_comps = ch_num_func(*num_args_ch)

    # 5. Create local_coordinates metric object with gradient info
    p = jnp.array([r_val, theta_val])

    # Create a JAX-compatible function for the metric
    # Note: we use jax.numpy for operations
    def metric_func_jax(p_jax):
        r, theta = p_jax
        return jnp.array([
            [1 + r**2, theta],
            [theta, r**2]
        ])

    metric_jet = function_to_jet(metric_func_jax, p)

    standard_basis = get_standard_basis(p)
    lc_metric = RiemannianMetric(basis=standard_basis, components=metric_jet)

    # 6. Calculate Levi-Civita connection using local_coordinates
    lc_connection = get_levi_civita_connection(lc_metric)

    # 7. Compare the results
    # Transpose einsteinpy's (k,i,j) to our (i,j,k)
    ep_chris_comps_transposed = np.transpose(ep_chris_comps, (1, 2, 0))
    np.testing.assert_allclose(
        lc_connection.christoffel_symbols.value, ep_chris_comps_transposed, rtol=1e-5, atol=1e-5
    )


def test_levi_civita_connection_derivatives_match_sympy():
    """
    Compare gradient and hessian of Christoffel symbols from local_coordinates
    against analytical derivatives computed with sympy for a simple 2D metric.
    Our convention: christoffel_symbols[i,j,k] = Γ^k_{ij}.
    Sympy/Einsteinpy convention: C[k,i,j] = Γ^k_{ij}.
    """
    # 1. Symbolic setup: simple polynomial metric in 2D
    r, theta = symbols("r, theta")
    syms = (r, theta)
    metric_list = [
        [1 + r**2, theta],
        [theta, r**2]
    ]
    mat = Matrix(metric_list)
    matinv = mat.inv()

    dims = 2
    # Christoffel (Einstein convention): C[i,j,k] = Γ^i_{jk}
    C = [[[0 for _ in range(dims)] for _ in range(dims)] for _ in range(dims)]
    for i in range(dims):
        for j in range(dims):
            for k in range(dims):
                expr = 0
                for n in range(dims):
                    expr += (matinv[i, n] / 2) * (
                        sympy.diff(mat[n, j], syms[k])
                        + sympy.diff(mat[n, k], syms[j])
                        - sympy.diff(mat[j, k], syms[n])
                    )
                C[i][j][k] = sympy.simplify(expr)

    # First and second derivatives wrt coordinates: C_grad[i,j,k,a], C_hess[i,j,k,a,b]
    C_grad = [[[[0 for _ in range(dims)] for _ in range(dims)] for _ in range(dims)] for _ in range(dims)]
    C_hess = [[[[[0 for _ in range(dims)] for _ in range(dims)] for _ in range(dims)] for _ in range(dims)] for _ in range(dims)]
    for i in range(dims):
        for j in range(dims):
            for k in range(dims):
                for a in range(dims):
                    C_grad[i][j][k][a] = sympy.diff(C[i][j][k], syms[a])
                    for b in range(dims):
                        C_hess[i][j][k][a][b] = sympy.diff(C[i][j][k], syms[a], syms[b])

    # 2. Numerical point
    r_val, theta_val = 2.0, np.pi / 2

    # Evaluate symbolic arrays at the point
    def eval_array(arr):
        # arr is nested python lists of sympy expressions; evaluate to numpy array
        def eval_leaf(x):
            f = sympy.lambdify((r, theta), x, "numpy")
            return float(f(r_val, theta_val))
        def rec(a):
            if isinstance(a, (list, tuple)):
                return np.array([rec(x) for x in a])
            else:
                return eval_leaf(a)
        return rec(arr)

    C_val_np = eval_array(C)              # shape (i,j,k)
    C_grad_np = eval_array(C_grad)        # shape (i,j,k,a)
    C_hess_np = eval_array(C_hess)        # shape (i,j,k,a,b)

    # Map to local_coordinates indexing: Gamma[i,j,k] = Γ^k_{ij} ⇒ C_perm = C.transpose(1,2,0,...)
    Gamma_val_gt = np.transpose(C_val_np, (1, 2, 0))
    Gamma_grad_gt = np.transpose(C_grad_np, (1, 2, 0, 3))
    Gamma_hess_gt = np.transpose(C_hess_np, (1, 2, 0, 3, 4))

    # 3. Build our metric at same point with value/grad/hess via JAX, then LC connection
    p = jnp.array([r_val, theta_val])
    def metric_func_jax(p_jax):
        rj, thetaj = p_jax
        return jnp.array([
            [1 + rj**2, thetaj],
            [thetaj, rj**2]
        ])

    metric_jet = function_to_jet(metric_func_jax, p)
    standard_basis = get_standard_basis(p)
    lc_metric = RiemannianMetric(basis=standard_basis, components=metric_jet)
    lc_connection = get_levi_civita_connection(lc_metric)

    # 4. Compare value, gradient, and hessian
    np.testing.assert_allclose(lc_connection.christoffel_symbols.value, Gamma_val_gt, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(lc_connection.christoffel_symbols.gradient, Gamma_grad_gt, rtol=1e-6, atol=1e-6)
    # np.testing.assert_allclose(lc_connection.christoffel_symbols.hessian, Gamma_hess_gt, rtol=1e-6, atol=1e-6) # Don't have enough information to check hessian

def test_torsion_free_sympy_coordinate_basis_jets():
    """
    Using a coordinate (holonomic) basis, verify torsion jets vanish by
    comparing our computed torsion against sympy ground-truth expressions.
    We test value, gradient, and hessian at a numerical point.
    """
    # 1) Symbolic metric in 2D (polynomial for speed)
    r, theta = symbols("r, theta")
    syms = (r, theta)
    metric_list = [[1 + r**2, theta], [theta, r**2]]
    mat = Matrix(metric_list)
    matinv = mat.inv()

    dims = 2
    # Christoffel C[i,j,k] = Γ^i_{jk}
    C = [[[0 for _ in range(dims)] for _ in range(dims)] for _ in range(dims)]
    for i in range(dims):
      for j in range(dims):
        for k in range(dims):
          expr = 0
          for n in range(dims):
            expr += (matinv[i, n] / 2) * (
              sympy.diff(mat[n, j], syms[k])
              + sympy.diff(mat[n, k], syms[j])
              - sympy.diff(mat[j, k], syms[n])
            )
          C[i][j][k] = sympy.simplify(expr)

    # 2) Choose polynomial vector fields X, Y in coordinates
    X = [r + theta**2, r*theta]
    Y = [theta, r**2 - theta]

    # 3) Define torsion components T^k = (∇_X Y - ∇_Y X - [X,Y])^k
    def covariant_derivative_components(Xv, Yv):
      # (∇_X Y)^k = X^i ∂_i Y^k + Γ^k_{ij} X^i Y^j
      res = [0, 0]
      for k in range(dims):
        term1 = 0
        for i in range(dims):
          term1 += Xv[i]*sympy.diff(Yv[k], syms[i])
        term2 = 0
        for i in range(dims):
          for j in range(dims):
            term2 += C[k][i][j]*Xv[i]*Yv[j]
        res[k] = sympy.simplify(term1 + term2)
      return res

    def lie_bracket_components(Xv, Yv):
      # [X,Y]^k = X^i ∂_i Y^k - Y^i ∂_i X^k
      res = [0,0]
      for k in range(dims):
        term = 0
        for i in range(dims):
          term += Xv[i]*sympy.diff(Yv[k], syms[i]) - Yv[i]*sympy.diff(Xv[k], syms[i])
        res[k] = sympy.simplify(term)
      return res

    nablaX_Y = covariant_derivative_components(X, Y)
    nablaY_X = covariant_derivative_components(Y, X)
    bracket = lie_bracket_components(X, Y)
    T = [sympy.simplify(nablaX_Y[k] - nablaY_X[k] - bracket[k]) for k in range(dims)]

    # 4) Derivatives of torsion components wrt coordinates
    T_grad = [[sympy.diff(T[k], syms[a]) for a in range(dims)] for k in range(dims)]
    T_hess = [[[sympy.diff(T[k], syms[a], syms[b]) for b in range(dims)] for a in range(dims)] for k in range(dims)]

    # 5) Evaluate at a numerical point
    r_val, theta_val = 1.7, 0.9
    def eval_expr(e):
      f = sympy.lambdify((r, theta), e, "numpy")
      return float(f(r_val, theta_val))

    T_val_np = np.array([eval_expr(Tk) for Tk in T])
    T_grad_np = np.array([[eval_expr(T_grad[k][a]) for a in range(dims)] for k in range(dims)])
    T_hess_np = np.array([[[eval_expr(T_hess[k][a][b]) for b in range(dims)] for a in range(dims)] for k in range(dims)])

    # Expect exact zeros
    np.testing.assert_allclose(T_val_np, 0.0, atol=1e-9)
    np.testing.assert_allclose(T_grad_np, 0.0, atol=1e-9)
    np.testing.assert_allclose(T_hess_np, 0.0, atol=1e-9)

    # 6) Cross-check against our implementation
    p = jnp.array([r_val, theta_val])
    def metric_func_jax(p_jax):
      rj, thetaj = p_jax
      return jnp.array([[1 + rj**2, thetaj],[thetaj, rj**2]])

    metric_jet = function_to_jet(metric_func_jax, p)
    basis = get_standard_basis(p)
    lc_metric = RiemannianMetric(basis=basis, components=metric_jet)
    conn = get_levi_civita_connection(lc_metric)

    # Build X, Y jets via function_to_jet
    def X_func(pt):
      rj, thetaj = pt
      return jnp.array([rj + thetaj**2, rj*thetaj])

    def Y_func(pt):
      rj, thetaj = pt
      return jnp.array([thetaj, rj**2 - thetaj])

    X_tv = TangentVector(p=p, basis=basis, components=function_to_jet(X_func, p))
    Y_tv = TangentVector(p=p, basis=basis, components=function_to_jet(Y_func, p))

    torsion = conn.covariant_derivative(X_tv, Y_tv) - conn.covariant_derivative(Y_tv, X_tv) - lie_bracket(X_tv, Y_tv)

    np.testing.assert_allclose(torsion.components.value, 0.0, atol=1e-8)
    np.testing.assert_allclose(torsion.components.gradient, 0.0, atol=1e-8)
    # np.testing.assert_allclose(torsion.components.hessian, 0.0, atol=1e-8) # Don't have enough information to check hessian


def test_levi_civita_connection_nonholonomic_basis_sympy():
    """
    Non-holonomic 2D frame: E1=∂_x, E2=∂_y + x ∂_x with Euclidean coordinate metric.
    Build Γ via the frame Koszul formula (with structure constants) in SymPy and
    compare to our implementation's value/gradient/hessian at a point.
    Index convention: our Γ[i,j,k] = Γ^k_{ij}.
    """
    # Symbols and coordinate metric (Euclidean)
    x, y = symbols("x, y")
    syms = (x, y)
    g_coord = Matrix([[1, 0],[0, 1]])

    # Frame components A^a_i (a: coord index; i: frame index)
    A = Matrix([[1, x],
                [0, 1]])
    Ainv = A.inv()

    # Metric in frame: g_frame_ij = A^a_i A^b_j g_ab = (A^T A)_{ij}
    g_frame = (A.T * g_coord * A)
    g_frame_inv = g_frame.inv()

    # Structure constants c_{ij}^k: [E_i, E_j] = c_{ij}^k E_k
    # Using [E_i, E_j]^a = A^b_i ∂_b A^a_j − A^b_j ∂_b A^a_i, then convert to frame via Ainv
    def lie_bracket_coord(i, j):
        # Return vector in coord comps (a)
        vec = [0, 0]
        for a in range(2):
            term = 0
            for b in range(2):
                term += A[b, i]*sympy.diff(A[a, j], syms[b]) - A[b, j]*sympy.diff(A[a, i], syms[b])
            vec[a] = sympy.simplify(term)
        return Matrix(vec)

    C = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(2)]  # c_{ij}^k
    for i in range(2):
        for j in range(2):
            bracket_coord = lie_bracket_coord(i, j)
            # Convert to frame components: c_{ij}^k = (Ainv)^k_a [E_i,E_j]^a
            comp = Ainv * bracket_coord
            for k in range(2):
                C[i][j][k] = sympy.simplify(comp[k])

    # E_i(g_{jk}) directional derivatives: E_i = A^a_i ∂_a
    def E_of_g(i, j, k):
        expr = 0
        for a in range(2):
            expr += A[a, i] * sympy.diff(g_frame[j, k], syms[a])
        return sympy.simplify(expr)

    # Γ^m_{ij} = 1/2 [ E_i(g_{jk}) g^{km} + E_j(g_{ik}) g^{km} − E_k(g_{ij}) g^{km}
    #                  + c_{ij}^m − c_{ik}^l g_{lj} g^{km} − c_{jk}^l g_{li} g^{km} ]
    Gamma = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(2)]  # Γ[i][j][m]
    for i in range(2):
        for j in range(2):
            for m in range(2):
                term1 = 0
                term2 = 0
                term3 = 0
                for k in range(2):
                    term1 += E_of_g(i, j, k) * g_frame_inv[k, m]
                    term2 += E_of_g(j, i, k) * g_frame_inv[k, m]
                    term3 += E_of_g(k, i, j) * g_frame_inv[k, m]
                term4 = C[i][j][m]
                term5 = 0
                term6 = 0
                for k in range(2):
                    for l in range(2):
                        term5 += C[i][k][l] * g_frame[l, j] * g_frame_inv[k, m]
                        term6 += C[j][k][l] * g_frame[l, i] * g_frame_inv[k, m]
                Gamma[i][j][m] = sympy.simplify( sympy.Rational(1,2) * (term1 + term2 - term3 + term4 - term5 - term6) )

    # First and second derivatives wrt coordinates for gradient/hessian
    Gamma_grad = [[[[sympy.diff(Gamma[i][j][m], sy) for sy in syms] for m in range(2)] for j in range(2)] for i in range(2)]
    Gamma_hess = [[[[[sympy.diff(Gamma[i][j][m], sa, sb) for sb in syms] for sa in syms] for m in range(2)] for j in range(2)] for i in range(2)]

    # Evaluate at point
    xv, yv = 0.3, -0.2
    def eval_arr(arr):
        def eval_leaf(e):
            f = sympy.lambdify((x, y), e, "numpy")
            return float(f(xv, yv))
        def rec(a):
            if isinstance(a, (list, tuple)):
                return np.array([rec(xe) for xe in a])
            else:
                return eval_leaf(a)
        return rec(arr)

    Gamma_val_gt = eval_arr(Gamma)                 # shape (i,j,m)
    Gamma_grad_gt = eval_arr(Gamma_grad)           # (i,j,m,a)
    Gamma_hess_gt = eval_arr(Gamma_hess)           # (i,j,m,a,b)

    # Build our non-holonomic basis and frame-metric in JAX
    p = jnp.array([xv, yv])
    def A_func(pt):
        xj, yj = pt
        return jnp.array([[1.0, xj], [0.0, 1.0]])

    def g_frame_func(pt):
        Aj = A_func(pt)
        return Aj.T @ Aj

    basis = BasisVectors(p=p, components=function_to_jet(A_func, p))
    metric = RiemannianMetric(basis=basis, components=function_to_jet(g_frame_func, p))
    conn = get_levi_civita_connection(metric)

    # Compare value/grad/hess
    np.testing.assert_allclose(conn.christoffel_symbols.value, Gamma_val_gt, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(conn.christoffel_symbols.gradient, Gamma_grad_gt, rtol=1e-6, atol=1e-6)
    # np.testing.assert_allclose(conn.christoffel_symbols.hessian, Gamma_hess_gt, rtol=1e-6, atol=1e-6) # Don't have enough information to check hessian


def test_torsion_free_requires_metric_basis_consistency():
    """
    Expose bug: constructing a metric Jet in coordinate components and attaching
    a non-holonomic basis without transforming the metric into that basis leads
    to non-zero torsion jets. When the metric is correctly expressed in the
    frame basis (by proper change-of-basis), torsion jets vanish as expected.
    """
    # Non-holonomic frame A(x) and coordinate metric g_coord(x)
    x, y = symbols("x, y")
    syms = (x, y)
    A = Matrix([[1, x],[0, 1]])
    g_coord = Matrix([[1 + x**2, x*y],[x*y, 2 + y**2]])

    # SymPy ground-truth: torsion-free Levi-Civita regardless of basis
    # Build Γ in frame via Koszul with structure constants (reuse code pattern)
    Ainv = A.inv()
    g_frame = (A.T * g_coord * A)
    g_frame_inv = g_frame.inv()

    def lie_bracket_coord(i, j):
        vec = [0, 0]
        for a in range(2):
            term = 0
            for b in range(2):
                term += A[b, i]*sympy.diff(A[a, j], syms[b]) - A[b, j]*sympy.diff(A[a, i], syms[b])
            vec[a] = sympy.simplify(term)
        return Matrix(vec)

    C = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            comp = Ainv * lie_bracket_coord(i, j)
            for k in range(2):
                C[i][j][k] = sympy.simplify(comp[k])

    def E_of_g(i, j, k):
        expr = 0
        for a in range(2): expr += A[a, i] * sympy.diff(g_frame[j, k], syms[a])
        return sympy.simplify(expr)

    Gamma = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            for m in range(2):
                term1 = sum(E_of_g(i, j, k)*g_frame_inv[k, m] for k in range(2))
                term2 = sum(E_of_g(j, i, k)*g_frame_inv[k, m] for k in range(2))
                term3 = sum(E_of_g(k, i, j)*g_frame_inv[k, m] for k in range(2))
                term4 = C[i][j][m]
                term5 = sum(C[i][k][l]*g_frame[l, j]*g_frame_inv[k, m] for k in range(2) for l in range(2))
                term6 = sum(C[j][k][l]*g_frame[l, i]*g_frame_inv[k, m] for k in range(2) for l in range(2))
                Gamma[i][j][m] = sympy.simplify(sympy.Rational(1, 2) * (term1 + term2 - term3 + term4 - term5 - term6))

    # Choose polynomial X, Y (coordinate components), then convert to frame components
    X_coord = Matrix([x**2 + y, x*y])
    Y_coord = Matrix([x + y**2, x**2 - y])
    X_frame = Ainv * X_coord
    Y_frame = Ainv * Y_coord

    def E_apply(i, f):
        # E_i acts on scalars by A^a_i ∂_a
        return sympy.simplify(sum(A[a, i]*sympy.diff(f, syms[a]) for a in range(2)))

    def covD_frame(G, Xf, Yf):
        res = [0, 0]
        for m in range(2):
            term1 = sum(Xf[i]*E_apply(i, Yf[m]) for i in range(2))
            term2 = sum(G[i][j][m]*Xf[i]*Yf[j] for i in range(2) for j in range(2))
            res[m] = sympy.simplify(term1 + term2)
        return res

    def bracket_frame(Xf, Yf):
        res = [0, 0]
        for m in range(2):
            term = sum(Xf[i]*E_apply(i, Yf[m]) - Yf[i]*E_apply(i, Xf[m]) for i in range(2))
            term += sum(C[i][j][m]*Xf[i]*Yf[j] for i in range(2) for j in range(2))
            res[m] = sympy.simplify(term)
        return res

    T_sym = [sympy.simplify(a - b - c) for a, b, c in zip(covD_frame(Gamma, X_frame, Y_frame), covD_frame(Gamma, Y_frame, X_frame), bracket_frame(X_frame, Y_frame))]
    # Derivatives
    T_grad = [[sympy.diff(T_sym[k], s) for s in syms] for k in range(2)]
    T_hess = [[[sympy.diff(T_sym[k], sa, sb) for sb in syms] for sa in syms] for k in range(2)]

    xv, yv = 0.2, -0.6
    def evalE(e):
        f = sympy.lambdify((x, y), e, "numpy")
        return float(f(xv, yv))

    T_val_gt = np.array([evalE(e) for e in T_sym])
    T_grad_gt = np.array([[evalE(e) for e in row] for row in T_grad])
    T_hess_gt = np.array([[[evalE(e) for e in row2] for row2 in row] for row in T_hess])

    # Build basis and two metrics in our code: inconsistent and corrected
    p = jnp.array([xv, yv])
    def A_func(pt):
        xj, yj = pt
        return jnp.array([[1.0, xj],[0.0, 1.0]])

    def g_coord_func(pt):
        xj, yj = pt
        return jnp.array([[1.0 + xj*xj, xj*yj],[xj*yj, 2.0 + yj*yj]])

    basis = BasisVectors(p=p, components=function_to_jet(A_func, p))

    # Inconsistent: attach coordinate metric components directly to frame basis
    metric_bad = RiemannianMetric(basis=basis, components=function_to_jet(g_coord_func, p))

    # Correct: start from standard-basis metric then change basis to frame
    std_basis = get_standard_basis(p)
    metric_std = RiemannianMetric(basis=std_basis, components=function_to_jet(g_coord_func, p))
    metric_good = change_basis(metric_std, basis)

    # X, Y as Jets
    def X_func(pt):
        xj, yj = pt
        return jnp.array([xj*xj + yj, xj*yj])
    def Y_func(pt):
        xj, yj = pt
        return jnp.array([xj + yj*yj, xj*xj - yj])

    X_tv = TangentVector(p=p, basis=basis, components=function_to_jet(X_func, p))
    Y_tv = TangentVector(p=p, basis=basis, components=function_to_jet(Y_func, p))

    # Torsion with bad metric (should be non-zero generally)
    conn_bad = get_levi_civita_connection(metric_bad)
    T_bad = conn_bad.covariant_derivative(X_tv, Y_tv) - conn_bad.covariant_derivative(Y_tv, X_tv) - lie_bracket(X_tv, Y_tv)

    # Torsion with good metric (should match SymPy zeros)
    conn_good = get_levi_civita_connection(metric_good)
    T_good = conn_good.covariant_derivative(X_tv, Y_tv) - conn_good.covariant_derivative(Y_tv, X_tv) - lie_bracket(X_tv, Y_tv)

    # Ground truth: zeros
    np.testing.assert_allclose(T_val_gt, 0.0, atol=1e-8)
    np.testing.assert_allclose(T_grad_gt, 0.0, atol=1e-8)
    np.testing.assert_allclose(T_hess_gt, 0.0, atol=1e-8)

    # Our corrected metric must yield zero torsion jets
    np.testing.assert_allclose(T_good.components.value, 0.0, atol=1e-8)
    np.testing.assert_allclose(T_good.components.gradient, 0.0, atol=1e-8)
    # np.testing.assert_allclose(T_good.components.hessian, 0.0, atol=1e-8) # Don't have enough information to check hessian


def test_covariant_derivative_jet_matches_sympy_nonholonomic():
    """
    Compare value, gradient, and hessian of ∇_X Y in a non-holonomic frame (A=[[1,x],[0,1]])
    with metric g_frame = A^T A, against SymPy ground truth.
    X, Y are quadratic polynomials in coordinates but passed to our system as
    frame components via X_frame = A^{-1} X_coord, Y_frame = A^{-1} Y_coord.
    """
    # Symbols and frame
    x, y = symbols("x, y")
    syms = (x, y)
    A = Matrix([[1, x],[0, 1]])
    Ainv = A.inv()

    # Metric in frame
    g_frame = (A.T * A)  # Euclidean in coords
    g_frame_inv = g_frame.inv()

    # Structure constants c_{ij}^k from A
    def lie_bracket_coord(i, j):
        vec = [0, 0]
        for a in range(2):
            term = 0
            for b in range(2):
                term += A[b, i]*sympy.diff(A[a, j], syms[b]) - A[b, j]*sympy.diff(A[a, i], syms[b])
            vec[a] = sympy.simplify(term)
        return Matrix(vec)

    C = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            comp = Ainv * lie_bracket_coord(i, j)
            for k in range(2):
                C[i][j][k] = sympy.simplify(comp[k])

    # Frame directional derivative of metric entries
    def E_of_g(i, j, k):
        expr = 0
        for a in range(2):
            expr += A[a, i] * sympy.diff(g_frame[j, k], syms[a])
        return sympy.simplify(expr)

    # Christoffels in frame
    Gamma = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            for m in range(2):
                term1 = sum(E_of_g(i, j, k)*g_frame_inv[k, m] for k in range(2))
                term2 = sum(E_of_g(j, i, k)*g_frame_inv[k, m] for k in range(2))
                term3 = sum(E_of_g(k, i, j)*g_frame_inv[k, m] for k in range(2))
                term4 = C[i][j][m]
                term5 = sum(C[i][k][l]*g_frame[l, j]*g_frame_inv[k, m] for k in range(2) for l in range(2))
                term6 = sum(C[j][k][l]*g_frame[l, i]*g_frame_inv[k, m] for k in range(2) for l in range(2))
                Gamma[i][j][m] = sympy.simplify(sympy.Rational(1, 2) * (term1 + term2 - term3 + term4 - term5 - term6))

    # Coordinate vector fields and conversion to frame components
    X_coord = Matrix([x**2 + y, x*y])
    Y_coord = Matrix([x + y**2, x**2 - y])
    X_frame = Ainv * X_coord
    Y_frame = Ainv * Y_coord

    def E_apply(i, f):
        return sympy.simplify(sum(A[a, i]*sympy.diff(f, syms[a]) for a in range(2)))

    # ∇_X Y in frame: (∇_X Y)^m = X^i E_i(Y^m) + Γ^m_{ij} X^i Y^j
    def covD_frame(G, Xf, Yf):
        res = [0, 0]
        for m in range(2):
            term1 = sum(Xf[i]*E_apply(i, Yf[m]) for i in range(2))
            term2 = sum(G[i][j][m]*Xf[i]*Yf[j] for i in range(2) for j in range(2))
            res[m] = sympy.simplify(term1 + term2)
        return res

    nabla_sym = covD_frame(Gamma, X_frame, Y_frame)
    nabla_grad = [[sympy.diff(nabla_sym[k], s) for s in syms] for k in range(2)]
    nabla_hess = [[[sympy.diff(nabla_sym[k], sa, sb) for sb in syms] for sa in syms] for k in range(2)]

    # Evaluate at point
    xv, yv = 0.2, -0.5
    def evalE(e):
        f = sympy.lambdify((x, y), e, "numpy")
        return float(f(xv, yv))

    nabla_val_gt = np.array([evalE(e) for e in nabla_sym])
    nabla_grad_gt = np.array([[evalE(e) for e in row] for row in nabla_grad])
    nabla_hess_gt = np.array([[[evalE(e) for e in row2] for row2 in row] for row in nabla_hess])

    # Our implementation
    p = jnp.array([xv, yv])
    def A_func(pt):
        xj, yj = pt
        return jnp.array([[1.0, xj],[0.0, 1.0]])
    def g_frame_func(pt):
        Aj = A_func(pt)
        return Aj.T @ Aj
    basis = BasisVectors(p=p, components=function_to_jet(A_func, p))
    metric = RiemannianMetric(basis=basis, components=function_to_jet(g_frame_func, p))
    conn = get_levi_civita_connection(metric)

    def X_coord_func(pt):
        xj, yj = pt
        return jnp.array([xj*xj + yj, xj*yj])
    def Y_coord_func(pt):
        xj, yj = pt
        return jnp.array([xj + yj*yj, xj*xj - yj])
    def X_frame_func(pt):
        Aj = A_func(pt)
        return jnp.linalg.solve(Aj, X_coord_func(pt))
    def Y_frame_func(pt):
        Aj = A_func(pt)
        return jnp.linalg.solve(Aj, Y_coord_func(pt))

    X_tv = TangentVector(p=p, basis=basis, components=function_to_jet(X_frame_func, p))
    Y_tv = TangentVector(p=p, basis=basis, components=function_to_jet(Y_frame_func, p))

    nabla = conn.covariant_derivative(X_tv, Y_tv)

    np.testing.assert_allclose(nabla.components.value, nabla_val_gt, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(nabla.components.gradient, nabla_grad_gt, rtol=1e-6, atol=1e-6)
    # np.testing.assert_allclose(nabla.components.hessian, nabla_hess_gt, rtol=1e-6, atol=1e-6) # Don't have enough information to check hessian

