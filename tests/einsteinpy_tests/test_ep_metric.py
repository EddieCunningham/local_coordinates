import numpy as np
import jax
import jax.numpy as jnp
from einsteinpy.symbolic import MetricTensor
from sympy import symbols, sin, cos
import sympy

from local_coordinates.metric import RiemannianMetric
from local_coordinates.basis import (
    BasisVectors,
    get_standard_basis,
    change_basis,
    get_basis_transform,
)
from local_coordinates.jet import Jet


def test_metric_comparison():
    """
    Compare an arbitrary symbolic metric from einsteinpy with our RiemannianMetric.
    """
    # 1. Define symbolic variables for coordinates
    t, r, theta, phi = symbols("t r theta phi")

    # 2. Define an arbitrary symbolic metric tensor
    # It must be symmetric.
    metric_list = [
        [1 + r**2, sin(t), r * cos(theta), t * sin(phi)],
        [sin(t), -(1 + theta**2), t * r, theta * phi],
        [r * cos(theta), t * r, -(r**2), cos(phi) * sin(t)],
        [t * sin(phi), theta * phi, cos(phi) * sin(t), -(r**2 * sin(theta)**2)]
    ]

    metric_sym = MetricTensor(
        metric_list, (t, r, theta, phi), name="ArbitraryMetric"
    )

    # 3. Lambdify the symbolic tensor to create a numerical function
    arg_list, metric_num_func = metric_sym.tensor_lambdify()

    # 4. Define a numerical point in spacetime to evaluate the metric at
    t_val = 1.0
    r_val = 10.0
    theta_val = np.pi / 4
    phi_val = np.pi / 2

    # 5. Map symbols to numerical values and get arguments in the correct order
    val_map = {
        "t": t_val,
        "r": r_val,
        "theta": theta_val,
        "phi": phi_val,
    }
    num_args = [val_map[str(arg)] for arg in arg_list]

    # 6. Calculate the ground truth metric components by evaluating the function
    ep_metric_components = metric_num_func(*num_args)

    # 7. Create our RiemannianMetric
    x_vec = np.array([t_val, r_val, theta_val, phi_val])
    x_4vec = jnp.array(x_vec)
    dim = 4
    basis = BasisVectors(
        p=x_4vec,
        components=Jet(
            value=jnp.eye(dim),
            gradient=jnp.zeros((dim, dim, dim)),
            hessian=jnp.zeros((dim, dim, dim, dim)),
        ),
    )

    # 8. The components are the metric tensor from the evaluated symbolic expression
    metric_jet = Jet(
        value=jnp.array(ep_metric_components), gradient=None, hessian=None, dim=dim
    )

    lc_metric = RiemannianMetric(basis=basis, components=metric_jet)

    # 9. Compare the components
    np.testing.assert_allclose(
        lc_metric.components.value, ep_metric_components, rtol=1e-5
    )


def test_change_basis_metric_einsteinpy():
    """
    Tests the change of basis for a RiemannianMetric against einsteinpy.
    """
    # 1. Define symbolic variables and arbitrary metric
    t, r, theta, phi = symbols("t r theta phi")
    metric_list = [
        [1, 0, 0, sin(t)],
        [0, -r**2, 0, 0],
        [0, 0, -sin(theta)**2, 0],
        [sin(t), 0, 0, -1],
    ]
    metric_sym = MetricTensor(
        metric_list, (t, r, theta, phi), name="ArbitraryMetric"
    )

    # 2. Define a random symbolic transformation matrix
    T_sym_list = [[cos(t), sin(t) * r, 0, 0], [-sin(t), cos(t), 0, 0], [0, 0, theta, phi], [0, 0, phi, theta]]
    T_sym = sympy.Array(T_sym_list)

    # 3. Transform the metric using einsteinpy
    # Note: einsteinpy's lorentz_transform is a general basis transform
    ep_metric_new = metric_sym.lorentz_transform(T_sym)

    # 4. Lambdify everything for numerical evaluation
    arg_list_orig, metric_num_func = metric_sym.tensor_lambdify()
    _, metric_new_num_func = ep_metric_new.tensor_lambdify()
    _, T_num_func = MetricTensor(T_sym, (t, r, theta, phi)).tensor_lambdify()


    # 5. Define numerical point and get numerical inputs
    t_val, r_val, theta_val, phi_val = 1.0, 10.0, np.pi / 4, np.pi / 2
    val_map = {"t": t_val, "r": r_val, "theta": theta_val, "phi": phi_val}
    num_args = [val_map[str(arg)] for arg in arg_list_orig]

    # 6. Evaluate to get numerical ground truth
    metric_std_num = metric_num_func(*num_args)
    ep_metric_new_num = metric_new_num_func(*num_args)
    T_num = T_num_func(*num_args)

    # 7. Perform the transformation with local_coordinates
    p = jnp.array([t_val, r_val, theta_val, phi_val])
    dim = 4

    std_basis = get_standard_basis(p)
    metric_jet_std = Jet(value=jnp.array(metric_std_num), gradient=None, hessian=None, dim=dim)
    lc_metric_std = RiemannianMetric(basis=std_basis, components=metric_jet_std)

    new_basis_jet = Jet(value=jnp.array(T_num), gradient=None, hessian=None, dim=dim)
    new_basis = BasisVectors(p=p, components=new_basis_jet)

    lc_metric_new = change_basis(lc_metric_std, new_basis)

    # 8. Compare the results
    # Note: einsteinpy's transform for a (0,2)-tensor is T.T @ g @ T
    # Our transform g' = (T_inv).T @ g @ (T_inv) assumes T transforms basis vectors.
    # To align them, we must provide get_basis_transform with the inverse of T.
    # It's easier to just compute the expected result manually from `einsteinpy`'s output.
    np.testing.assert_allclose(lc_metric_new.components.value, ep_metric_new_num, rtol=1e-5)


def test_change_coordinates_metric_einsteinpy():
    """
    Tests change of coordinates for a RiemannianMetric against a symbolic
    transformation calculated with einsteinpy and sympy.
    """
    # 1. Start with Minkowski metric in Cartesian coordinates
    from einsteinpy.symbolic.predefined import MinkowskiCartesian, MinkowskiPolar
    from sympy import Matrix

    syms_cart = symbols("t x y z")
    t, x, y, z = syms_cart
    c = symbols("c")
    g_cart_sym = MinkowskiCartesian(c=c)

    # 2. Define coordinate transformation: Cartesian -> Spherical
    syms_polar = symbols("t r theta phi")
    _t, r, theta, phi = syms_polar

    # Transformation equations (x^a -> x'^b)
    transform = Matrix([t, r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)])

    # 3. Calculate Jacobian J_ab = dx^a / dx'^b
    J = transform.jacobian(syms_polar)

    # 4. Symbolically transform the metric: g' = J.T * g * J
    g_cart_matrix = Matrix(g_cart_sym.tensor())
    g_polar_sym_tensor = J.T * g_cart_matrix * J
    g_polar_sym_simplified = sympy.simplify(g_polar_sym_tensor)

    # Sanity check: compare with predefined polar metric
    g_polar_predefined = Matrix(MinkowskiPolar(c=c).tensor())
    assert sympy.simplify(g_polar_sym_simplified - g_polar_predefined) == sympy.zeros(4)

    # 5. Lambdify for numerical evaluation
    arg_list_cart, g_cart_num_func = g_cart_sym.tensor_lambdify()
    arg_list_polar, g_polar_num_func = MetricTensor(g_polar_sym_simplified, syms_polar).tensor_lambdify()

    # 6. Define numerical point and evaluate
    x_val, y_val, z_val = 1.0, 1.0, 1.0
    r_val = np.sqrt(x_val**2 + y_val**2 + z_val**2)
    theta_val = np.arccos(z_val / r_val)
    phi_val = np.arctan2(y_val, x_val)

    p_cart = jnp.array([0.0, x_val, y_val, z_val])
    g_cart_num = g_cart_num_func(0.0, x_val, y_val, z_val, 1.0)
    g_polar_num_gt = g_polar_num_func(0.0, r_val, theta_val, phi_val, 1.0)

    # 7. Perform transformation with local_coordinates
    cartesian_basis = get_standard_basis(p_cart)
    metric_jet_cart = Jet(value=jnp.array(g_cart_num), gradient=None, hessian=None, dim=4)
    lc_metric_cart = RiemannianMetric(basis=cartesian_basis, components=metric_jet_cart)

    def cart_to_polar(xyz_vec):
        t, x, y, z = xyz_vec
        r = jnp.sqrt(x**2 + y**2 + z**2)
        theta = jnp.arccos(z / r)
        phi = jnp.arctan2(y, x)
        return jnp.array([t, r, theta, phi])

    def polar_to_cart(polar_vec):
        t, r, theta, phi = polar_vec
        x = r * jnp.sin(theta) * jnp.cos(phi)
        y = r * jnp.sin(theta) * jnp.sin(phi)
        z = r * jnp.cos(theta)
        return jnp.array([t, x, y, z])

    # For metric transformation via change_basis, we need to construct a basis
    # whose components matrix equals the Jacobian J = d(cart)/d(polar).
    # The change_basis function for (0,2) tensors computes g' = Tinv.T @ g @ Tinv
    # where Tinv = inv(get_basis_transform(old, new)) = new_basis_components.
    # For the correct metric transformation g' = J.T @ g @ J, we need Tinv = J.
    # Therefore, new_basis_components = J (the Jacobian of polar_to_cart).
    p_polar = cart_to_polar(p_cart)
    J_polar_to_cart = jax.jacrev(polar_to_cart)(p_polar)  # d(cart)/d(polar)
    polar_basis_for_metric = BasisVectors(
        p=p_cart,
        components=Jet(value=jnp.array(J_polar_to_cart), gradient=None, hessian=None, dim=4)
    )
    lc_metric_polar = change_basis(lc_metric_cart, polar_basis_for_metric)

    # 8. Compare
    np.testing.assert_allclose(lc_metric_polar.components.value, g_polar_num_gt, rtol=1e-5, atol=1e-5)
