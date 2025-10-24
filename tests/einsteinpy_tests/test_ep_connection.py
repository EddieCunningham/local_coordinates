import numpy as np
import jax.numpy as jnp
from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols
from sympy import symbols, sin, cos, Matrix
import sympy

from local_coordinates.metric import RiemannianMetric
from local_coordinates.basis import BasisVectors, get_standard_basis, change_basis
from local_coordinates.jet import Jet, function_to_jet
from local_coordinates.connection import Connection, get_levi_civita_connection


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
    np.testing.assert_allclose(
        lc_connection.christoffel_symbols.value, ep_chris_comps, rtol=1e-5, atol=1e-5
    )

def test_change_basis_connection():
    """
    Tests the change of basis for a Connection against einsteinpy,
    assuming a linear transformation where Christoffel symbols transform as tensors.
    """
    # 1. Setup symbolic metric and Christoffel symbols in standard basis
    r, theta = symbols("r, theta")
    syms = (r, theta)
    # Simple 2D polar metric for simplicity
    metric_list = [[1., 0.], [0., r**2]]
    metric_sym = MetricTensor(metric_list, syms)
    ch_sym = ChristoffelSymbols.from_metric(metric_sym)

    # 2. Define a constant (linear) transformation matrix and transform in einsteinpy
    T_sym = Matrix([[1, 2], [3, 4]])
    # For linear transforms, Christoffels transform as a (1,2) tensor.
    # einsteinpy's `lorentz_transform` does exactly this.
    ch_sym_new = ch_sym.lorentz_transform(T_sym)

    # 3. Lambdify for numerical evaluation
    arg_list_ch, ch_num_func = ch_sym.tensor_lambdify()
    arg_list_ch_new, ch_new_num_func = ch_sym_new.tensor_lambdify()

    # 4. Choose a point and get numerical ground truth values
    r_val, theta_val = 2.0, np.pi / 2
    val_map = {"r": r_val, "theta": theta_val}
    num_args_ch = [val_map[str(arg)] for arg in arg_list_ch]
    num_args_ch_new = [val_map[str(arg)] for arg in arg_list_ch_new]

    chris_comps_std = ch_num_func(*num_args_ch)
    chris_comps_new_gt = ch_new_num_func(*num_args_ch_new)

    # 5. Perform the transformation with local_coordinates
    p = jnp.array([r_val, theta_val])
    dim = 2

    # Connection in standard basis
    standard_basis = get_standard_basis(p)
    chris_jet_std = Jet(value=jnp.array(chris_comps_std), gradient=None, hessian=None, dim=dim)
    lc_connection_std = Connection(basis=standard_basis, christoffel_symbols=chris_jet_std)

    # New basis from transformation matrix. Basis vectors transform by T_inv.
    T_num = np.array(T_sym).astype(float)
    new_basis_comps = jnp.linalg.inv(T_num)
    new_basis_jet = Jet(value=new_basis_comps, gradient=jnp.zeros((dim, dim, dim)), hessian=None, dim=dim)
    new_basis = BasisVectors(p=p, components=new_basis_jet)

    # 6. Change basis in local_coordinates
    lc_connection_new = change_basis(lc_connection_std, new_basis)

    # 7. Compare
    np.testing.assert_allclose(
        lc_connection_new.christoffel_symbols.value, chris_comps_new_gt, rtol=1e-5, atol=1e-5
    )