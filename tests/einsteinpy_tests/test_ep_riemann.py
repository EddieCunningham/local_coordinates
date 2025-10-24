import numpy as np
import jax.numpy as jnp
from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols, RiemannCurvatureTensor
from sympy import symbols, sin, cos
import sympy

from local_coordinates.metric import RiemannianMetric
from local_coordinates.basis import get_standard_basis
from local_coordinates.jet import function_to_jet
from local_coordinates.connection import get_levi_civita_connection
from local_coordinates.riemann import get_riemann_curvature_tensor


def test_riemann_curvature_tensor():
  """
  Compare the Riemann curvature tensor from a Connection with
  einsteinpy's symbolic RiemannCurvatureTensor.
  """
  # 1. Define a 2D symbolic metric
  r, theta = symbols("r, theta")
  syms = (r, theta)
  metric_list = [
    [1 + r**2, 0],
    [0, r**2]
  ]
  metric_sym = MetricTensor(metric_list, syms)

  # 2. Calculate Riemann tensor symbolically with einsteinpy
  rm_sym = RiemannCurvatureTensor.from_metric(metric_sym)

  # 3. Lambdify for numerical evaluation
  arg_list_rm, rm_num_func = rm_sym.tensor_lambdify()

  # 4. Define a numerical point and evaluation arguments
  r_val, theta_val = 2.0, np.pi / 2
  val_map = {"r": r_val, "theta": theta_val}
  num_args_rm = [val_map[str(arg)] for arg in arg_list_rm]

  # Ground truth Riemann tensor components
  ep_riemann_comps = rm_num_func(*num_args_rm)

  # 5. Create local_coordinates metric object with derivatives
  p = jnp.array([r_val, theta_val])

  def metric_func_jax(p_jax):
    r, theta = p_jax
    return jnp.array([
        [1 + r**2, 0.],
        [0., r**2]
    ])

  metric_jet = function_to_jet(metric_func_jax, p)
  standard_basis = get_standard_basis(p)
  lc_metric = RiemannianMetric(basis=standard_basis, components=metric_jet)

  # 6. Calculate connection and then Riemann tensor using local_coordinates
  lc_connection = get_levi_civita_connection(lc_metric)
  lc_riemann = get_riemann_curvature_tensor(lc_connection)

  # 7. Compare the results
  np.testing.assert_allclose(
    lc_riemann.components.value, ep_riemann_comps, rtol=1e-5, atol=1e-5
  )
