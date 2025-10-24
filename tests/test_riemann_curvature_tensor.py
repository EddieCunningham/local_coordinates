import jax.numpy as jnp
from jax import random
import numpy as np

from local_coordinates.metric import RiemannianMetric
from local_coordinates.basis import get_standard_basis
from local_coordinates.jet import function_to_jet
from local_coordinates.connection import get_levi_civita_connection
from local_coordinates.riemann import get_riemann_curvature_tensor


def test_riemann_symmetries():
  """
  Tests that the calculated RiemannCurvatureTensor satisfies its
  fundamental symmetries for a randomly generated metric.
  """
  key = random.PRNGKey(42)
  dim = 3
  p = random.normal(key, (dim,))
  key, subkey = random.split(key)

  # 1. Create a JAX function that returns a random (but deterministic)
  #    symmetric matrix to represent the metric components.
  #    This allows `function_to_jet` to calculate derivatives.
  W = random.normal(subkey, (dim, dim, dim))

  def random_metric_func(point):
    # A simple non-trivial function of the coordinates
    g = jnp.einsum('ijk,j,k->i', W, point, point)
    # Make it a symmetric matrix
    g_matrix = jnp.outer(g, g) + jnp.eye(dim)  # Add identity to ensure invertibility
    return g_matrix

  # 2. Calculate the Riemann tensor for this metric
  metric_jet = function_to_jet(random_metric_func, p)
  standard_basis = get_standard_basis(p)
  metric = RiemannianMetric(basis=standard_basis, components=metric_jet)

  connection = get_levi_civita_connection(metric)
  riemann_tensor = get_riemann_curvature_tensor(connection)
  R = riemann_tensor.components.value  # Components R^i_{jkl}

  # 3. Check symmetries
  # a) Antisymmetry in last two indices: R^i_{jkl} = -R^i_{jlk}
  np.testing.assert_allclose(R, -R.transpose((0, 1, 3, 2)), atol=1e-5)

  # b) Interchange symmetry of pairs: R_{ijkl} = R_{klij}
  # First, lower the index of R^i_{jkl} to get R_{mjkl}
  g = metric.components.value
  R_cov = jnp.einsum("im,mjkl->ijkl", g, R)
  np.testing.assert_allclose(R_cov, R_cov.transpose((2, 3, 0, 1)), atol=1e-5)

  # c) First Bianchi identity: R^i_{jkl} + R^i_{klj} + R^i_{ljk} = 0
  # This involves cyclic permutation of the last three indices (j, k, l).
  # R(i,j,k,l) + R(i,k,l,j) + R(i,l,j,k)
  bianchi_sum = R + R.transpose((0, 2, 3, 1)) + R.transpose((0, 3, 1, 2))
  np.testing.assert_allclose(bianchi_sum, jnp.zeros_like(R), atol=1e-5)
