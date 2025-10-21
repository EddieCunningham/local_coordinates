import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')  # ensure non-interactive backend for tests
import os
import pytest
import matplotlib.pyplot as plt

from local_coordinates.jet import Jet
from local_coordinates.basis import BasisVectors
from local_coordinates.plot_basis import plot_coordinate_grid


def test_plot_coordinate_grid_identity_basis(tmp_path):
  # Simple 2D identity frame at point p with zero second derivatives
  p = jnp.array([1.0, 2.0])
  frame = jnp.eye(2)
  hessian = jnp.zeros((2, 2, 2))
  components_jet = Jet(value=frame, gradient=hessian, hessian=None)
  basis = BasisVectors(p=p, components=components_jet)

  savepath = tmp_path / 'grid_identity.png'
  fig, ax = plot_coordinate_grid(basis, num=11, span=1.0, savepath=str(savepath), title='identity')

  assert os.path.exists(savepath)
  assert os.path.getsize(savepath) > 0
  plt.close(fig)


def test_plot_coordinate_grid_chart_based_basis(tmp_path):
  # Use a simple chart: polar to cartesian
  def chart(u):
    r, theta = u
    return jnp.array([r * jnp.cos(theta), r * jnp.sin(theta)])

  u = jnp.array([2.0, jnp.pi / 4])
  p = chart(u)
  frame = jax.jacfwd(chart)(u)  # shape (2,2)
  # Hessian H[k, j, i] = d^2 x^i / du^k du^j
  H_kji = jax.jacfwd(jax.jacrev(chart))(u)
  # Convert to convention (i, j, k) = d(E_j)^i / du^k
  hessian = jnp.transpose(H_kji, (2, 1, 0))

  components_jet = Jet(value=frame, gradient=hessian, hessian=None)
  basis = BasisVectors(p=p, components=components_jet)

  savepath = tmp_path / 'grid_polar.png'
  fig, ax = plot_coordinate_grid(basis, num=9, span=1.05, savepath=str(savepath), title='polar')

  assert os.path.exists(savepath)
  assert os.path.getsize(savepath) > 0
  plt.close(fig)


def _random_invertible_frame_2x2(key):
  A = jax.random.normal(key, (2, 2))
  # Make it more likely invertible by adding scaled identity
  A = A + 0.5 * jnp.eye(2)
  return A


def test_plot_coordinate_grid_random_basis(tmp_path):
  key = jax.random.PRNGKey(0)
  p = jax.random.normal(key, (2,))
  frame = _random_invertible_frame_2x2(jax.random.split(key)[0])
  # Non-symmetric random hessian (i, j, k)
  H = jax.random.normal(jax.random.split(key)[1], (2, 2, 2))

  components_jet = Jet(value=frame, gradient=H, hessian=None)
  basis = BasisVectors(p=p, components=components_jet)

  savepath = tmp_path / 'grid_random.png'
  fig, ax = plot_coordinate_grid(basis, num=7, span=1.0, savepath=str(savepath), title='random')

  assert os.path.exists(savepath)
  assert os.path.getsize(savepath) > 0
  plt.close(fig)

