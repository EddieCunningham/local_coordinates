import jax
import jax.numpy as jnp

from local_coordinates.curved_flow import SecondOrderFlow, ThirdOrderFlow
from local_coordinates.metric import RiemannianMetric
from local_coordinates.connection import Connection, get_levi_civita_connection
from local_coordinates.riemann import RiemannCurvatureTensor, get_riemann_curvature_tensor
import matplotlib.pyplot as plt

def test_second_order_flow_initialization():
  key = jax.random.PRNGKey(0)
  N = 4

  k1, k2, k3 = jax.random.split(key, 3)
  J = jax.random.normal(k1, (N, N))
  H = jax.random.normal(k2, (N, N, N))
  T = jax.random.normal(k3, (N, N, N, N))

  qf = SecondOrderFlow(J=J, H=H)

  assert qf.J.shape == (N, N)
  assert qf.H.shape == (N, N, N)
  assert jnp.array_equal(qf.J, J)
  # Expect symmetrized versions
  expected_H = 0.5 * (H + jnp.swapaxes(H, -1, -2))
  assert jnp.allclose(qf.H, expected_H)
  assert qf.batch_size is None

def test_second_order_flow_metric():
  key = jax.random.PRNGKey(0)
  N = 2

  k1, k2 = jax.random.split(key)
  J = jax.random.normal(k1, (N, N))
  H = jax.random.normal(k2, (N, N, N))

  qf = SecondOrderFlow(J=J, H=H)
  z = jax.random.normal(key, (N,))
  metric: RiemannianMetric = qf.get_metric(z)
  connection: Connection = get_levi_civita_connection(metric)
  curvature: RiemannCurvatureTensor = get_riemann_curvature_tensor(connection)

def test_third_order_flow_initialization():
  key = jax.random.PRNGKey(1)
  N = 3

  k1, k2, k3 = jax.random.split(key, 3)
  J = jax.random.normal(k1, (N, N))
  H = jax.random.normal(k2, (N, N, N))
  T = jax.random.normal(k3, (N, N, N, N))

  tf = ThirdOrderFlow(J=J, H=H, T=T)

  assert tf.J.shape == (N, N)
  assert tf.H.shape == (N, N, N)
  assert tf.T.shape == (N, N, N, N)
  assert jnp.array_equal(tf.J, J)

  expected_H = 0.5 * (H + jnp.swapaxes(H, -1, -2))
  assert jnp.allclose(tf.H, expected_H)

  perms = [(1,2,3),(1,3,2),(2,1,3),(2,3,1),(3,1,2),(3,2,1)]
  Ts = [jnp.transpose(T, (0,) + p) for p in perms]
  expected_T = sum(Ts) / len(Ts)
  assert jnp.allclose(tf.T, expected_T)
  assert tf.batch_size is None


def test_third_order_flow_metric():
  key = jax.random.PRNGKey(2)
  N = 2

  k1, k2, k3 = jax.random.split(key, 3)
  J = jax.random.normal(k1, (N, N))
  H = jax.random.normal(k2, (N, N, N))
  T = jax.random.normal(k3, (N, N, N, N))

  tf = ThirdOrderFlow(J=J, H=H, T=T)
  z = jax.random.normal(key, (N,))
  metric: RiemannianMetric = tf.get_metric(z)
  connection: Connection = get_levi_civita_connection(metric)
  curvature: RiemannCurvatureTensor = get_riemann_curvature_tensor(connection)


def test_second_order_flow_grid_eval():
  key = jax.random.PRNGKey(3)
  N = 2

  k1, k2 = jax.random.split(key)
  J = jax.random.normal(k1, (N, N))
  H = jax.random.normal(k2, (N, N, N))

  qf = SecondOrderFlow(J=J, H=H)

  n = 11
  xs = jnp.linspace(-1.0, 1.0, n)
  ys = jnp.linspace(-1.0, 1.0, n)
  X, Y = jnp.meshgrid(xs, ys, indexing="xy")
  points = jnp.stack([X.ravel(), Y.ravel()], axis=1)  # (n*n, 2)

  outputs = jax.vmap(lambda z: qf(z))(points)

  assert outputs.shape == (n * n, N)
  assert jnp.all(jnp.isfinite(outputs))


def test_third_order_flow_grid_eval():
  key = jax.random.PRNGKey(4)
  N = 2

  k1, k2, k3 = jax.random.split(key, 3)
  J = jax.random.normal(k1, (N, N))
  H = jax.random.normal(k2, (N, N, N))
  T = jax.random.normal(k3, (N, N, N, N))

  tf = ThirdOrderFlow(J=J, H=H, T=T)

  n = 11
  xs = jnp.linspace(-1.0, 1.0, n)
  ys = jnp.linspace(-1.0, 1.0, n)
  X, Y = jnp.meshgrid(xs, ys, indexing="xy")
  points = jnp.stack([X.ravel(), Y.ravel()], axis=1)  # (n*n, 2)

  outputs = jax.vmap(lambda z: tf(z))(points)

  assert outputs.shape == (n * n, N)
  assert jnp.all(jnp.isfinite(outputs))