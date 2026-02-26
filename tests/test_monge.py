import jax
import jax.numpy as jnp
import jax.random as random
import pytest
from local_coordinates.monge import get_monge_metric, get_second_fundamental_form, get_shape_matrix
from local_coordinates.metric import RiemannianMetric
from local_coordinates.basis import get_standard_basis
from local_coordinates.jet import Jet
from local_coordinates.tangent import TangentVector
from local_coordinates.normal_coords import get_rnc_jacobians
from local_coordinates.exponential_map import exponential_map_ode
from local_disentanglement.energies.funnel import Funnel


def test_get_monge_metric_returns_riemannian_metric():
  """
  get_monge_metric returns a RiemannianMetric with standard basis and (n, n) components.
  """
  def f(x):
    return -0.5 * jnp.sum(x**2)

  x = jnp.array([1.0, -0.5])
  metric = get_monge_metric(f, x)

  assert isinstance(metric, RiemannianMetric)
  assert metric.basis.p.shape == x.shape
  assert metric.components.value.shape == (2, 2)


def test_get_monge_metric_at_stationary_point_is_identity():
  """
  When nabla f = 0, the Monge metric reduces to the identity.
  """
  def f(x):
    return -0.5 * jnp.sum(x**2)

  x = jnp.array([0.0, 0.0])
  metric = get_monge_metric(f, x)

  assert jnp.allclose(metric.components.value, jnp.eye(2))


def test_get_monge_metric_formula():
  """
  For f(x) = a^T x, nabla f = a and g = I + a a^T.
  """
  a = jnp.array([1.0, 2.0])

  def f(x):
    return jnp.dot(a, x)

  x = jnp.array([0.5, -0.3])
  metric = get_monge_metric(f, x)

  expected = jnp.eye(2) + jnp.outer(a, a)
  assert jnp.allclose(metric.components.value, expected)


def test_get_monge_metric_with_scaled_function():
  """
  For f(x) = alpha * a^T x, nabla f = alpha * a and g = I + alpha^2 a a^T.
  This verifies that scaling can be absorbed into f by the caller.
  """
  a = jnp.array([1.0, 2.0])
  alpha = 0.7

  def f(x):
    return alpha * jnp.dot(a, x)

  x = jnp.array([0.5, -0.3])
  metric = get_monge_metric(f, x)

  expected = jnp.eye(2) + alpha**2 * jnp.outer(a, a)
  assert jnp.allclose(metric.components.value, expected)


def test_get_monge_metric_determinant():
  """
  det(g) = 1 + ||nabla f||^2 for the rank-1 update g = I + nabla f nabla f^T.
  """
  a = jnp.array([1.0, 0.5])

  def f(x):
    return jnp.dot(a, x)

  x = jnp.array([1.0, 2.0])
  metric = get_monge_metric(f, x)

  expected_det = 1.0 + jnp.sum(a**2)
  actual_det = jnp.linalg.det(metric.components.value)

  assert jnp.allclose(actual_det, expected_det)


def test_get_monge_metric_inverse():
  """
  Sherman-Morrison inverse of g = I + nabla f nabla f^T gives
  g^{-1} = I - nabla f nabla f^T / (1 + ||nabla f||^2).
  """
  a = jnp.array([1.0, 2.0])

  def f(x):
    return jnp.dot(a, x)

  x = jnp.array([0.0, 0.0])
  metric = get_monge_metric(f, x)

  norm_sq = jnp.sum(a**2)
  expected_inv = jnp.eye(2) - jnp.outer(a, a) / (1.0 + norm_sq)
  actual_inv = jnp.linalg.inv(metric.components.value)

  assert jnp.allclose(actual_inv, expected_inv)


def test_get_monge_metric_constant_f_is_identity():
  """
  When f is constant, nabla f = 0 and the metric is the identity everywhere.
  """
  def f(x):
    return 0.0 * jnp.sum(x)  # constant (but traceable by JAX)

  x = jnp.array([1.0, 2.0])
  metric = get_monge_metric(f, x)

  assert jnp.allclose(metric.components.value, jnp.eye(2))


def test_get_monge_metric_symmetric_positive_definite():
  """
  Monge metric components are symmetric and positive definite.
  """
  def f(x):
    return -0.5 * jnp.sum((x - jnp.array([1.0, -1.0]))**2)

  x = jnp.array([0.5, 0.5])
  metric = get_monge_metric(f, x)

  G = metric.components.value
  assert jnp.allclose(G, G.T)
  evals = jnp.linalg.eigvalsh(G)
  assert jnp.all(evals > 0.1)


@pytest.mark.plot
def test_monge_metric_rnc_heatmap(plot_save_dir):
  """
  Plot 2D energy heatmap with Taylor-approximated RNC coordinate grids at 2-3 points.
  Skipped in default runs; use pytest -m plot or pytest -k monge_metric_rnc_heatmap to run.
  Without --plot-dir: calls plt.show(). With --plot-dir DIR: saves to DIR/monge_rnc_heatmap.png.
  """
  import matplotlib
  if plot_save_dir is not None:
    matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  alpha = 0.5
  def f(x):
    return alpha * (-0.5 * jnp.sum((x - jnp.array([0.0, 0.0]))**2))

  n = 80
  x0 = jnp.linspace(-2.0, 2.0, n)
  x1 = jnp.linspace(-2.0, 2.0, n)
  X0, X1 = jnp.meshgrid(x0, x1, indexing="ij")
  grid = jnp.stack([X0.ravel(), X1.ravel()], axis=1)
  E = jax.vmap(lambda z: -f(z) / alpha)(grid).reshape(n, n)

  fig, ax = plt.subplots(figsize=(8, 6))
  ax.pcolormesh(x0, x1, E.T, shading="auto", cmap="viridis")
  ax.set_xlim(-2, 2)
  ax.set_ylim(-2, 2)
  ax.set_aspect("equal")

  centers = [
    jnp.array([0.0, 0.0]),
    jnp.array([1.0, 0.5]),
    jnp.array([-0.5, -0.5]),
  ]
  v_vals = jnp.linspace(-0.35, 0.35, 15)
  colors = ["white", "cyan", "orange"]

  for p, color in zip(centers, colors):
    metric = get_monge_metric(f, p)
    J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)
    J = J_v_to_x.value

    for v0 in v_vals:
      v_line = jnp.stack([jnp.full_like(v_vals, v0), v_vals], axis=1)
      x_line = p + jnp.einsum("ij,nj->ni", J, v_line)
      ax.plot(x_line[:, 0], x_line[:, 1], color=color, linewidth=0.8, alpha=0.9)
    for v1 in v_vals:
      v_line = jnp.stack([v_vals, jnp.full_like(v_vals, v1)], axis=1)
      x_line = p + jnp.einsum("ij,nj->ni", J, v_line)
      ax.plot(x_line[:, 0], x_line[:, 1], color=color, linewidth=0.8, alpha=0.9)
    ax.plot(p[0], p[1], "o", color=color, markersize=4)

  ax.set_xlabel("x0")
  ax.set_ylabel("x1")
  ax.set_title("Energy heatmap and RNC grid (Taylor order 1) at 3 points")
  if plot_save_dir is not None:
    fig.savefig(plot_save_dir / "monge_rnc_heatmap.png", dpi=120)
  else:
    plt.show()
  plt.close(fig)


@pytest.mark.plot
def test_monge_metric_rnc_heatmap_funnel(plot_save_dir):
  """
  Plot 2D Neal's funnel energy heatmap with Taylor-approximated RNC grids at 2-3 points.
  Skipped in default runs; use pytest -m plot or pytest -k monge_metric_rnc_heatmap_funnel to run.
  Without --plot-dir: calls plt.show(). With --plot-dir DIR: saves to DIR/monge_rnc_heatmap_funnel.png.
  """
  import matplotlib
  if plot_save_dir is not None:
    matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  funnel = Funnel(ndim=2, sigma_0=3.0)
  alpha = 0.5
  f = lambda x: alpha * (-funnel.energy(x))

  n_sample = 20
  key = random.PRNGKey(42)
  key, sample_key = random.split(key)
  centers = funnel.sample(sample_key, n_sample)

  pad = 0.8
  z0_lim = (
    min(-5.0, float(jnp.min(centers[:, 0])) - pad),
    max(2.66, float(jnp.max(centers[:, 0])) + pad),
  )
  z1_lim = (
    min(-5.0, float(jnp.min(centers[:, 1])) - pad),
    max(5.0, float(jnp.max(centers[:, 1])) + pad),
  )
  n = 200
  x0 = jnp.linspace(z0_lim[0], z0_lim[1], n)
  x1 = jnp.linspace(z1_lim[0], z1_lim[1], n)
  X0, X1 = jnp.meshgrid(x0, x1, indexing="ij")
  grid = jnp.stack([X0.ravel(), X1.ravel()], axis=1)
  E = jax.vmap(funnel.energy)(grid).reshape(n, n)
  Z = jnp.exp(-E)
  Z_flat = Z.ravel()
  vmin = float(jnp.percentile(Z_flat, 2))
  vmax = float(jnp.percentile(Z_flat, 92))

  fig, ax = plt.subplots(figsize=(8, 6))
  ax.pcolormesh(x0, x1, Z.T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
  ax.set_xlim(z0_lim)
  ax.set_ylim(z1_lim)
  ax.set_aspect("equal")

  v_vals = jnp.linspace(-0.4, 0.4, 15)
  cmap = plt.get_cmap("tab10")
  colors = [cmap(i % 10) for i in range(n_sample)]

  for p, color in zip(centers, colors):
    metric = get_monge_metric(f, p)
    J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)
    J = J_v_to_x.value
    J2 = J_v_to_x.gradient

    def x_from_v(v_line):
      order1 = jnp.einsum("ij,nj->ni", J, v_line)
      order2 = 0.5 * jnp.einsum("iab,na,nb->ni", J2, v_line, v_line)
      return p + order1 + order2

    for v0 in v_vals:
      v_line = jnp.stack([jnp.full_like(v_vals, v0), v_vals], axis=1)
      x_line = x_from_v(v_line)
      ax.plot(x_line[:, 0], x_line[:, 1], color=color, linewidth=0.8, alpha=0.9)
    for v1 in v_vals:
      v_line = jnp.stack([v_vals, jnp.full_like(v_vals, v1)], axis=1)
      x_line = x_from_v(v_line)
      ax.plot(x_line[:, 0], x_line[:, 1], color=color, linewidth=0.8, alpha=0.9)
    ax.plot(p[0], p[1], "o", color=color, markersize=4)

  ax.set_xlabel("z0")
  ax.set_ylabel("z1")
  ax.set_title(f"Neal's funnel density and RNC grid (Taylor order 2) at {n_sample} sampled points")
  if plot_save_dir is not None:
    fig.savefig(plot_save_dir / "monge_rnc_heatmap_funnel.png", dpi=120)
  else:
    plt.show()
  plt.close(fig)


@pytest.mark.plot
def test_monge_metric_rnc_heatmap_funnel_alpha_grid(plot_save_dir):
  """ IMPORTANT
  Six panels of funnel heatmap + RNC grids at sampled points for various alpha values.
  Uses x64 precision. Skipped in default runs; use pytest -m plot or -k test_monge_metric_rnc_heatmap_funnel_alpha_grid.
  With --plot-dir DIR: saves to DIR/monge_rnc_heatmap_funnel_alpha_grid.png.
  """
  jax.config.update("jax_enable_x64", True)
  import matplotlib
  if plot_save_dir is not None:
    matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  funnel = Funnel(ndim=2, sigma_0=3.0)
  log_likelihood = lambda x: -funnel.energy(x)

  n_sample = 10
  key = random.PRNGKey(42)
  key, sample_key = random.split(key)
  centers = funnel.sample(sample_key, n_sample)

  pad = 0.8
  z0_lim = (
    min(-5.0, float(jnp.min(centers[:, 0])) - pad),
    max(2.66, float(jnp.max(centers[:, 0])) + pad),
  )
  z1_lim = (
    min(-5.0, float(jnp.min(centers[:, 1])) - pad),
    max(5.0, float(jnp.max(centers[:, 1])) + pad),
  )
  n = 200
  x0 = jnp.linspace(z0_lim[0], z0_lim[1], n)
  x1 = jnp.linspace(z1_lim[0], z1_lim[1], n)
  X0, X1 = jnp.meshgrid(x0, x1, indexing="ij")
  grid = jnp.stack([X0.ravel(), X1.ravel()], axis=1)
  E = jax.vmap(funnel.energy)(grid).reshape(n, n)
  Z = jnp.exp(-E)
  Z_flat = Z.ravel()
  vmin = float(jnp.percentile(Z_flat, 2))
  vmax = float(jnp.percentile(Z_flat, 92))

  v_vals = jnp.linspace(-0.4, 0.4, 15)
  n_v = v_vals.shape[0]
  v_lines_1 = jnp.stack([jnp.stack([jnp.full_like(v_vals, v0), v_vals], axis=1) for v0 in v_vals], axis=0)
  v_lines_2 = jnp.stack([jnp.stack([v_vals, jnp.full_like(v_vals, v1)], axis=1) for v1 in v_vals], axis=0)

  def x_from_v_line(v_line, p, J, J2):
    order1 = jnp.einsum("ij,nj->ni", J, v_line)
    order2 = 0.5 * jnp.einsum("iab,na,nb->ni", J2, v_line, v_line)
    return p + order1 + order2

  def compute_rnc_lines_one_alpha(centers, alpha, v_lines_1, v_lines_2):
    f = lambda x: alpha * log_likelihood(x)
    metrics = jax.vmap(get_monge_metric, (None, 0))(f, centers)
    J_x_to_v, J_v_to_x = jax.vmap(get_rnc_jacobians)(metrics)
    J = J_v_to_x.value
    J2 = J_v_to_x.gradient
    x_f1 = jax.vmap(
      lambda p, J_i, J2_i: jax.vmap(x_from_v_line, (0, None, None, None))(v_lines_1, p, J_i, J2_i),
      (0, 0, 0),
    )(centers, J, J2)
    x_f2 = jax.vmap(
      lambda p, J_i, J2_i: jax.vmap(x_from_v_line, (0, None, None, None))(v_lines_2, p, J_i, J2_i),
      (0, 0, 0),
    )(centers, J, J2)
    return (x_f1, x_f2)

  alphas_arr = jnp.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
  compute_all = jax.jit(
    jax.vmap(compute_rnc_lines_one_alpha, (None, 0, None, None))
  )
  x_f1_all, x_f2_all = compute_all(centers, alphas_arr, v_lines_1, v_lines_2)
  x_f1_all = jnp.asarray(x_f1_all)
  x_f2_all = jnp.asarray(x_f2_all)

  alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  fig, axes = plt.subplots(2, 3, figsize=(14, 10))
  axes = axes.ravel()
  cmap = plt.get_cmap("tab10")
  colors = [cmap(i % 10) for i in range(n_sample)]

  for idx, (ax, alpha) in enumerate(zip(axes, alphas)):
    ax.pcolormesh(x0, x1, Z.T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xlim(z0_lim)
    ax.set_ylim(z1_lim)
    ax.set_aspect("equal")

    x_f1 = x_f1_all[idx]
    x_f2 = x_f2_all[idx]
    for c, color in enumerate(colors):
      for line in range(n_v):
        ax.plot(x_f1[c, line, :, 0], x_f1[c, line, :, 1], color=color, linewidth=0.8, alpha=0.9)
      for line in range(n_v):
        ax.plot(x_f2[c, line, :, 0], x_f2[c, line, :, 1], color=color, linewidth=0.8, alpha=0.9)
      ax.plot(centers[c, 0], centers[c, 1], "o", color=color, markersize=4)

    ax.set_title(f"alpha = {alpha}")
    if idx >= 3:
      ax.set_xlabel("z0")
    if idx % 3 == 0:
      ax.set_ylabel("z1")

  if plot_save_dir is not None:
    fig.savefig(plot_save_dir / "monge_rnc_heatmap_funnel_alpha_grid.png", dpi=120)
  else:
    plt.show()
  plt.close(fig)


@pytest.mark.plot
def test_monge_metric_funnel_geodesics_heatmap(plot_save_dir):
  """
  Single funnel density heatmap with geodesics integrated along RNC basis directions.
  Geodesics stop when they leave plot bounds or when log-likelihood drops below threshold.
  Skipped in default runs; use pytest -m plot or -k test_monge_metric_funnel_geodesics_heatmap.
  With --plot-dir DIR: saves to DIR/monge_funnel_geodesics_heatmap.png.
  """
  import matplotlib
  if plot_save_dir is not None:
    matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  funnel = Funnel(ndim=2, sigma_0=3.0)
  alpha = 0.5
  f = lambda x: alpha * (-funnel.energy(x))
  scale = 10.0
  num_steps = 200
  log_lik_drop = 3.0
  n_sample = 20
  dim = 2

  key = random.PRNGKey(42)
  key, sample_key = random.split(key)
  centers = funnel.sample(sample_key, n_sample)

  pad = 0.8
  z0_lim = (
    min(-5.0, float(jnp.min(centers[:, 0])) - pad),
    max(2.66, float(jnp.max(centers[:, 0])) + pad),
  )
  z1_lim = (
    min(-5.0, float(jnp.min(centers[:, 1])) - pad),
    max(5.0, float(jnp.max(centers[:, 1])) + pad),
  )
  n = 200
  x0 = jnp.linspace(z0_lim[0], z0_lim[1], n)
  x1 = jnp.linspace(z1_lim[0], z1_lim[1], n)
  X0, X1 = jnp.meshgrid(x0, x1, indexing="ij")
  grid = jnp.stack([X0.ravel(), X1.ravel()], axis=1)
  E = jax.vmap(funnel.energy)(grid).reshape(n, n)
  Z = jnp.exp(-E)
  Z_flat = Z.ravel()
  vmin = float(jnp.percentile(Z_flat, 2))
  vmax = float(jnp.percentile(Z_flat, 92))

  def metric_fn(x):
    return get_monge_metric(f, x)

  def one_geodesic(p, v_comp, num_steps_inner):
    basis_std = get_standard_basis(p)
    v_jet = Jet(
      value=v_comp,
      gradient=jnp.zeros((dim, dim)),
      hessian=jnp.zeros((dim, dim, dim)),
    )
    v = TangentVector(p=p, components=v_jet, basis=basis_std)
    _, trajectory = exponential_map_ode(
      v, metric_fn=metric_fn, num_steps=num_steps_inner, return_trajectory=True
    )
    return trajectory

  @jax.jit
  def compute_all_geodesics(centers_j, scale_j, log_lik_drop_j):
    n_c = centers_j.shape[0]
    num_steps_j = num_steps
    metrics = jax.vmap(
      lambda c: get_monge_metric(f, c)
    )(centers_j)
    J_v_to_x_all = jax.vmap(get_rnc_jacobians)(metrics)
    J_val = J_v_to_x_all[1].value
    p_batch = jnp.repeat(centers_j, 4, axis=0)
    v_batch = scale_j * jnp.stack(
      [
        J_val[:, :, 0],
        -J_val[:, :, 0],
        J_val[:, :, 1],
        -J_val[:, :, 1],
      ],
      axis=1,
    ).reshape(n_c * 4, dim)
    trajectories = jax.vmap(one_geodesic, (0, 0, None))(p_batch, v_batch, num_steps_j)
    log_lik_centers = jax.vmap(lambda x: -funnel.energy(x))(centers_j)
    log_lik_threshold = jnp.repeat(log_lik_centers - log_lik_drop_j, 4)
    z0_lo, z0_hi = z0_lim[0], z0_lim[1]
    z1_lo, z1_hi = z1_lim[0], z1_lim[1]
    in_bounds = (
      (trajectories[:, :, 0] >= z0_lo)
      & (trajectories[:, :, 0] <= z0_hi)
      & (trajectories[:, :, 1] >= z1_lo)
      & (trajectories[:, :, 1] <= z1_hi)
    )
    log_lik_traj = jax.vmap(jax.vmap(lambda x: -funnel.energy(x)))(trajectories)
    log_ok = log_lik_traj >= log_lik_threshold[:, None]
    mask = in_bounds & log_ok
    stop_idx = jnp.maximum(
      jnp.max(jnp.where(mask, jnp.arange(num_steps_j), -1), axis=1), 0
    )
    return trajectories, stop_idx

  trajectories, stop_idx = compute_all_geodesics(
    centers, scale, log_lik_drop
  )
  n_geodesics = trajectories.shape[0]

  cmap = plt.get_cmap("tab10")
  colors = [cmap(i % 10) for i in range(n_sample)]

  fig, ax = plt.subplots(figsize=(8, 6))
  ax.pcolormesh(x0, x1, Z.T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
  ax.set_xlim(z0_lim)
  ax.set_ylim(z1_lim)
  ax.set_aspect("equal")

  for g in range(n_geodesics):
    c = g // 4
    color = colors[c]
    traj = trajectories[g]
    stop = int(stop_idx[g])
    truncated = traj[: stop + 1]
    if len(truncated) >= 2:
      ax.plot(truncated[:, 0], truncated[:, 1], color=color, linewidth=1.0, alpha=0.9)
    elif len(truncated) == 1:
      ax.plot(truncated[0, 0], truncated[0, 1], "o", color=color, markersize=3)
  for c, p in enumerate(centers):
    ax.plot(p[0], p[1], "o", color=colors[c], markersize=5)

  ax.set_xlabel("z0")
  ax.set_ylabel("z1")
  ax.set_title("Funnel density and geodesics along RNC directions")
  if plot_save_dir is not None:
    fig.savefig(plot_save_dir / "monge_funnel_geodesics_heatmap.png", dpi=120)
  else:
    plt.show()
  plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Tests for get_second_fundamental_form
# ──────────────────────────────────────────────────────────────────────

def test_get_second_fundamental_form_returns_matrix():
  """get_second_fundamental_form returns an (n, n) matrix."""
  def f(x):
    return -0.5 * jnp.sum(x**2)

  x = jnp.array([1.0, -0.5])
  h = get_second_fundamental_form(f, x)
  assert h.shape == (2, 2)


def test_second_fundamental_form_linear_f_is_zero():
  """For f(x) = a^T x, Hessian is zero so h = 0."""
  a = jnp.array([3.0, -1.0])

  def f(x):
    return jnp.dot(a, x)

  x = jnp.array([0.5, -0.3])
  h = get_second_fundamental_form(f, x)
  assert jnp.allclose(h, jnp.zeros((2, 2)), atol=1e-12)


def test_second_fundamental_form_quadratic_at_origin():
  """For f(x) = 0.5 x^T A x at x=0, grad_f=0 so h = Hessian(f) = A."""
  A = jnp.array([[2.0, 0.5], [0.5, 3.0]])

  def f(x):
    return 0.5 * x @ A @ x

  x = jnp.zeros(2)
  h = get_second_fundamental_form(f, x)
  assert jnp.allclose(h, A, atol=1e-12)


def test_second_fundamental_form_quadratic_away_from_origin():
  """For f(x) = 0.5 x^T A x, h = A / sqrt(1 + ||Ax||^2)."""
  A = jnp.array([[2.0, 0.0], [0.0, 5.0]])

  def f(x):
    return 0.5 * x @ A @ x

  x = jnp.array([1.0, 0.5])
  grad_f = A @ x
  norm_sq = jnp.dot(grad_f, grad_f)
  expected = A / jnp.sqrt(1.0 + norm_sq)
  h = get_second_fundamental_form(f, x)
  assert jnp.allclose(h, expected, atol=1e-12)


def test_second_fundamental_form_symmetric():
  """Second fundamental form matrix is symmetric."""
  def f(x):
    return jnp.sin(x[0]) * jnp.cos(x[1]) + 0.3 * x[0] ** 2

  x = jnp.array([0.7, -0.4])
  h = get_second_fundamental_form(f, x)
  assert jnp.allclose(h, h.T, atol=1e-12)


def test_second_fundamental_form_g_times_s_equals_h():
  """From the notes, g @ s = h where g is Monge metric and s is shape operator."""
  def f(x):
    return jnp.sin(x[0]) * jnp.cos(x[1]) + 0.3 * x[0] ** 2

  x = jnp.array([0.7, -0.4])
  g = get_monge_metric(f, x).components.value
  s = get_shape_matrix(f, x)
  h = get_second_fundamental_form(f, x)
  assert jnp.allclose(g @ s, h, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# Tests for get_shape_matrix
# ──────────────────────────────────────────────────────────────────────

def test_shape_matrix_constant_f_is_zero():
  """When f is constant, hess_f = 0 so the shape operator is the zero matrix."""
  def f(x):
    return 0.0 * jnp.sum(x)

  x = jnp.array([1.0, 2.0])
  s = get_shape_matrix(f, x)
  assert s.shape == (2, 2)
  assert jnp.allclose(s, jnp.zeros((2, 2)), atol=1e-12)


def test_shape_matrix_linear_f_is_zero():
  """A linear f defines a plane, which has zero curvature everywhere."""
  a = jnp.array([3.0, -1.0])

  def f(x):
    return jnp.dot(a, x)

  x = jnp.array([0.5, -0.3])
  s = get_shape_matrix(f, x)
  assert jnp.allclose(s, jnp.zeros((2, 2)), atol=1e-12)


def test_shape_matrix_quadratic_at_origin():
  """
  For f(x) = 0.5 x^T A x, grad_f(0) = 0 and hess_f = A, so s(0) = A.
  """
  A = jnp.array([[2.0, 0.5], [0.5, 3.0]])

  def f(x):
    return 0.5 * x @ A @ x

  x = jnp.zeros(2)
  s = get_shape_matrix(f, x)
  assert jnp.allclose(s, A, atol=1e-12)


def test_shape_matrix_quadratic_away_from_origin():
  """
  For f(x) = 0.5 x^T A x away from the origin, verify against the full formula
    s = (1 / sqrt(1 + ||Ax||^2)) * (I - Ax (Ax)^T / (1 + ||Ax||^2)) @ A
  """
  A = jnp.array([[2.0, 0.0], [0.0, 5.0]])

  def f(x):
    return 0.5 * x @ A @ x

  x = jnp.array([1.0, 0.5])
  grad_f = A @ x
  norm_sq = jnp.dot(grad_f, grad_f)
  g_inv = jnp.eye(2) - jnp.outer(grad_f, grad_f) / (1.0 + norm_sq)
  expected = g_inv @ A / jnp.sqrt(1.0 + norm_sq)

  s = get_shape_matrix(f, x)
  assert jnp.allclose(s, expected, atol=1e-12)


def test_shape_matrix_g_times_s_is_symmetric():
  """
  g @ s equals the scalar second fundamental form h, which is symmetric.
  """
  def f(x):
    return jnp.sin(x[0]) * jnp.cos(x[1]) + 0.3 * x[0] ** 2

  x = jnp.array([0.7, -0.4])
  s = get_shape_matrix(f, x)
  grad_f = jax.grad(f)(x)
  g = jnp.eye(2) + jnp.outer(grad_f, grad_f)
  gs = g @ s
  assert jnp.allclose(gs, gs.T, atol=1e-12)


def test_shape_matrix_1d_classical_curvature():
  """
  In 1D, the shape operator is a scalar equal to the classical curvature
    kappa = f'' / (1 + f'^2)^{3/2}
  """
  def f(x):
    return x[0] ** 2

  x = jnp.array([1.5])
  s = get_shape_matrix(f, x)
  fp = 2.0 * x[0]
  fpp = 2.0
  expected_kappa = fpp / (1.0 + fp ** 2) ** 1.5
  assert jnp.allclose(s[0, 0], expected_kappa, atol=1e-12)


def test_shape_matrix_higher_dim():
  """Shape matrix works in dimensions higher than 2."""
  def f(x):
    return jnp.sum(x ** 2)

  x = jnp.zeros(5)
  s = get_shape_matrix(f, x)
  assert s.shape == (5, 5)
  # At the origin, grad_f = 0 and hess_f = 2I, so s = 2I
  assert jnp.allclose(s, 2.0 * jnp.eye(5), atol=1e-12)


@pytest.mark.plot
def test_shape_matrix_eigenvectors_quadratic(plot_save_dir):
  """
  Plot eigenvectors and eigenvalues of the shape operator for an anisotropic
  quadratic f(x) = 0.5 x^T diag(a) x on a grid of points over the energy
  heatmap. Eigenvectors are drawn as arrows scaled by eigenvalue magnitude.
  """
  import matplotlib
  if plot_save_dir is not None:
    matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  a = jnp.array([1.0, 4.0])

  def f(x):
    return 0.5 * jnp.sum(a * x ** 2)

  # Energy heatmap
  n = 100
  lo, hi = -2.5, 2.5
  xs = jnp.linspace(lo, hi, n)
  X0, X1 = jnp.meshgrid(xs, xs, indexing="ij")
  grid = jnp.stack([X0.ravel(), X1.ravel()], axis=1)
  E = jax.vmap(f)(grid).reshape(n, n)

  fig, ax = plt.subplots(figsize=(8, 8))
  ax.pcolormesh(xs, xs, E.T, shading="auto", cmap="viridis")
  ax.set_aspect("equal")

  # Compute shape operator on a coarser grid
  ng = 9
  pts = jnp.linspace(-2.0, 2.0, ng)
  P0, P1 = jnp.meshgrid(pts, pts, indexing="ij")
  centers = jnp.stack([P0.ravel(), P1.ravel()], axis=1)

  shape_matrices = jax.vmap(lambda p: get_shape_matrix(f, p))(centers)
  evals, evecs = jnp.linalg.eig(shape_matrices)
  evals = evals.real
  evecs = evecs.real

  arrow_len = 0.15
  colors = ["white", "cyan"]
  for i in range(centers.shape[0]):
    p = centers[i]
    for d in range(2):
      v = evecs[i, :, d]
      v = v / jnp.linalg.norm(v)
      dx = arrow_len * v
      ax.annotate(
        "", xy=(float(p[0] + dx[0]), float(p[1] + dx[1])),
        xytext=(float(p[0]), float(p[1])),
        arrowprops=dict(arrowstyle="->", color=colors[d], lw=1.5),
      )

  ax.set_xlabel("x0")
  ax.set_ylabel("x1")
  ax.set_title(
    f"Shape operator principal directions, f(x) = 0.5*(x0^2 + {a[1]:.0f}*x1^2)"
  )
  if plot_save_dir is not None:
    fig.savefig(
      plot_save_dir / "shape_eigenvectors_quadratic.png", dpi=120
    )
  else:
    plt.show()
  plt.close(fig)


@pytest.mark.plot
def test_shape_matrix_eigenvectors_funnel(plot_save_dir):
  """
  Plot eigenvectors and eigenvalues of the shape operator for Neal's funnel
  distribution. Arrows are drawn at sampled points over the density heatmap,
  with each principal direction shown in a different colour and scaled by
  the absolute eigenvalue.
  """
  import matplotlib
  if plot_save_dir is not None:
    matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  funnel = Funnel(ndim=2, sigma_0=3.0)
  alpha = 1.0

  def f(x):
    return jnp.exp(alpha * (-funnel.energy(x)))

  # Sample points for eigenvector overlay
  n_sample = 30
  key = random.PRNGKey(42)
  key, sample_key = random.split(key)
  centers = funnel.sample(sample_key, n_sample)

  pad = 0.8
  z0_lim = (
    min(-5.0, float(jnp.min(centers[:, 0])) - pad),
    max(2.66, float(jnp.max(centers[:, 0])) + pad),
  )
  z1_lim = (
    min(-5.0, float(jnp.min(centers[:, 1])) - pad),
    max(5.0, float(jnp.max(centers[:, 1])) + pad),
  )

  # Density heatmap
  n = 200
  x0 = jnp.linspace(z0_lim[0], z0_lim[1], n)
  x1 = jnp.linspace(z1_lim[0], z1_lim[1], n)
  X0, X1 = jnp.meshgrid(x0, x1, indexing="ij")
  grid = jnp.stack([X0.ravel(), X1.ravel()], axis=1)
  E = jax.vmap(funnel.energy)(grid).reshape(n, n)
  Z = jnp.exp(-E)
  Z_flat = Z.ravel()
  vmin = float(jnp.percentile(Z_flat, 2))
  vmax = float(jnp.percentile(Z_flat, 92))

  fig, ax = plt.subplots(figsize=(8, 6))
  ax.pcolormesh(
    x0, x1, Z.T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax
  )
  ax.set_xlim(z0_lim)
  ax.set_ylim(z1_lim)
  ax.set_aspect("equal")

  # Compute shape operator at each sample point
  shape_matrices = jax.vmap(lambda p: get_shape_matrix(f, p))(centers)
  evals, evecs = jnp.linalg.eig(shape_matrices)
  evals = evals.real
  evecs = evecs.real

  # Verify that eigenvectors are g-orthogonal (not Euclidean-orthogonal).
  # The shape operator is self-adjoint w.r.t. g, so e1^T g e2 = 0.
  def get_metric_matrix(p):
    grad_f = jax.grad(f)(p)
    return jnp.eye(p.shape[0]) + jnp.outer(grad_f, grad_f)
  metrics = jax.vmap(get_metric_matrix)(centers)
  for i in range(n_sample):
    e0 = evecs[i, :, 0]
    e1 = evecs[i, :, 1]
    g_mat = metrics[i]
    g_inner = e0 @ g_mat @ e1
    assert jnp.abs(g_inner) < 1e-4, (
      f"Eigenvectors not g-orthogonal at point {i}: e0^T g e1 = {g_inner}"
    )

  arrow_len = 0.3
  colors = ["white", "cyan"]
  for i in range(n_sample):
    p = centers[i]
    for d in range(2):
      v = evecs[i, :, d]
      v = v / jnp.linalg.norm(v)
      dx = arrow_len * v
      ax.annotate(
        "", xy=(float(p[0] + dx[0]), float(p[1] + dx[1])),
        xytext=(float(p[0]), float(p[1])),
        arrowprops=dict(arrowstyle="->", color=colors[d], lw=1.5),
      )
    ax.plot(float(p[0]), float(p[1]), "o", color="red", markersize=3)

  ax.set_xlabel("z0")
  ax.set_ylabel("z1")
  ax.set_title(
    "Shape operator principal directions on Neal's funnel "
    f"(alpha={alpha})"
  )
  if plot_save_dir is not None:
    fig.savefig(
      plot_save_dir / "shape_eigenvectors_funnel.png", dpi=120
    )
  else:
    plt.show()
  plt.close(fig)


@pytest.mark.plot
def test_hessian_eigenvectors_funnel(plot_save_dir):
  """
  Plot eigenvectors of the Hessian of the energy on Neal's funnel.

  Unlike the shape operator eigenvectors (which are g-orthogonal), the
  Hessian is symmetric so its eigenvectors are Euclidean-orthogonal.
  """
  import matplotlib
  if plot_save_dir is not None:
    matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  funnel = Funnel(ndim=2, sigma_0=3.0)

  n_sample = 30
  key = random.PRNGKey(42)
  key, sample_key = random.split(key)
  centers = funnel.sample(sample_key, n_sample)

  pad = 0.8
  z0_lim = (
    min(-5.0, float(jnp.min(centers[:, 0])) - pad),
    max(2.66, float(jnp.max(centers[:, 0])) + pad),
  )
  z1_lim = (
    min(-5.0, float(jnp.min(centers[:, 1])) - pad),
    max(5.0, float(jnp.max(centers[:, 1])) + pad),
  )

  # Density heatmap
  n = 200
  x0 = jnp.linspace(z0_lim[0], z0_lim[1], n)
  x1 = jnp.linspace(z1_lim[0], z1_lim[1], n)
  X0, X1 = jnp.meshgrid(x0, x1, indexing="ij")
  grid = jnp.stack([X0.ravel(), X1.ravel()], axis=1)
  E = jax.vmap(funnel.energy)(grid).reshape(n, n)
  Z = jnp.exp(-E)
  Z_flat = Z.ravel()
  vmin = float(jnp.percentile(Z_flat, 2))
  vmax = float(jnp.percentile(Z_flat, 92))

  fig, ax = plt.subplots(figsize=(8, 6))
  ax.pcolormesh(
    x0, x1, Z.T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax
  )
  ax.set_xlim(z0_lim)
  ax.set_ylim(z1_lim)
  ax.set_aspect("equal")

  # Compute Hessian of the energy at each sample point
  hessians = jax.vmap(jax.hessian(funnel.energy))(centers)
  evals, evecs = jnp.linalg.eigh(hessians)

  arrow_len = 0.3
  colors = ["white", "cyan"]
  for i in range(n_sample):
    p = centers[i]
    for d in range(2):
      v = evecs[i, :, d]
      v = v / jnp.linalg.norm(v)
      dx = arrow_len * v
      ax.annotate(
        "", xy=(float(p[0] + dx[0]), float(p[1] + dx[1])),
        xytext=(float(p[0]), float(p[1])),
        arrowprops=dict(arrowstyle="->", color=colors[d], lw=1.5),
      )
    ax.plot(float(p[0]), float(p[1]), "o", color="red", markersize=3)

  ax.set_xlabel("z0")
  ax.set_ylabel("z1")
  ax.set_title("Hessian eigenvectors of energy on Neal's funnel")
  if plot_save_dir is not None:
    fig.savefig(
      plot_save_dir / "hessian_eigenvectors_funnel.png", dpi=120
    )
  else:
    plt.show()
  plt.close(fig)
