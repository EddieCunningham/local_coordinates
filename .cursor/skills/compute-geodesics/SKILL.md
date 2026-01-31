---
name: compute-geodesics
description: Compute geodesics using the exponential and logarithmic maps. Use when computing shortest paths, geodesic distances, parallel transport, or geodesic interpolation on Riemannian manifolds.
---

# Compute Geodesics

Compute geodesics using the exponential and logarithmic maps.

## When to Use

- User wants to compute shortest paths on a curved manifold
- User needs the exponential map exp_p(v)
- User needs the logarithmic map log_p(q)
- User wants to trace geodesic trajectories
- User mentions "geodesic", "shortest path", "exponential map"

## Key Imports

```python
import jax
import jax.numpy as jnp
from local_coordinates.jet import Jet, function_to_jet
from local_coordinates.basis import get_standard_basis
from local_coordinates.metric import RiemannianMetric
from local_coordinates.tangent import TangentVector
from local_coordinates.exponential_map import (
    exponential_map,
    exponential_map_taylor,
    exponential_map_ode,
    logarithmic_map_taylor
)
```

## Prerequisites

```python
jax.config.update("jax_enable_x64", True)
```

## Background

The exponential map exp_p: T_p M -> M takes a tangent vector v at p and returns the endpoint of the geodesic starting at p with initial velocity v at parameter t=1.

The logarithmic map log_p: M -> T_p M is the inverse, computing the initial velocity v such that exp_p(v) = q.

Two implementations are available:
1. **Taylor approximation**: Fast, accurate locally around p
2. **ODE solver**: Slower, but globally accurate for large displacements

## Instructions

### Step 1: Create a Metric

```python
def metric_components(x):
    return jnp.array([
        [1 + 0.1*x[0]**2, 0.0],
        [0.0, 1 + 0.1*x[1]**2]
    ])

p = jnp.array([0.0, 0.0])
basis = get_standard_basis(p)
metric_jet = function_to_jet(metric_components, p)
metric = RiemannianMetric(basis=basis, components=metric_jet)
```

### Step 2: Create a Tangent Vector

```python
v_components = jnp.array([1.0, 0.5])
v_jet = Jet(
    value=v_components,
    gradient=jnp.zeros((2, 2)),
    hessian=jnp.zeros((2, 2, 2))
)
v = TangentVector(p=p, components=v_jet, basis=basis)
```

### Step 3a: Exponential Map (Taylor Method)

Fast, local approximation:

```python
q = exponential_map(metric, v, method="taylor")
print(f"exp_p(v) = {q}")
```

Or directly:

```python
q = exponential_map_taylor(metric, v)
```

### Step 3b: Exponential Map (ODE Method)

Globally accurate using numerical integration:

```python
# Define a function that returns the metric at any point
def make_metric(x):
    basis = get_standard_basis(x)
    metric_jet = function_to_jet(metric_components, x)
    return RiemannianMetric(basis=basis, components=metric_jet)

q = exponential_map_ode(v, metric_fn=make_metric, num_steps=100)
print(f"exp_p(v) = {q}")
```

### Step 4: Get the Full Geodesic Trajectory

```python
ts, trajectory = exponential_map_ode(
    v,
    metric_fn=make_metric,
    num_steps=50,
    return_trajectory=True
)

print("Geodesic trajectory:")
for i in range(0, len(ts), 10):
    print(f"  t={ts[i]:.2f}: ({trajectory[i, 0]:.4f}, {trajectory[i, 1]:.4f})")
```

### Step 5: Logarithmic Map

Compute the tangent vector from p to q:

```python
# Given a target point q
q = jnp.array([0.8, 0.4])

# Compute log_p(q) - the tangent vector that maps to q
v_rnc = logarithmic_map_taylor(metric, q)
print(f"log_p(q) in RNC = {v_rnc}")
```

## Complete Example

```python
import jax
import jax.numpy as jnp
from local_coordinates.jet import Jet, function_to_jet
from local_coordinates.basis import get_standard_basis
from local_coordinates.metric import RiemannianMetric
from local_coordinates.tangent import TangentVector
from local_coordinates.exponential_map import (
    exponential_map_taylor,
    exponential_map_ode,
    logarithmic_map_taylor
)

jax.config.update("jax_enable_x64", True)

# Define metric (hyperbolic-like, varies with position)
def metric_components(x):
    scale = 1.0 / (1.0 + 0.1 * jnp.sum(x**2))
    return scale * jnp.eye(2)

def make_metric(x):
    basis = get_standard_basis(x)
    g_jet = function_to_jet(metric_components, x)
    return RiemannianMetric(basis=basis, components=g_jet)

# Starting point and initial velocity
p = jnp.array([0.0, 0.0])
v_components = jnp.array([1.0, 0.5])

# Create tangent vector
basis = get_standard_basis(p)
v_jet = Jet(
    value=v_components,
    gradient=jnp.zeros((2, 2)),
    hessian=jnp.zeros((2, 2, 2))
)
v = TangentVector(p=p, components=v_jet, basis=basis)

# Compute exponential map with Taylor method
metric = make_metric(p)
q_taylor = exponential_map_taylor(metric, v)
print(f"exp_p(v) via Taylor: {q_taylor}")

# Compute exponential map with ODE method
q_ode = exponential_map_ode(v, metric_fn=make_metric, num_steps=100)
print(f"exp_p(v) via ODE: {q_ode}")

# Get full trajectory
ts, trajectory = exponential_map_ode(
    v,
    metric_fn=make_metric,
    num_steps=50,
    return_trajectory=True
)

print("\nGeodesic trajectory:")
for i in range(0, len(ts), 10):
    print(f"  t={ts[i]:.2f}: ({trajectory[i, 0]:.4f}, {trajectory[i, 1]:.4f})")

# Compute logarithmic map (inverse)
v_recovered = logarithmic_map_taylor(metric, q_taylor)
print(f"\nlog_p(q) recovers v: {v_recovered}")
```

## Choosing Between Methods

### Taylor Method (`method="taylor"`)
- **Pros**: Fast, no ODE solver overhead
- **Cons**: Only accurate locally around p
- **Use when**: Computing many short geodesic segments, local approximations

### ODE Method (`method="ode"`)
- **Pros**: Globally accurate, can handle large displacements
- **Cons**: Slower, requires defining `metric_fn`
- **Use when**: Computing long geodesics, need high accuracy

## Geodesic Interpolation

To interpolate along a geodesic from p to q:

```python
# Find the direction from p to q
v_rnc = logarithmic_map_taylor(metric, q)

# Interpolate at parameter t in [0, 1]
def geodesic_point(t):
    scaled_v = TangentVector(
        p=p,
        components=Jet(
            value=t * v_rnc,  # Scale the velocity
            gradient=jnp.zeros((2, 2)),
            hessian=jnp.zeros((2, 2, 2))
        ),
        basis=basis
    )
    return exponential_map_taylor(metric, scaled_v)

# Get points along geodesic
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    pt = geodesic_point(t)
    print(f"t={t}: {pt}")
```

## Important Notes

- The exponential map solves the geodesic equation with Christoffel symbols
- Taylor method uses third-order approximation from RNC Jacobians
- ODE method uses the Dopri5 solver from diffrax
- The logarithmic map returns components in Riemann normal coordinates
- For flat space (Euclidean metric), geodesics are straight lines
