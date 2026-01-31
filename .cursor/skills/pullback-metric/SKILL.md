---
name: pullback-metric
description: Compute pullback metrics under coordinate transformations. Use when changing coordinate systems (polar, spherical), computing induced metrics on submanifolds, or working with embeddings.
---

# Pullback Metric

Compute the pullback metric f*g under a coordinate transformation f.

## When to Use

- User wants to change coordinate systems (Cartesian to polar, etc.)
- User needs the metric in a different coordinate system
- User is working with embeddings or submanifolds
- User mentions "pullback", "induced metric", "coordinate transformation"

## Key Imports

```python
import jax
import jax.numpy as jnp
from local_coordinates.jet import Jet
from local_coordinates.basis import get_standard_basis
from local_coordinates.metric import RiemannianMetric, pullback_metric
```

## Prerequisites

```python
jax.config.update("jax_enable_x64", True)
```

## Background

Given a map f: M -> N and a metric g on N, the pullback metric f*g on M is defined by:

```
(f*g)_ij(x) = (df^a/dx^i) g_ab(f(x)) (df^b/dx^j)
```

This is the metric that makes f preserve lengths: ||v||_{f*g} = ||df(v)||_g

## Instructions

### Step 1: Define the Coordinate Transformation

Create a function that maps from source coordinates to target coordinates:

```python
def polar_to_cartesian(q):
    r, phi = q[0], q[1]
    return jnp.array([r * jnp.cos(phi), r * jnp.sin(phi)])
```

### Step 2: Create the Target Metric

Create the metric on the target space (where f maps to):

```python
# Euclidean metric on R^2 (Cartesian coordinates)
p_cart = jnp.array([1.0, 0.0])  # Some point in Cartesian coords
basis_cart = get_standard_basis(p_cart)
euclidean_2d = RiemannianMetric(
    basis=basis_cart,
    components=Jet(
        value=jnp.eye(2),
        gradient=jnp.zeros((2, 2, 2)),
        hessian=jnp.zeros((2, 2, 2, 2))
    )
)
```

### Step 3: Compute the Pullback

```python
# Point in source coordinates (polar)
r_val, phi_val = 2.0, jnp.pi / 4.0
p_polar = jnp.array([r_val, phi_val])

# Pullback the Euclidean metric to polar coordinates
polar_metric = pullback_metric(p_polar, polar_to_cartesian, euclidean_2d)

print(f"Metric in polar coordinates:\n{polar_metric.components.value}")
# Expected: [[1.0, 0.0], [0.0, r^2]] = [[1.0, 0.0], [0.0, 4.0]]
```

## Common Examples

### Polar Coordinates

```python
def polar_to_cartesian(q):
    r, phi = q[0], q[1]
    return jnp.array([r * jnp.cos(phi), r * jnp.sin(phi)])

# Pullback Euclidean metric
# Result: ds^2 = dr^2 + r^2 dphi^2
# At r=2: [[1, 0], [0, 4]]
```

### Spherical Coordinates (3D)

```python
def spherical_to_cartesian(q):
    r, theta, phi = q[0], q[1], q[2]
    return jnp.array([
        r * jnp.sin(theta) * jnp.cos(phi),
        r * jnp.sin(theta) * jnp.sin(phi),
        r * jnp.cos(theta)
    ])

# Pullback gives: ds^2 = dr^2 + r^2 dtheta^2 + r^2 sin^2(theta) dphi^2
```

### Elliptical Coordinates

```python
def elliptical_to_cartesian(q):
    u, v = q[0], q[1]
    a = 1.0  # semi-axis parameter
    return jnp.array([
        a * jnp.cosh(u) * jnp.cos(v),
        a * jnp.sinh(u) * jnp.sin(v)
    ])
```

## Complete Example: Polar Coordinates

```python
import jax
import jax.numpy as jnp
from local_coordinates.jet import Jet
from local_coordinates.basis import get_standard_basis
from local_coordinates.metric import RiemannianMetric, pullback_metric

jax.config.update("jax_enable_x64", True)

# Coordinate transformation
def polar_to_cartesian(q):
    r, phi = q[0], q[1]
    return jnp.array([r * jnp.cos(phi), r * jnp.sin(phi)])

# Point in polar coordinates
r_val, phi_val = 2.0, jnp.pi / 4.0
p_polar = jnp.array([r_val, phi_val])
p_cart = polar_to_cartesian(p_polar)

# Euclidean metric in Cartesian coordinates
basis_cart = get_standard_basis(p_cart)
euclidean_2d = RiemannianMetric(
    basis=basis_cart,
    components=Jet(
        value=jnp.eye(2),
        gradient=jnp.zeros((2, 2, 2)),
        hessian=jnp.zeros((2, 2, 2, 2))
    )
)

# Pullback to polar coordinates
polar_metric = pullback_metric(p_polar, polar_to_cartesian, euclidean_2d)

print("Euclidean metric in Cartesian coords:")
print(euclidean_2d.components.value)

print("\nPullback metric in polar coords:")
print(polar_metric.components.value)
# Should be [[1, 0], [0, r^2]] = [[1, 0], [0, 4]]

# Verify: g_phiphi = r^2
expected_g_phiphi = r_val**2
actual_g_phiphi = polar_metric.components.value[1, 1]
print(f"\nExpected g_φφ = r² = {expected_g_phiphi}")
print(f"Actual g_φφ = {actual_g_phiphi}")
```

## Verifying the Pullback

The pullback preserves the line element. For polar coords:
- Cartesian: ds² = dx² + dy²
- Polar: ds² = dr² + r²dφ²

Both should give the same arc length for any curve.

## Important Notes

- The current implementation requires source and target spaces to have the same dimension
- For same-dimension transformations (coordinate changes), this works seamlessly
- The pullback metric automatically tracks derivatives via Jets
- The result is expressed in the standard basis of the source coordinates
- Mappings between different-dimensional spaces require manual handling
