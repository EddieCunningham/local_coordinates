---
name: riemann-normal-coordinates
description: Transform geometric objects to Riemann normal coordinates where the metric is identity and Christoffel symbols vanish at the origin. Use for simplifying local computations, verifying geometric identities, or working with geodesics.
---

# Riemann Normal Coordinates

Transform geometric objects to Riemann normal coordinates (RNC).

## When to Use

- User wants to simplify local geometric computations
- User needs coordinates where the metric is locally flat
- User wants to verify that Christoffel symbols vanish at a point
- User is working with geodesics emanating from a point
- User mentions "normal coordinates", "geodesic coordinates", "locally flat"

## Key Imports

```python
import jax
import jax.numpy as jnp
from local_coordinates.jet import function_to_jet
from local_coordinates.basis import get_standard_basis
from local_coordinates.metric import RiemannianMetric
from local_coordinates.connection import get_levi_civita_connection
from local_coordinates.normal_coords import (
    get_rnc_jacobians,
    get_rnc_basis,
    to_riemann_normal_coordinates,
    get_transformation_to_riemann_normal_coordinates,
    get_transformation_from_riemann_normal_coordinates
)
```

## Prerequisites

```python
jax.config.update("jax_enable_x64", True)
```

## Background

Riemann normal coordinates (RNC) are a special coordinate system centered at a point p where:
- The metric is the identity at p: g_ij(0) = δ_ij
- The Christoffel symbols vanish at p: Γ^k_ij(0) = 0
- The first derivatives of the metric vanish at p: ∂g_ij/∂v^k(0) = 0
- Coordinate lines are geodesics emanating from p

## Instructions

### Step 1: Create a Metric

```python
def metric_components(x):
    return jnp.array([
        [1 + 0.1*x[0]**2, 0.05*x[0]*x[1]],
        [0.05*x[0]*x[1], 1 + 0.1*x[1]**2]
    ])

p = jnp.array([1.0, 1.0])
basis = get_standard_basis(p)
metric_jet = function_to_jet(metric_components, p)
metric = RiemannianMetric(basis=basis, components=metric_jet)
```

### Step 2: Get the RNC Jacobians

The Jacobians encode the coordinate transformation:

```python
J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)

# J_x_to_v: dv/dx (transforms TO RNC)
# J_v_to_x: dx/dv (transforms FROM RNC)
```

### Step 3: Transform the Metric to RNC

```python
metric_rnc = to_riemann_normal_coordinates(metric)

# Verify metric is identity at origin
print("Metric in RNC (should be identity):")
print(metric_rnc.components.value)
```

### Step 4: Transform Other Objects to RNC

```python
# Transform a connection
connection = get_levi_civita_connection(metric)
connection_rnc = to_riemann_normal_coordinates(connection, metric)

# Verify Christoffel symbols vanish
print("Christoffel symbols in RNC (should be ~0):")
print(jnp.max(jnp.abs(connection_rnc.christoffel_symbols.value)))
```

## Transforming Different Object Types

### Metric

```python
metric_rnc = to_riemann_normal_coordinates(metric)
```

### Connection

```python
connection_rnc = to_riemann_normal_coordinates(connection, metric)
```

### Tensor

```python
from local_coordinates.tensor import Tensor
tensor_rnc = to_riemann_normal_coordinates(tensor, metric)
```

### Tangent Vector

```python
from local_coordinates.tangent import TangentVector
vector_rnc = to_riemann_normal_coordinates(vector, metric)
```

### Frame

```python
from local_coordinates.frame import Frame
frame_rnc = to_riemann_normal_coordinates(frame, metric)
```

## Getting the RNC Basis

The RNC basis vectors expressed in original coordinates:

```python
rnc_basis = get_rnc_basis(metric)
print("RNC basis components:")
print(rnc_basis.components.value)
```

## Complete Example

```python
import jax
import jax.numpy as jnp
from local_coordinates.jet import function_to_jet
from local_coordinates.basis import get_standard_basis
from local_coordinates.metric import RiemannianMetric
from local_coordinates.connection import get_levi_civita_connection
from local_coordinates.normal_coords import (
    get_rnc_jacobians,
    to_riemann_normal_coordinates
)

jax.config.update("jax_enable_x64", True)

# Create a non-trivial metric
def metric_components(x):
    return jnp.array([
        [1 + 0.1*x[0]**2, 0.05*x[0]*x[1]],
        [0.05*x[0]*x[1], 1 + 0.1*x[1]**2]
    ])

p = jnp.array([1.0, 1.0])
basis = get_standard_basis(p)
metric_jet = function_to_jet(metric_components, p)
metric = RiemannianMetric(basis=basis, components=metric_jet)

print("Original metric at p:")
print(metric.components.value)

# Get RNC Jacobians
J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)
print("\nJacobian dx/dv:")
print(J_v_to_x.value)

# Transform metric to RNC
metric_rnc = to_riemann_normal_coordinates(metric)
print("\nMetric in RNC (should be identity):")
print(metric_rnc.components.value)

# Verify Christoffel symbols vanish
connection = get_levi_civita_connection(metric)
connection_rnc = to_riemann_normal_coordinates(connection, metric)
print("\nMax Christoffel symbol in RNC (should be ~0):")
print(jnp.max(jnp.abs(connection_rnc.christoffel_symbols.value)))
```

## Efficiency: Reusing Jacobians

When transforming multiple objects, compute the Jacobians once:

```python
# Compute Jacobians once
J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)

# Reuse for multiple transformations
metric_rnc = to_riemann_normal_coordinates(metric, J_x_to_v=J_x_to_v, J_v_to_x=J_v_to_x)
connection_rnc = to_riemann_normal_coordinates(connection, metric, J_x_to_v=J_x_to_v, J_v_to_x=J_v_to_x)
tensor_rnc = to_riemann_normal_coordinates(tensor, metric, J_x_to_v=J_x_to_v, J_v_to_x=J_v_to_x)
```

## Important Notes

- RNC are only valid locally around the center point p
- The metric becomes identity AT THE ORIGIN (v=0), not everywhere
- First derivatives of metric vanish at origin, but second derivatives encode curvature
- The Riemann curvature tensor is invariant and can be read off from metric second derivatives
- RNC are particularly useful for proving local geometric identities
