---
name: compute-curvature
description: Compute Riemannian curvature from a metric, including Levi-Civita connection, Christoffel symbols, Riemann curvature tensor, Ricci tensor, and scalar curvature. Use when analyzing manifold geometry or checking flatness.
---

# Compute Curvature

Compute geometric curvature quantities from a Riemannian metric.

## When to Use

- User wants to analyze the curvature of a manifold
- User needs Christoffel symbols for geodesic equations
- User wants to compute Riemann, Ricci, or scalar curvature
- User wants to check if a space is flat
- User mentions "connection", "curvature", "Christoffel"

## Key Imports

```python
import jax
import jax.numpy as jnp
from local_coordinates.jet import function_to_jet
from local_coordinates.basis import get_standard_basis
from local_coordinates.metric import RiemannianMetric
from local_coordinates.connection import Connection, get_levi_civita_connection
from local_coordinates.riemann import (
    RiemannCurvatureTensor,
    get_riemann_curvature_tensor,
    RicciTensor,
    get_ricci_tensor
)
```

## Prerequisites

```python
jax.config.update("jax_enable_x64", True)
```

## Instructions

### Step 1: Create a Metric

First, create a `RiemannianMetric` object (see `create-riemannian-metric` skill):

```python
def metric_components(x):
    return jnp.array([
        [1 + 0.1*x[0]**2, 0.0],
        [0.0, 1 + 0.1*x[1]**2]
    ])

p = jnp.array([1.0, 1.0])
basis = get_standard_basis(p)
metric_jet = function_to_jet(metric_components, p)
metric = RiemannianMetric(basis=basis, components=metric_jet)
```

### Step 2: Compute the Levi-Civita Connection

The Levi-Civita connection is the unique torsion-free, metric-compatible connection:

```python
connection = get_levi_civita_connection(metric)
```

### Step 3: Access Christoffel Symbols

The Christoffel symbols are indexed as Gamma^k_ij = christoffel_symbols[i, j, k]:

```python
Gamma = connection.christoffel_symbols.value
print(f"Christoffel symbols shape: {Gamma.shape}")  # (D, D, D)
print(f"Gamma^0_00 = {Gamma[0, 0, 0]}")
```

### Step 4: Compute Riemann Curvature Tensor

The (3,1) Riemann curvature tensor R^m_ijk:

```python
riemann = get_riemann_curvature_tensor(connection)
R = riemann.components.value
print(f"Riemann tensor shape: {R.shape}")  # (D, D, D, D)
```

Index convention: R[i, j, k, m] = R^m_ijk

### Step 5: Compute Ricci Tensor

The Ricci tensor is the contraction R_ab = R^i_aib:

```python
ricci = get_ricci_tensor(connection, R=riemann)
Ric = ricci.components.value
print(f"Ricci tensor:\n{Ric}")
```

### Step 6: Compute Scalar Curvature

The scalar curvature is R = g^ij R_ij:

```python
g_inv = jnp.linalg.inv(metric.components.value)
scalar_curvature = jnp.einsum("ij,ij->", g_inv, ricci.components.value)
print(f"Scalar curvature: {scalar_curvature}")
```

## Complete Example

```python
import jax
import jax.numpy as jnp
from local_coordinates.jet import function_to_jet
from local_coordinates.basis import get_standard_basis
from local_coordinates.metric import RiemannianMetric
from local_coordinates.connection import get_levi_civita_connection
from local_coordinates.riemann import get_riemann_curvature_tensor, get_ricci_tensor

jax.config.update("jax_enable_x64", True)

# Define metric
def metric_components(x):
    return jnp.array([
        [1 + 0.1*x[0]**2, 0.0],
        [0.0, 1 + 0.1*x[1]**2]
    ])

p = jnp.array([1.0, 1.0])
basis = get_standard_basis(p)
metric_jet = function_to_jet(metric_components, p)
metric = RiemannianMetric(basis=basis, components=metric_jet)

# Compute all curvature quantities
connection = get_levi_civita_connection(metric)
riemann = get_riemann_curvature_tensor(connection)
ricci = get_ricci_tensor(connection, R=riemann)

# Scalar curvature
g_inv = jnp.linalg.inv(metric.components.value)
scalar_curvature = jnp.einsum("ij,ij->", g_inv, ricci.components.value)

print(f"Metric at p:\n{metric.components.value}")
print(f"Christoffel symbols:\n{connection.christoffel_symbols.value}")
print(f"Ricci tensor:\n{ricci.components.value}")
print(f"Scalar curvature: {scalar_curvature}")
```

## Checking Flatness

A space is flat if the Riemann tensor vanishes:

```python
is_flat = jnp.allclose(riemann.components.value, 0.0, atol=1e-10)
print(f"Space is flat: {is_flat}")
```

## Using Covariant Derivatives

The connection provides covariant derivatives:

```python
from local_coordinates.tangent import TangentVector
from local_coordinates.jet import Jet

# Create tangent vectors X and Y
v_jet = Jet(
    value=jnp.array([1.0, 0.0]),
    gradient=jnp.zeros((2, 2)),
    hessian=jnp.zeros((2, 2, 2))
)
X = TangentVector(p=p, components=v_jet, basis=basis)
Y = TangentVector(p=p, components=v_jet, basis=basis)

# Compute covariant derivative nabla_X Y
nabla_X_Y = connection.covariant_derivative(X, Y)
print(f"Covariant derivative components: {nabla_X_Y.components.value}")
```

## Riemann Tensor Symmetries

The Riemann tensor satisfies:
- Skew symmetry: R_ijkl = -R_jikl
- Interchange symmetry: R_ijkl = R_klij
- First Bianchi identity: R_ijkl + R_jkil + R_kijl = 0

## Important Notes

- The Riemann tensor is a (3,1) tensor with index convention R[i, j, k, m] = R^m_ijk
- The Christoffel symbols are NOT tensors and transform specially under coordinate changes
- For flat space (Euclidean), all curvature quantities vanish
- Ricci tensor is symmetric: R_ij = R_ji
