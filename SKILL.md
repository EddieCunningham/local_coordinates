---
name: local-coordinates
description: JAX-based differential geometry library for Riemannian manifolds. Use when working with metrics, curvature, geodesics, coordinate transformations, or any differential geometry computation that requires automatic differentiation with second-order derivatives.
---

# local_coordinates Library

A JAX-based framework for differential geometry computations on Riemannian manifolds.

## When to Use This Library

- Computing Riemannian metrics and their properties
- Calculating curvature tensors (Riemann, Ricci, scalar)
- Working with coordinate transformations and pullback metrics
- Computing geodesics via exponential and logarithmic maps
- Transforming to Riemann normal coordinates
- Any computation requiring automatic differentiation with gradients AND Hessians

## Key Capabilities

| Capability | Description |
|------------|-------------|
| **Jets** | Second-order automatic differentiation (value, gradient, Hessian) |
| **Metrics** | Riemannian metrics with index raising/lowering |
| **Connections** | Levi-Civita connection and Christoffel symbols |
| **Curvature** | Riemann curvature tensor, Ricci tensor, scalar curvature |
| **Coordinates** | Pullback metrics, Riemann normal coordinates |
| **Geodesics** | Exponential and logarithmic maps |

## Installation

### Using uv (recommended)

```bash
uv sync
```

### Using pip

```bash
pip install -e .
```

### Requirements

- Python 3.12+
- JAX and jaxlib
- Equinox
- jaxtyping
- diffrax (for ODE-based geodesic computation)

## Quick Start

```python
import jax
import jax.numpy as jnp
from local_coordinates.jet import function_to_jet
from local_coordinates.basis import get_standard_basis
from local_coordinates.metric import RiemannianMetric
from local_coordinates.connection import get_levi_civita_connection
from local_coordinates.riemann import get_riemann_curvature_tensor, get_ricci_tensor

# Enable 64-bit precision (required for numerical stability)
jax.config.update("jax_enable_x64", True)

# Define a position-dependent metric
def metric_components(x):
    return jnp.array([
        [1 + 0.1*x[0]**2, 0.0],
        [0.0, 1 + 0.1*x[1]**2]
    ])

# Create the metric at a point
p = jnp.array([1.0, 1.0])
basis = get_standard_basis(p)
metric_jet = function_to_jet(metric_components, p)
metric = RiemannianMetric(basis=basis, components=metric_jet)

# Compute geometric quantities
connection = get_levi_civita_connection(metric)
riemann = get_riemann_curvature_tensor(connection)
ricci = get_ricci_tensor(connection, R=riemann)

# Scalar curvature
g_inv = jnp.linalg.inv(metric.components.value)
scalar_curvature = jnp.einsum("ij,ij->", g_inv, ricci.components.value)

print(f"Metric at p:\n{metric.components.value}")
print(f"Scalar curvature: {scalar_curvature}")
```

## Available Skills

Detailed skills for specific tasks are available in `.cursor/skills/`:

| Skill | Description | Path |
|-------|-------------|------|
| **create-riemannian-metric** | Create RiemannianMetric objects from metric functions | `.cursor/skills/create-riemannian-metric/` |
| **compute-curvature** | Compute Levi-Civita connection, Riemann tensor, Ricci tensor | `.cursor/skills/compute-curvature/` |
| **pullback-metric** | Compute pullback metrics under coordinate transformations | `.cursor/skills/pullback-metric/` |
| **riemann-normal-coordinates** | Transform objects to Riemann normal coordinates | `.cursor/skills/riemann-normal-coordinates/` |
| **compute-geodesics** | Compute geodesics via exponential/logarithmic maps | `.cursor/skills/compute-geodesics/` |
| **jet-differentiation** | Use Jets for second-order automatic differentiation | `.cursor/skills/jet-differentiation/` |

## Library Architecture

```
local_coordinates/
├── jet.py              # Jets: second-order Taylor data (value, gradient, hessian)
├── jacobian.py         # Jacobians for coordinate transformations
├── basis.py            # BasisVectors: tangent space bases
├── frame.py            # Frame: basis vectors with Lie brackets
├── tangent.py          # TangentVector: vectors in tangent spaces
├── tensor.py           # Tensor: generic (k,l) tensors
├── metric.py           # RiemannianMetric: inner products on tangent spaces
├── connection.py       # Connection: Christoffel symbols, covariant derivatives
├── riemann.py          # RiemannCurvatureTensor and RicciTensor
├── normal_coords.py    # Riemann normal coordinates
└── exponential_map.py  # Exponential and logarithmic maps
```

## Important Conventions

### 64-bit Precision

Always enable 64-bit precision at the start of your script:

```python
jax.config.update("jax_enable_x64", True)
```

### Column-Vector Convention

The library uses the column-vector convention throughout:
- Basis vectors are stored as columns of matrices
- `E[:, j]` represents the j-th basis vector
- `E[a, j]` is the a-th component of the j-th basis vector

### Index Conventions

- Christoffel symbols: `Gamma[i, j, k]` = Γ^k_ij
- Riemann tensor: `R[i, j, k, m]` = R^m_ijk
- Tensors use 1-based indexing for raise/lower operations

## Further Reading

- See `TUTORIAL.md` for comprehensive documentation with examples
- See individual skill files in `.cursor/skills/` for task-specific guidance
- See `tests/` directory for usage examples and test cases
