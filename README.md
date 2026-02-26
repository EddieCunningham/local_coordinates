# local_coordinates

A JAX-based framework for differential geometry computations on Riemannian manifolds.

## Overview

The `local_coordinates` library provides a type-safe, coordinate-aware system for performing differential geometry computations. It leverages JAX's automatic differentiation to compute not just gradients, but also Hessians, enabling second-order geometric computations like curvature tensors and geodesics.

**Key use cases:**

- Computing Riemannian metrics and their properties
- Calculating curvature tensors (Riemann, Ricci, scalar)
- Working with coordinate transformations and pullback metrics
- Computing geodesics via exponential and logarithmic maps
- Transforming to Riemann normal coordinates

## Features

- **Jets**: Second-order automatic differentiation storing value, gradient, and Hessian
- **Riemannian Metrics**: Define metrics, compute inner products, raise and lower indices
- **Connections**: Levi-Civita connection and Christoffel symbols
- **Curvature**: Riemann curvature tensor, Ricci tensor, and scalar curvature
- **Coordinate Transformations**: Pullback metrics under coordinate changes
- **Normal Coordinates**: Transform to Riemann normal coordinates where the metric is locally Euclidean
- **Geodesics**: Exponential and logarithmic maps via Taylor approximation or ODE integration

## Installation

```bash
pip install localgeom
```

For development, clone the repo and install in editable mode:

```bash
git clone https://github.com/EddieCunningham/local_coordinates.git
cd local_coordinates
pip install -e ".[test]"
```

Requires Python 3.12+.

## Quick Start

```python
import jax
import jax.numpy as jnp
from jax import Array
from local_coordinates.jet import Jet, function_to_jet
from local_coordinates.basis import BasisVectors, get_standard_basis
from local_coordinates.metric import RiemannianMetric
from local_coordinates.connection import Connection, get_levi_civita_connection
from local_coordinates.riemann import (
    RiemannCurvatureTensor,
    RicciTensor,
    get_riemann_curvature_tensor,
    get_ricci_tensor,
)

# Define a position-dependent metric
def metric_components(x: Array) -> Array:
    return jnp.array([
        [1 + 0.1*x[0]**2, 0.0],
        [0.0, 1 + 0.1*x[1]**2]
    ])

# Create the metric at a point
p: Array = jnp.array([1.0, 1.0])
basis: BasisVectors = get_standard_basis(p)
metric_jet: Jet = function_to_jet(metric_components, p)
metric: RiemannianMetric = RiemannianMetric(basis=basis, components=metric_jet)

# Compute geometric quantities
connection: Connection = get_levi_civita_connection(metric)
riemann: RiemannCurvatureTensor = get_riemann_curvature_tensor(connection)
ricci: RicciTensor = get_ricci_tensor(connection, R=riemann)

# Scalar curvature
g_inv: Array = jnp.linalg.inv(metric.components.value)
scalar_curvature: Array = jnp.einsum("ij,ij->", g_inv, ricci.components.value)

print(f"Metric at p:\n{metric.components.value}")
print(f"Scalar curvature: {scalar_curvature}")
```

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

## Documentation

For comprehensive documentation with examples, see [TUTORIAL.md](TUTORIAL.md).

AI assistant skills for task-specific guidance are available in `.cursor/skills/`:

| Skill | Description |
|-------|-------------|
| **create-riemannian-metric** | Create RiemannianMetric objects from metric functions |
| **compute-curvature** | Compute Levi-Civita connection, Riemann tensor, Ricci tensor |
| **pullback-metric** | Compute pullback metrics under coordinate transformations |
| **riemann-normal-coordinates** | Transform objects to Riemann normal coordinates |
| **compute-geodesics** | Compute geodesics via exponential/logarithmic maps |
| **jet-differentiation** | Use Jets for second-order automatic differentiation |

## Important Conventions

### Column-Vector Convention

The library uses the column-vector convention throughout:

- Basis vectors are stored as columns of matrices
- `E[:, j]` represents the j-th basis vector
- `E[a, j]` is the a-th component of the j-th basis vector

### Index Conventions

- Christoffel symbols: `Gamma[i, j, k]` = Γ^k\_ij
- Riemann tensor: `R[i, j, k, m]` = R^m\_ijk
- Tensors use 1-based indexing for raise/lower operations

## License

See [LICENSE](LICENSE) for details.
