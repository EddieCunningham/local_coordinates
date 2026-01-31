---
name: create-riemannian-metric
description: Create RiemannianMetric objects for differential geometry computations. Use when defining custom metrics, working with curved geometries, or starting any Riemannian geometry workflow in the local_coordinates library.
---

# Create Riemannian Metric

Create `RiemannianMetric` objects from metric component functions for differential geometry computations.

## When to Use

- User wants to define a metric on a manifold
- User needs to work with custom or position-dependent geometries
- Starting point for computing curvature, geodesics, or coordinate transformations
- User mentions "metric tensor", "inner product", "Riemannian geometry"

## Key Imports

```python
import jax
import jax.numpy as jnp
from local_coordinates.jet import function_to_jet
from local_coordinates.basis import get_standard_basis
from local_coordinates.metric import RiemannianMetric, get_euclidean_metric
```

## Prerequisites

Always enable 64-bit precision for numerical stability:

```python
jax.config.update("jax_enable_x64", True)
```

## Instructions

### Step 1: Define the Metric Components Function

Create a function that returns the metric tensor components g_ij as a function of position:

```python
def metric_components(x):
    # x is a 1D array of coordinates
    # Return a (D, D) symmetric matrix where D = x.shape[0]
    return jnp.array([
        [1 + 0.1*x[0]**2, 0.0],
        [0.0, 1 + 0.1*x[1]**2]
    ])
```

The metric must be symmetric and positive definite.

### Step 2: Choose the Evaluation Point

```python
p = jnp.array([1.0, 1.0])  # Point at which to evaluate
```

### Step 3: Create the Metric Jet

The metric needs its value and derivatives stored in a Jet:

```python
metric_jet = function_to_jet(metric_components, p)
```

This computes:
- `metric_jet.value`: The metric g_ij at p
- `metric_jet.gradient`: First derivatives of the metric
- `metric_jet.hessian`: Second derivatives of the metric

### Step 4: Create the Basis and Metric

```python
basis = get_standard_basis(p)
metric = RiemannianMetric(basis=basis, components=metric_jet)
```

## Common Patterns

### Euclidean Metric (Flat Space)

```python
p = jnp.array([1.0, 2.0])
metric = get_euclidean_metric(p)
# metric.components.value is identity matrix
```

### Diagonal Metric

```python
def diagonal_metric(x):
    return jnp.diag(jnp.array([
        1 + x[0]**2,
        1 + x[1]**2
    ]))
```

### Position-Dependent Conformal Metric

```python
def conformal_metric(x):
    scale = 1.0 / (1.0 + 0.1 * jnp.sum(x**2))
    return scale * jnp.eye(x.shape[0])
```

### Off-Diagonal Metric

```python
def non_diagonal_metric(x):
    return jnp.array([
        [1 + 0.1*x[0]**2, 0.05*x[0]*x[1]],
        [0.05*x[0]*x[1], 1 + 0.1*x[1]**2]
    ])
```

## Evaluating the Metric on Tangent Vectors

To compute the inner product g(X, Y):

```python
from local_coordinates.tangent import TangentVector
from local_coordinates.jet import Jet

# Create tangent vectors
v_jet = Jet(
    value=jnp.array([1.0, 0.0]),
    gradient=jnp.zeros((2, 2)),
    hessian=jnp.zeros((2, 2, 2))
)
X = TangentVector(p=p, components=v_jet, basis=basis)
Y = TangentVector(p=p, components=v_jet, basis=basis)

# Evaluate inner product
inner_product = metric(X, Y)  # Returns a Jet
print(inner_product.value)  # The scalar value
```

## Important Notes

- The library uses the column-vector convention: basis vectors are columns of matrices
- Metrics must be symmetric positive definite
- The `function_to_jet` computes derivatives automatically via JAX autodiff
- All geometric objects carry their derivatives for proper transformation behavior
