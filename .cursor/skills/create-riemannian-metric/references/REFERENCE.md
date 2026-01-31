# Riemannian Metric Mathematical Reference

## Definition

A **Riemannian metric** g on a manifold M is a smooth assignment of an inner product g_p on each tangent space T_p M. In local coordinates, the metric is represented by a symmetric, positive-definite matrix:

```
g = g_{ij} dx^i ⊗ dx^j
```

where g_{ij}(x) are smooth functions of the coordinates.

## Metric Evaluation

Given two tangent vectors X = X^i E_i and Y = Y^j E_j at a point p, the metric evaluates as:

```
g(X, Y) = g_{ij} X^i Y^j
```

where Einstein summation is implied over repeated indices.

## Key Properties

### Symmetry

The metric tensor is symmetric:

```
g_{ij} = g_{ji}
```

This ensures g(X, Y) = g(Y, X) for all tangent vectors.

### Positive Definiteness

For any nonzero tangent vector X:

```
g(X, X) = g_{ij} X^i X^j > 0
```

This ensures the metric defines a valid inner product.

## Index Manipulation

### Raising an Index

Convert a covariant index to contravariant using the inverse metric:

```
v^i = g^{ij} alpha_j
```

where g^{ij} is the inverse matrix: g^{ik} g_{kj} = delta^i_j.

### Lowering an Index

Convert a contravariant index to covariant using the metric:

```
alpha_i = g_{ij} v^j
```

### Round-Trip Identity

Raising then lowering (or vice versa) recovers the original:

```
g_{ij} g^{jk} v_k = v_i
```

## Basis Transformation

When changing from basis {E_i} to basis {E'_i} with E'_i = T^j_i E_j, the metric components transform as:

```
g'_{ij} = T^k_i T^l_j g_{kl}
```

Or in matrix form:

```
g' = T^T g T
```

where T is the basis transformation matrix.

## Euclidean Metric

The simplest metric is the Euclidean metric:

```
g_{ij} = delta_{ij} (Kronecker delta)
```

This gives the standard dot product:

```
g(X, Y) = X · Y = sum_i X^i Y^i
```

## Position-Dependent Metrics

In general, the metric components can vary with position:

```
g_{ij}(x) = f(x) delta_{ij} + h(x) x_i x_j + ...
```

The derivatives of the metric are needed for computing Christoffel symbols and curvature.

## Key Invariants

1. **Symmetry**: g_{ij} = g_{ji} for all i, j
2. **Positive definiteness**: All eigenvalues of g_{ij} are positive
3. **Index round-trip**: raise(lower(v)) = v and lower(raise(alpha)) = alpha
4. **Identity metric dot product**: When g = I, g(X, Y) = X · Y

## Implementation Notes

- Metric components are stored as a Jet to track derivatives
- The basis vectors are always expressed in the standard Euclidean basis
- Use `function_to_jet(metric_components, p)` to compute metric with derivatives
- 64-bit precision is recommended: `jax.config.update("jax_enable_x64", True)`
