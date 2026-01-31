# Pullback Metric Mathematical Reference

## Pullback Definition

Given a smooth map f: M -> N and a metric h on N, the **pullback metric** f*h on M is defined by:

```
(f*h)_p(X, Y) = h_{f(p)}(df_p(X), df_p(Y))
```

where df_p is the differential (pushforward) of f at p.

## Component Formula

In local coordinates, if f: x -> y and h is the metric on the y-coordinates, the pullback metric components are:

```
(f*h)_{ij}(x) = (df^a/dx^i) h_{ab}(f(x)) (df^b/dx^j)
```

In matrix notation:

```
g = (Df)^T h (Df)
```

where Df is the Jacobian matrix of f.

## Tensor Transformation Laws

### Covariant (0,2) Tensor (like metric)

For a coordinate change from x to z with Jacobian J_i^a = dx^a/dz^i:

```
g'_{ij} = J_i^a J_j^b g_{ab}
```

### General (k,l) Tensor

For a tensor with k contravariant and l covariant indices:

```
T'^{i_1...i_k}_{j_1...j_l} = G^{i_1}_{a_1} ... G^{i_k}_{a_k} J^{b_1}_{j_1} ... J^{b_l}_{j_l} T^{a_1...a_k}_{b_1...b_l}
```

where:
- G_a^i = dz^i/dx^a (forward Jacobian) for contravariant indices
- J_j^b = dx^b/dz^j (inverse Jacobian) for covariant indices

## Important Special Cases

### Identity Map

For the identity map f(x) = x:

```
pullback_metric(id, h) = h
```

The metric is unchanged.

### Linear Map

For a linear map f(x) = Ax where A is a constant matrix:

```
pullback_metric(Ax, I) = A^T A
```

where I is the identity metric.

### Polar to Cartesian

For polar coordinates (r, phi) mapping to Cartesian (x, y):

```
f(r, phi) = (r cos(phi), r sin(phi))

Df = [[cos(phi), -r sin(phi)],
      [sin(phi),  r cos(phi)]]

pullback of Euclidean = [[1, 0],
                         [0, r^2]]
```

This gives the standard polar metric: ds^2 = dr^2 + r^2 dphi^2

### Spherical to Cartesian

For spherical coordinates (r, theta, phi) mapping to Cartesian (x, y, z):

```
f(r, theta, phi) = (r sin(theta) cos(phi), r sin(theta) sin(phi), r cos(theta))

pullback of Euclidean = [[1, 0, 0],
                         [0, r^2, 0],
                         [0, 0, r^2 sin^2(theta)]]
```

## Derivative Transformation

When computing the pullback metric with derivatives (for curvature calculations), the gradient transforms as:

```
dg'_{ij}/dz^k = J_k^c (dg_{ab}/dx^c) J_i^a J_j^b + (d^2x^a/dz^k dz^i) g_{ab} J_j^b + J_i^a g_{ab} (d^2x^b/dz^k dz^j)
```

This involves both the chain rule and the Hessian of the coordinate transformation.

## Key Invariants

1. **Identity preservation**: pullback_metric(id, h) = h
2. **Composition**: (g o f)*h = f*(g*h)
3. **Isometry preservation**: f is an isometry iff f*h = h
4. **Line element preservation**: ds^2 is invariant under coordinate changes

## Implementation Notes

- `pullback_metric(x, f, g)` computes the pullback at point x under map f of metric g
- The function automatically computes the Jacobian and its derivatives via JAX autodiff
- Result is returned in the standard basis of the source coordinates
- Current implementation requires source and target to have the same dimension
