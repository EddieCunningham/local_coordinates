# Jet Mathematical Reference

## Definition

Let M be a smooth, d-dimensional manifold, let p be a point in M, and let F: M -> R^n be a smooth function. Given a local coordinate system around p, the **second-order Jet** of F at p is the tuple:

```
J[F]_p = (F_p^k, dF_p^k/dx^i, d^2F_p^k/dx^i dx^j)
```

where i,j = 1,...,d and k = 1,...,n.

## Taylor Expansion

The Jet represents the second-order Taylor expansion of F at p:

```
F(q)^k ≈ F_p^k + Σ_i (dF_p^k/dx^i)(q^i - p^i) + (1/2) Σ_{i,j} (d^2F_p^k/dx^i dx^j)(q^i - p^i)(q^j - p^j)
```

for q in a neighborhood of p.

## Change of Coordinates

When changing from x-coordinates to z-coordinates, the Jet components transform as follows.

### Gradient transformation

```
dF_p^k/dz^i = (dx^a/dz^i) (dF_p^k/dx^a)
```

### Hessian transformation

```
d^2F_p^k/dz^i dz^j = (dx^b/dz^j) [ (d^2F_p^k/dx^d dx^b) - (d^2z^c/dx^b dx^d)(dx^a/dz^c)(dF_p^k/dx^a) ] (dx^d/dz^i)
```

where `dx/dz = (dz/dx)^{-1}` is the inverse Jacobian.

## Pushforward Through Smooth Map

Given a smooth map T: R^n -> R^m and G = T o F, the Jet of G at p is computed via the chain rule.

### Gradient pushforward

```
dG_p^k/dx^i = (dT^k/dF^a) (dF_p^a/dx^i)
```

### Hessian pushforward

```
d^2G_p^k/dx^i dx^j = (dF_p^b/dx^j)(d^2T^k/dF^b dF^a)(dF_p^a/dx^i) + (dT^k/dF^a)(d^2F_p^a/dx^i dx^j)
```

The first term is the "curvature" contribution from the nonlinearity of T, and the second term is the "transport" of the Hessian of F.

## Key Invariants

1. **Value preservation**: The value component F_p is unchanged under coordinate transformations
2. **Chain rule consistency**: Jet composition must satisfy the chain rule for both gradients and Hessians
3. **Coordinate round-trip**: Transforming to new coordinates and back must recover the original Jet
4. **Symmetry**: The Hessian must be symmetric: `d^2F/dx^i dx^j = d^2F/dx^j dx^i`

## Implementation Notes

- The `@jet_decorator` lifts functions to operate on Jets by computing JVPs (Jacobian-vector products)
- `function_to_jet(f, x)` computes J[f]_x using `jax.jacrev` and `jax.jacfwd(jax.jacrev)`
- All Jets in a decorated function must have the same coordinate dimension
- PyTree-valued Jets are supported with consistent structure across value/gradient/hessian
