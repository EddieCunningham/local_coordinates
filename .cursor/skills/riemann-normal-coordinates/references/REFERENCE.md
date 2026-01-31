# Riemann Normal Coordinates Mathematical Reference

## Definition

Given a Riemannian manifold (M, g) and a point p in M, **Riemann normal coordinates** (RNC) are constructed using the exponential map and an orthonormal frame at p:

```
v(q) = E^{-1} o log_p(q)
```

where E(v) = v^i E_i maps coordinates to tangent vectors, and log_p is the inverse of the exponential map.

## Key Properties at the Origin

In RNC centered at p, the following hold at v = 0:

1. **Metric is identity**: g_{ij}(0) = delta_{ij}
2. **First derivatives vanish**: dg_{ij}/dv^k|_{v=0} = 0
3. **Christoffel symbols vanish**: Gamma^k_{ij}(0) = 0
4. **Geodesics are straight lines**: gamma(t) = tv in v-coordinates

## Metric Taylor Expansion

The metric in RNC has the following Taylor expansion:

```
g_{ij}(v) = delta_{ij} + (1/3) R_{kilj}(p) v^k v^l + O(|v|^3)
```

where R_{kilj} are the components of the Riemann curvature tensor at p.

## Metric Derivatives

At the origin v = 0:

**First derivative (vanishes)**:
```
dg_{ij}/dv^k|_{v=0} = 0
```

**Second derivative (encodes curvature)**:
```
d^2g_{ij}/dv^a dv^b|_{v=0} = (1/3)(R_{aibj}(p) + R_{biaj}(p))
```

## Christoffel Symbol Gradient

At the origin:
```
dGamma^k_{ij}/dv^l|_{v=0} = (1/3)(R^k_{ijl} + R^k_{jil})
```

## Log Determinant Identity

The second derivative of the log determinant relates to the Ricci tensor:
```
d^2 log(det(g))/dv^i dv^j|_{v=0} = -(2/3) Ric_{ij}
```

## Coordinate Transformation Taylor Coefficients

For the map x(v) from RNC to original coordinates:

**First order** (orthonormal frame):
```
dx^i/dv^j = J^i_j
```

**Second order** (involves Christoffel symbols):
```
d^2x^i/dv^j dv^k = -Gamma_bar^i_{ab} J^a_j J^b_k
```

**Third order** (symmetrized):
```
d^3x^i/dv^a dv^b dv^c = Sym_{abc}(-dGamma_bar^i_{jk}/dx^m J^m_a J^j_b J^k_c 
                                   + 2 Gamma_bar^i_{jk} Gamma_bar^j_{mn} J^m_a J^n_b J^k_c)
```

where Sym_{abc} averages over all 6 permutations of (a, b, c).

## Jacobi Fields in RNC

A simple Jacobi field along a geodesic gamma(t) = t E_0 is:
```
S(t) = t (d/dv^i)
```

This satisfies:
- S(0) = 0
- nabla_T S(0) = d/dv^i
- nabla_T^2 S = R(T, S)T (Jacobi equation)

## Key Invariants

1. **Metric identity at origin**: g_{ij}(0) = delta_{ij}
2. **Vanishing metric gradient**: dg_{ij}/dv^k|_0 = 0
3. **Vanishing Christoffel symbols**: Gamma^k_{ij}(0) = 0
4. **Orthonormal basis**: g(E_i, E_j)|_p = delta_{ij}
5. **Taylor coefficient symmetry**: d^2x/dv^j dv^k and d^3x/dv^a dv^b dv^c are symmetric
6. **Ricci scalar invariance**: R = g^{ab} R_{ab} is coordinate-invariant

## Implementation Notes

- `get_rnc_jacobians(metric)` returns (J_x_to_v, J_v_to_x) for coordinate transformations
- `get_rnc_basis(metric)` returns the RNC basis vectors in original coordinates
- `to_riemann_normal_coordinates(obj, metric)` transforms various geometric objects
- The Jacobian J_v_to_x encodes the map x(v), J_x_to_v encodes v(x)
- RNC are only valid locally around the center point p
