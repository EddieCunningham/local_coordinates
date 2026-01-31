# Geodesics Mathematical Reference

## Geodesic Equation

A **geodesic** is a curve gamma(t) that satisfies the geodesic equation:

```
d^2 gamma^i/dt^2 + Gamma^i_{jk}(gamma) (d gamma^j/dt)(d gamma^k/dt) = 0
```

This is the equation for parallel transport of the velocity vector along itself.

## Exponential Map

The **exponential map** exp_p: T_p M -> M takes a tangent vector v at p and returns the endpoint of the geodesic with initial conditions:
- gamma(0) = p
- d gamma/dt(0) = v

Formally:
```
exp_p(v) = gamma(1)
```

where gamma is the unique geodesic satisfying the initial conditions.

## Logarithmic Map

The **logarithmic map** log_p: M -> T_p M is the inverse of the exponential map:
```
log_p(exp_p(v)) = v
exp_p(log_p(q)) = q
```

(valid within the injectivity radius of p)

## Taylor Approximation

For small displacements, the exponential map can be approximated using Taylor series:

```
x(v) = p + J*v + (1/2)*H*v*v + (1/6)*T*v*v*v + O(v^4)
```

where:
- J = dx/dv (orthonormal frame)
- H = d^2x/dv^2 = -Gamma*J*J (second derivatives)
- T = d^3x/dv^3 (third derivatives, symmetrized)

This uses the RNC Jacobian coefficients.

## ODE Formulation

The geodesic equation is a second-order ODE. Converting to first-order system:

Let y = [gamma, dot_gamma] be the state vector (concatenation of position and velocity).

```
dy/dt = [dot_gamma, -Gamma(gamma) * dot_gamma * dot_gamma]
```

This can be solved using standard ODE integrators (Dopri5, etc.).

## Key Properties

### Identity at Zero

```
exp_p(0) = p
```

The exponential map of the zero vector returns the base point.

### Euclidean Case

For flat (Euclidean) space where Gamma = 0:

```
exp_p(v) = p + v
```

Geodesics are straight lines.

### Geodesic Equation Satisfaction

Along the geodesic gamma(t):

```
nabla_T T = 0
```

where T = d gamma/dt is the tangent vector. The velocity is parallel transported along itself.

### Round-Trip Property

For q sufficiently close to p:

```
exp_p(log_p(q)) ≈ q
log_p(exp_p(v)) ≈ v
```

(exact within the injectivity radius)

### Length Minimization

Geodesics locally minimize arc length:

```
L(gamma) = integral_0^1 sqrt(g(dot_gamma, dot_gamma)) dt
```

is minimized by geodesics among all curves connecting the endpoints.

## Taylor vs ODE Methods

| Method | Pros | Cons | Use When |
|--------|------|------|----------|
| Taylor | Fast, no ODE overhead | Only local accuracy | Short geodesics, many evaluations |
| ODE | Globally accurate | Slower, needs metric_fn | Long geodesics, high accuracy |

## Geodesic Taylor Coefficients

From the geodesic equation, the Taylor coefficients at t=0 are:

**First derivative**:
```
d gamma^i/dt|_{t=0} = v^i
```

**Second derivative** (from geodesic equation):
```
d^2 gamma^i/dt^2|_{t=0} = -Gamma^i_{jk}(p) v^j v^k
```

**Third derivative** (differentiating geodesic equation):
```
d^3 gamma^i/dt^3|_{t=0} = -d Gamma^i_{jk}/dx^m * v^m v^j v^k + 2 Gamma^i_{jk} Gamma^j_{mn} v^m v^n v^k
```

(symmetrized over velocity indices)

## Key Invariants

1. **Zero velocity**: exp_p(0) = p
2. **Euclidean**: exp_p(v) = p + v when Gamma = 0
3. **Round-trip**: log_p(exp_p(v)) ≈ v for small v
4. **Geodesic equation**: nabla_T T = 0 along geodesics

## Implementation Notes

- `exponential_map_taylor(metric, v)` uses RNC Jacobian for fast local computation
- `exponential_map_ode(v, metric_fn, num_steps)` solves geodesic ODE numerically
- `logarithmic_map_taylor(metric, q)` computes inverse using RNC Jacobian
- Taylor method uses third-order approximation
- ODE method uses Dopri5 solver from diffrax
