---
name: jet-differentiation
description: Use Jets for automatic differentiation with second-order derivatives (value, gradient, Hessian). Use when computing derivatives through geometric operations, working with Taylor expansions, or propagating derivatives through compositions.
---

# Jet Differentiation

Use Jets for automatic differentiation with second-order derivatives.

## When to Use

- User needs to compute function values along with gradients and Hessians
- User wants to propagate derivatives through composed operations
- User needs Taylor approximations at a point
- User wants to change coordinates for derivative data
- User mentions "Jet", "Taylor expansion", "second-order derivatives"

## Key Imports

```python
import jax
import jax.numpy as jnp
from local_coordinates.jet import (
    Jet,
    function_to_jet,
    jet_decorator,
    change_coordinates,
    get_identity_jet
)
```

## Prerequisites

```python
jax.config.update("jax_enable_x64", True)
```

## Background

A `Jet` stores the second-order Taylor data of a function F at a point p:
- `value`: F(p), the function value
- `gradient`: ∂F/∂x^i, the first derivatives
- `hessian`: ∂²F/∂x^i∂x^j, the second derivatives

This enables:
- Automatic propagation of derivatives through compositions
- Taylor evaluation at nearby points
- Coordinate transformations of derivative data

## Instructions

### Creating Jets from Functions

Use `function_to_jet` to compute a Jet from a function:

```python
def f(x):
    return x[0]**2 + jnp.sin(x[1])

p = jnp.array([1.0, 0.5])
jet = function_to_jet(f, p)

print(f"Value: {jet.value}")           # f(p) = 1 + sin(0.5) ≈ 1.479
print(f"Gradient: {jet.gradient}")     # [2*x[0], cos(x[1])] = [2.0, 0.877...]
print(f"Hessian:\n{jet.hessian}")      # [[2, 0], [0, -sin(0.5)]]
```

### Creating Jets Manually

For cases where you have precomputed derivatives:

```python
jet = Jet(
    value=jnp.array([1.0, 2.0]),
    gradient=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
    hessian=jnp.zeros((2, 2, 2))
)
```

### Taylor Evaluation

Evaluate the Taylor approximation at a displacement dx from p:

```python
jet = function_to_jet(f, p)
dx = jnp.array([0.1, -0.1])

# Approximate f(p + dx) using Taylor expansion
approx = jet(dx)
# approx ≈ f(p) + grad·dx + 0.5*dx^T·H·dx
```

### The @jet_decorator

Lift functions to operate on Jets, automatically propagating derivatives:

```python
@jet_decorator
def square(x):
    return x**2

# Create an identity jet (input is the coordinate itself)
x_jet = function_to_jet(lambda t: t, jnp.array(3.0))
result = square(x_jet)

print(f"Value: {result.value}")      # 9.0
print(f"Gradient: {result.gradient}")  # [6.0] (2x * dx/dt)
print(f"Hessian: {result.hessian}")    # [[2.0]]
```

### Composing Operations with Jets

Multiple decorated functions compose correctly:

```python
@jet_decorator
def add(x, y):
    return x + y

@jet_decorator
def multiply(x, y):
    return x * y

jet_a = function_to_jet(lambda t: t[0], jnp.array([2.0, 3.0]))
jet_b = function_to_jet(lambda t: t[1], jnp.array([2.0, 3.0]))

sum_jet = add(jet_a, jet_b)
prod_jet = multiply(jet_a, jet_b)
```

### Changing Coordinates

Transform a Jet from x-coordinates to z-coordinates:

```python
from local_coordinates.jet import change_coordinates

# Original jet in x-coordinates
jet_x = function_to_jet(f, x)

# Coordinate transformation z = g(x)
def x_to_z(x):
    return jnp.array([x[0] + x[1], x[0] - x[1]])

# Transform to z-coordinates
jet_z = change_coordinates(jet_x, x_to_z, x)
```

## Complete Example

```python
import jax
import jax.numpy as jnp
from local_coordinates.jet import Jet, function_to_jet, jet_decorator

jax.config.update("jax_enable_x64", True)

# Define a function
def f(x):
    return jnp.array([x[0]**2 + x[1], jnp.sin(x[0] * x[1])])

# Create jet at a point
p = jnp.array([1.0, 2.0])
jet = function_to_jet(f, p)

print("Function value at p:")
print(jet.value)

print("\nJacobian (gradient):")
print(jet.gradient)

print("\nHessian (per output component):")
print(jet.hessian)

# Taylor evaluation
dx = jnp.array([0.1, -0.05])
approx = jet(dx)
exact = f(p + dx)
print(f"\nTaylor approximation: {approx}")
print(f"Exact value: {exact}")
print(f"Error: {jnp.abs(approx - exact)}")

# Using jet_decorator for custom operations
@jet_decorator
def norm_squared(x):
    return jnp.sum(x**2)

result = norm_squared(jet)
print(f"\n||f(p)||² = {result.value}")
print(f"Gradient of ||f||²: {result.gradient}")
```

## Identity Jet

For creating identity transformations:

```python
from local_coordinates.jet import get_identity_jet

# 2x2 identity jet (for coordinate identity)
identity = get_identity_jet(2)
# value = [[1,0],[0,1]], gradient = zeros, hessian = zeros
```

## Common Patterns

### Jet for Metric Components

```python
def metric_components(x):
    return jnp.array([
        [1 + x[0]**2, 0],
        [0, 1 + x[1]**2]
    ])

metric_jet = function_to_jet(metric_components, p)
# metric_jet.value: (2, 2) metric at p
# metric_jet.gradient: (2, 2, 2) metric derivatives
# metric_jet.hessian: (2, 2, 2, 2) metric second derivatives
```

### Extracting Sub-Jets

```python
# Get the gradient as a Jet (for computing gradient of gradient)
gradient_jet = jet.get_gradient_jet()

# Get the hessian as a Jet
hessian_jet = jet.get_hessian_jet()
```

### Scalar Operations on Jets

```python
# Scalar multiplication
scaled_jet = 2.0 * jet  # Right-multiply by scalar

# Addition with scalar
shifted_jet = jet + 1.0
```

## Important Notes

- Jets store derivatives with respect to a chosen coordinate system
- The `@jet_decorator` uses JVP to propagate derivatives through arbitrary functions
- All Jets in a decorated function must have the same coordinate dimension
- The Hessian is automatically symmetrized on construction
- For PyTree-valued functions, `function_to_jet` returns a PyTree of Jets
- Jets with `gradient=None` behave as constants in decorated functions
