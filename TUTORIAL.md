# local_coordinates Library Tutorial

A comprehensive guide to using the `local_coordinates` library for differential geometry computations in JAX.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [High-Level Architecture](#high-level-architecture)
4. [Core Concepts](#core-concepts)
   - [Jets: Automatic Differentiation](#jets-automatic-differentiation)
   - [Jacobians: Coordinate Maps](#jacobians-coordinate-maps)
   - [Basis Vectors](#basis-vectors)
   - [Frames](#frames)
   - [Tangent Vectors](#tangent-vectors)
   - [Tensors](#tensors)
5. [Riemannian Geometry](#riemannian-geometry)
   - [Riemannian Metrics](#riemannian-metrics)
   - [Connections and Christoffel Symbols](#connections-and-christoffel-symbols)
   - [Curvature Tensors](#curvature-tensors)
6. [Special Coordinate Systems](#special-coordinate-systems)
   - [Riemann Normal Coordinates](#riemann-normal-coordinates)
   - [Exponential and Logarithmic Maps](#exponential-and-logarithmic-maps)
7. [Complete Examples](#complete-examples)
8. [AI Assistant Skills](#ai-assistant-skills)

---

## Introduction

The `local_coordinates` library is a JAX-based framework for performing differential geometry computations on Riemannian manifolds. It provides a type-safe, coordinate-aware system for working with:

- **Automatic differentiation** via Jets, which store function values along with their gradients and Hessians
- **Geometric objects** like basis vectors, frames, tangent vectors, and tensors
- **Riemannian geometry** including metrics, connections, and curvature tensors
- **Coordinate transformations** with proper handling of basis changes and coordinate changes

The library uses the column-vector convention throughout: basis vectors are stored as columns of matrices, so `E[:, j]` represents the j-th basis vector.

---

## Installation

The library requires Python 3.12+ and can be installed using `uv`:

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

### Dependencies

The core dependencies are:
- **JAX** and **jaxlib**: For automatic differentiation and array operations
- **Equinox**: For PyTree-based modules
- **jaxtyping**: For shape annotations and type safety
- **diffrax**: For ODE solvers (used in exponential map)

---

## High-Level Architecture

The library is organized into several interconnected modules:

```
Core Foundations
├── jet.py          # Jets for automatic differentiation (value, gradient, hessian)
└── jacobian.py     # Coordinate map Jacobians with derivatives

Geometric Objects
├── basis.py        # BasisVectors: tangent space bases
├── frame.py        # Frame: sets of basis vectors with Lie brackets
├── tangent.py      # TangentVector: vectors in tangent spaces
└── tensor.py       # Tensor: generic (k,l) tensors

Riemannian Geometry
├── metric.py       # RiemannianMetric: inner products on tangent spaces
├── connection.py   # Connection: Christoffel symbols and covariant derivatives
└── riemann.py      # RiemannCurvatureTensor and RicciTensor

Coordinate Systems
├── normal_coords.py    # Riemann normal coordinates
└── exponential_map.py  # Exponential and logarithmic maps
```

The modules build on each other: Jets underpin everything, basis vectors use Jets to store derivatives, tensors use basis vectors, and so on up to curvature computations.

---

## Core Concepts

### Jets: Automatic Differentiation

A `Jet` represents the second-order Taylor approximation of a function at a point. It stores:
- `value`: The function value F(p)
- `gradient`: First derivatives ∂F/∂x^i
- `hessian`: Second derivatives ∂²F/∂x^i∂x^j

```python
import jax.numpy as jnp
from jax import Array
from local_coordinates.jet import Jet, function_to_jet, jet_decorator

# Create a Jet from a function
def f(x: Array) -> Array:
    return x[0]**2 + jnp.sin(x[1])

p: Array = jnp.array([1.0, 0.5])
jet: Jet = function_to_jet(f, p)

print(f"Value: {jet.value}")           # f(p) = 1 + sin(0.5) ≈ 1.479
print(f"Gradient: {jet.gradient}")     # [2*x[0], cos(x[1])] = [2.0, 0.877...]
print(f"Hessian:\n{jet.hessian}")      # [[2, 0], [0, -sin(0.5)]]
```

#### The `@jet_decorator`

The `@jet_decorator` lifts functions to operate on Jets, automatically propagating derivatives:

```python
@jet_decorator
def square(x: Array) -> Array:
    return x**2

# Create a simple identity jet
x_jet: Jet = function_to_jet(lambda t: t, jnp.array(3.0))
result: Jet = square(x_jet)

print(f"Value: {result.value}")      # 9.0
print(f"Gradient: {result.gradient}")  # [6.0] (chain rule: 2x * dx/dt)
print(f"Hessian: {result.hessian}")    # [[2.0]]
```

#### Taylor Evaluation

Jets can be evaluated at nearby points using their Taylor expansion:

```python
jet: Jet = function_to_jet(f, p)
dx: Array = jnp.array([0.1, -0.1])
approx: Array = jet(dx)  # f(p + dx) ≈ f(p) + grad·dx + 0.5*dx^T·H·dx
```

### Jacobians: Coordinate Maps

A `Jacobian` represents the derivatives of a coordinate transformation, storing:
- `value`: The Jacobian matrix ∂z^i/∂x^j
- `gradient`: Second derivatives ∂²z^i/∂x^j∂x^k
- `hessian`: Third derivatives ∂³z^i/∂x^j∂x^k∂x^l

```python
from local_coordinates.jacobian import Jacobian, function_to_jacobian

# Polar to Cartesian transformation
def polar_to_cartesian(rtheta: Array) -> Array:
    r, theta = rtheta[0], rtheta[1]
    return jnp.array([r * jnp.cos(theta), r * jnp.sin(theta)])

p_polar: Array = jnp.array([2.0, jnp.pi/4])
J: Jacobian = function_to_jacobian(polar_to_cartesian, p_polar)

print(f"Jacobian matrix:\n{J.value}")
# [[cos(θ), -r*sin(θ)],
#  [sin(θ),  r*cos(θ)]]
```

#### Jacobian Inversion and Composition

```python
# Get the inverse Jacobian (for inverse coordinate transformation)
J_inv: Jacobian = J.get_inverse()

# Compose Jacobians (chain rule)
from local_coordinates.jacobian import compose
J_composed: Jacobian = compose(J1, J2)  # J1 ∘ J2
```

### Basis Vectors

`BasisVectors` represents a set of tangent space basis vectors, always expressed in the standard Euclidean basis. The components are stored as a Jet to track derivatives.

```python
from local_coordinates.basis import BasisVectors, get_standard_basis, get_basis_transform

# Get the standard basis at a point
p: Array = jnp.array([1.0, 2.0])
standard_basis: BasisVectors = get_standard_basis(p)

# The standard basis has identity components
print(f"Basis components:\n{standard_basis.components.value}")
# [[1, 0],
#  [0, 1]]
```

#### Column-Vector Convention

The library uses the column-vector convention:
- `E[a, j]` = a-th component of j-th basis vector
- Columns are vectors: `E[:, j]` = basis vector E_j

#### Basis Transformations

```python
# Get transformation matrix between bases
# If V = V^j E_j = W^i B_i, then W = T @ V
T: Jet = get_basis_transform(from_basis, to_basis)
```

### Frames

A `Frame` is a set of basis vectors with support for computing Lie brackets between pairs:

```python
from local_coordinates.frame import Frame, basis_to_frame, get_lie_bracket_between_frame_pairs

# Convert BasisVectors to Frame
frame: Frame = basis_to_frame(basis)

# Compute Lie brackets [E_i, E_j] for all pairs
lie_brackets: Jet = get_lie_bracket_between_frame_pairs(frame)
```

For a coordinate basis (basis vectors that are partial derivatives), the Lie brackets vanish: [∂/∂x^i, ∂/∂x^j] = 0.

### Tangent Vectors

`TangentVector` represents a vector in a tangent space, with components expressed in a given basis:

```python
from local_coordinates.tangent import TangentVector, lie_bracket

# Create a tangent vector
X: TangentVector = TangentVector(
    p=p,
    components=Jet(value=jnp.array([1.0, 0.0]), gradient=..., hessian=...),
    basis=standard_basis
)

# Compute the Lie bracket [X, Y]
bracket: TangentVector = lie_bracket(X, Y)

# Convert to standard basis
X_std: TangentVector = X.to_standard_basis()
```

### Tensors

`Tensor` represents a general (k, l) tensor with k covariant and l contravariant indices:

```python
from local_coordinates.tensor import Tensor, TensorType, change_basis

# Create a (1, 1) mixed tensor (like a linear map)
tensor_type: TensorType = TensorType(k=1, l=1)
tensor: Tensor = Tensor(
    tensor_type=tensor_type,
    basis=basis,
    components=Jet(value=jnp.eye(2), gradient=..., hessian=...)
)

# Change to a different basis
tensor_new_basis: Tensor = change_basis(tensor, new_basis)
```

---

## Riemannian Geometry

### Riemannian Metrics

A `RiemannianMetric` is a (2, 0) tensor that defines an inner product on tangent spaces:

```python
from local_coordinates.metric import (
    RiemannianMetric,
    get_euclidean_metric,
    pullback_metric,
    raise_index,
    lower_index
)

# Get the Euclidean metric
p: Array = jnp.array([1.0, 2.0])
euclidean_metric: RiemannianMetric = get_euclidean_metric(p)

# Create a custom metric from components
def metric_components(x: Array) -> Array:
    # Example: metric that varies with position
    return jnp.array([
        [1 + x[0]**2, 0],
        [0, 1 + x[1]**2]
    ])

metric_jet: Jet = function_to_jet(metric_components, p)
basis: BasisVectors = get_standard_basis(p)
metric: RiemannianMetric = RiemannianMetric(basis=basis, components=metric_jet)

# Evaluate the metric on two tangent vectors: g(X, Y)
inner_product: Jet = metric(X, Y)
```

#### Index Manipulation

```python
# Raise an index using the metric inverse
tensor_raised: Tensor = raise_index(tensor, metric, index=1)  # 1-based indexing

# Lower an index using the metric
tensor_lowered: Tensor = lower_index(tensor, metric, index=2)
```

#### Pullback Metric

Given a map f: M → N and a metric g on N, compute the pullback metric f*g on M.

**Note**: The current implementation requires that the source and target spaces have compatible coordinate dimensions for automatic derivative propagation. For same-dimension coordinate transformations, this works seamlessly:

```python
from local_coordinates.jet import Jet
from local_coordinates.basis import get_standard_basis
from local_coordinates.metric import RiemannianMetric, pullback_metric

# Polar to Cartesian transformation (R^2 -> R^2)
def polar_to_cartesian(q: Array) -> Array:
    r, phi = q[0], q[1]
    return jnp.array([r * jnp.cos(phi), r * jnp.sin(phi)])

r_val, phi_val = 2.0, jnp.pi / 4.0
p_polar: Array = jnp.array([r_val, phi_val])
p_cart: Array = polar_to_cartesian(p_polar)

# Euclidean metric on R^2 (Cartesian coordinates)
basis_cart: BasisVectors = get_standard_basis(p_cart)
euclidean_2d: RiemannianMetric = RiemannianMetric(
    basis=basis_cart,
    components=Jet(
        value=jnp.eye(2),
        gradient=jnp.zeros((2, 2, 2)),
        hessian=jnp.zeros((2, 2, 2, 2))
    )
)

# Pullback to polar coordinates
polar_metric: RiemannianMetric = pullback_metric(p_polar, polar_to_cartesian, euclidean_2d)

# Result: diag(1, r^2) - the standard polar coordinate metric
print(polar_metric.components.value)
# [[1.0, 0.0],
#  [0.0, 4.0]]  # r^2 = 4 when r = 2
```

### Connections and Christoffel Symbols

A `Connection` stores the Christoffel symbols Γ^k_{ij} and provides covariant differentiation:

```python
from local_coordinates.connection import Connection, get_levi_civita_connection

# Get the Levi-Civita connection from a metric
connection: Connection = get_levi_civita_connection(metric)

# Access Christoffel symbols: Γ^k_{ij} = christoffel_symbols[i, j, k]
Gamma: Array = connection.christoffel_symbols.value

# Compute covariant derivative ∇_X Y
nabla_X_Y: TangentVector = connection.covariant_derivative(X, Y)
```

The Levi-Civita connection is the unique torsion-free, metric-compatible connection. It satisfies:
- **Torsion-free**: ∇_X Y - ∇_Y X = [X, Y]
- **Metric-compatible**: X(g(Y, Z)) = g(∇_X Y, Z) + g(Y, ∇_X Z)

### Curvature Tensors

The `RiemannCurvatureTensor` is a (3, 1) tensor measuring the curvature of the manifold:

```python
from local_coordinates.riemann import (
    RiemannCurvatureTensor,
    get_riemann_curvature_tensor,
    RicciTensor,
    get_ricci_tensor
)

# Compute the Riemann curvature tensor from a connection
riemann: RiemannCurvatureTensor = get_riemann_curvature_tensor(connection)

# Components: R_{ijk}^m = riemann.components.value[i, j, k, m]
R: Array = riemann.components.value

# Evaluate on tangent vectors: R(X, Y)Z
curvature_vector: TangentVector = riemann(X, Y, Z)

# Compute the Ricci tensor (contraction of Riemann)
ricci: RicciTensor = get_ricci_tensor(connection)
Ric: Array = ricci.components.value  # R_{ab} = R_{iab}^i
```

#### Riemann Tensor Symmetries

The Riemann tensor satisfies several symmetries:
- Skew symmetry: R_{ijkl} = -R_{jikl}
- Interchange symmetry: R_{ijkl} = R_{klij}
- First Bianchi identity: R_{ijkl} + R_{jkil} + R_{kijl} = 0

---

## Special Coordinate Systems

### Riemann Normal Coordinates

Riemann normal coordinates (RNC) are a special coordinate system centered at a point p where:
- The metric is the identity at p: g_{ij}(p) = δ_{ij}
- The Christoffel symbols vanish at p: Γ^k_{ij}(p) = 0
- The first derivatives of the metric vanish at p

```python
from local_coordinates.normal_coords import (
    get_rnc_jacobians,
    get_rnc_basis,
    to_riemann_normal_coordinates
)
from local_coordinates.jacobian import Jacobian

# Get the Jacobians for transforming to/from RNC
J_x_to_v, J_v_to_x = get_rnc_jacobians(metric)  # Both are Jacobian objects

# J_x_to_v: dv/dx (transforms TO RNC)
# J_v_to_x: dx/dv (transforms FROM RNC)

# Get the RNC basis vectors expressed in original coordinates
rnc_basis: BasisVectors = get_rnc_basis(metric)

# Transform geometric objects to RNC
metric_rnc: RiemannianMetric = to_riemann_normal_coordinates(metric)
connection_rnc: Connection = to_riemann_normal_coordinates(connection, metric)
tensor_rnc: Tensor = to_riemann_normal_coordinates(tensor, metric)
```

In RNC, the coordinate lines are geodesics emanating from the origin, making them ideal for local computations.

### Exponential and Logarithmic Maps

The exponential map exp_p: T_p M → M takes a tangent vector v at p and returns the endpoint of the geodesic starting at p with initial velocity v:

```python
from local_coordinates.exponential_map import (
    exponential_map,
    exponential_map_taylor,
    exponential_map_ode,
    logarithmic_map_taylor
)
from typing import Callable

# Create a tangent vector at p
v: TangentVector = TangentVector(p=p, components=v_jet, basis=standard_basis)

# Compute exp_p(v) using Taylor approximation (fast, local)
q: Array = exponential_map(metric, v, method="taylor")

# Or using ODE integration (slower, globally accurate)
def metric_fn(x: Array) -> RiemannianMetric:
    # Return the metric at point x
    return RiemannianMetric(basis=get_standard_basis(x), components=...)

q: Array = exponential_map_ode(v, metric_fn, num_steps=100)

# Get the full geodesic trajectory
ts, trajectory = exponential_map_ode(v, metric_fn, return_trajectory=True)
# ts: Array, trajectory: Array
```

The logarithmic map is the inverse:

```python
# Compute log_p(q) - the tangent vector that maps to q
v_components: Array = logarithmic_map_taylor(metric, q)
```

---

## Complete Examples

### Example 1: Computing Metric and Curvature

This example computes the Levi-Civita connection, Riemann curvature tensor, and Ricci tensor for a position-dependent metric:

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
# This represents a curved 2D space where the metric varies with position
def metric_components(x: Array) -> Array:
    return jnp.array([
        [1 + 0.1*x[0]**2, 0.0],
        [0.0, 1 + 0.1*x[1]**2]
    ])

# Point at which to compute geometric quantities
p: Array = jnp.array([1.0, 1.0])

# Create the metric
basis: BasisVectors = get_standard_basis(p)
metric_jet: Jet = function_to_jet(metric_components, p)
metric: RiemannianMetric = RiemannianMetric(basis=basis, components=metric_jet)

print("Metric components at p:")
print(metric.components.value)

# Compute the Levi-Civita connection
connection: Connection = get_levi_civita_connection(metric)
print("\nChristoffel symbols Γ^k_{ij}:")
print(connection.christoffel_symbols.value)

# Compute the Riemann curvature tensor
riemann: RiemannCurvatureTensor = get_riemann_curvature_tensor(connection)
R: Array = riemann.components.value
print(f"\nRiemann tensor shape: {R.shape}")

# Compute the Ricci tensor
ricci: RicciTensor = get_ricci_tensor(connection, R=riemann)
print("\nRicci tensor:")
print(ricci.components.value)

# Compute the Ricci scalar (scalar curvature)
g_inv: Array = jnp.linalg.inv(metric.components.value)
ricci_scalar: Array = jnp.einsum("ij,ij->", g_inv, ricci.components.value)
print(f"\nRicci scalar: {ricci_scalar}")
```

### Example 2: Coordinate Transformation Round-Trip

This example verifies that transforming to RNC and back recovers the original metric:

```python
import jax.numpy as jnp
from jax import Array
from local_coordinates.jet import Jet, function_to_jet
from local_coordinates.basis import BasisVectors, get_standard_basis
from local_coordinates.metric import RiemannianMetric
from local_coordinates.connection import Connection, get_levi_civita_connection
from local_coordinates.normal_coords import (
    get_rnc_jacobians,
    to_riemann_normal_coordinates
)

# Create a non-trivial metric
def metric_components(x: Array) -> Array:
    return jnp.array([
        [1 + 0.1*x[0]**2, 0.05*x[0]*x[1]],
        [0.05*x[0]*x[1], 1 + 0.1*x[1]**2]
    ])

p: Array = jnp.array([1.0, 1.0])
basis: BasisVectors = get_standard_basis(p)
metric_jet: Jet = function_to_jet(metric_components, p)
metric: RiemannianMetric = RiemannianMetric(basis=basis, components=metric_jet)

print("Original metric at p:")
print(metric.components.value)

# Transform to RNC
metric_rnc: RiemannianMetric = to_riemann_normal_coordinates(metric)

print("\nMetric in RNC (should be identity at origin):")
print(metric_rnc.components.value)

# Verify Christoffel symbols vanish
connection: Connection = get_levi_civita_connection(metric)
connection_rnc: Connection = to_riemann_normal_coordinates(connection, metric)

print("\nChristoffel symbols in RNC (should vanish at origin):")
print(jnp.max(jnp.abs(connection_rnc.christoffel_symbols.value)))
```

### Example 3: Geodesics via Exponential Map

This example computes geodesics on a curved surface:

```python
import jax
import jax.numpy as jnp
from jax import Array
from local_coordinates.jet import Jet, function_to_jet
from local_coordinates.basis import BasisVectors, get_standard_basis
from local_coordinates.metric import RiemannianMetric
from local_coordinates.tangent import TangentVector
from local_coordinates.exponential_map import exponential_map_ode

# Define a metric that varies with position (hyperbolic-like)
def make_metric(x: Array) -> RiemannianMetric:
    basis: BasisVectors = get_standard_basis(x)
    # Recompute with proper derivatives
    g_jet: Jet = function_to_jet(
        lambda y: (1.0 / (1.0 + 0.1 * jnp.sum(y**2))) * jnp.eye(2),
        x
    )
    return RiemannianMetric(basis=basis, components=g_jet)

# Starting point and initial velocity
p: Array = jnp.array([0.0, 0.0])
v_components: Array = jnp.array([1.0, 0.5])

# Create tangent vector
basis: BasisVectors = get_standard_basis(p)
v_jet: Jet = Jet(
    value=v_components,
    gradient=jnp.zeros((2, 2)),
    hessian=jnp.zeros((2, 2, 2))
)
v: TangentVector = TangentVector(p=p, components=v_jet, basis=basis)

# Compute geodesic trajectory
ts, trajectory = exponential_map_ode(
    v,
    metric_fn=make_metric,
    num_steps=50,
    return_trajectory=True
)
# ts: Array of time values, trajectory: Array of shape (num_steps, 2)

print("Geodesic trajectory:")
for i in range(0, len(ts), 10):
    print(f"  t={ts[i]:.2f}: ({trajectory[i, 0]:.4f}, {trajectory[i, 1]:.4f})")
```

---

## Summary

The `local_coordinates` library provides a comprehensive toolkit for differential geometry computations:

1. **Jets** enable automatic differentiation with second-order derivatives
2. **Basis vectors and frames** handle coordinate-aware computations
3. **Tensors** support arbitrary (k, l) tensors with proper transformation rules
4. **Metrics** define inner products and enable index manipulation
5. **Connections** provide covariant derivatives via Christoffel symbols
6. **Curvature tensors** measure the intrinsic curvature of manifolds
7. **Normal coordinates** simplify local computations where the metric is identity
8. **Exponential maps** compute geodesics and relate tangent spaces to the manifold

The library's design emphasizes type safety, coordinate awareness, and seamless integration with JAX's automatic differentiation capabilities.

---

## AI Assistant Skills

This library includes AI assistant skills that provide task-specific guidance when working with Cursor or other AI-powered development environments. Skills are structured instructions that help the AI understand how to use the library correctly.

### Available Skills

The following skills are available in `.cursor/skills/`:

| Skill | Description | Use When |
|-------|-------------|----------|
| **create-riemannian-metric** | Create `RiemannianMetric` objects from metric functions | Defining custom metrics, starting geometry computations |
| **compute-curvature** | Compute Levi-Civita connection, Christoffel symbols, Riemann tensor, Ricci tensor | Analyzing manifold curvature, checking flatness |
| **pullback-metric** | Compute pullback metrics under coordinate transformations | Changing coordinate systems (polar, spherical), studying embeddings |
| **riemann-normal-coordinates** | Transform objects to Riemann normal coordinates | Simplifying local computations, verifying geometric identities |
| **compute-geodesics** | Compute geodesics via exponential and logarithmic maps | Computing shortest paths, geodesic distances |
| **jet-differentiation** | Use Jets for second-order automatic differentiation | Working with Taylor expansions, propagating derivatives |

### How to Use Skills

Skills are automatically discovered by Cursor when you open this project. You can:

1. **Automatic invocation**: Simply describe what you want to do, and the AI will use the relevant skill
2. **Manual invocation**: Type `/` followed by the skill name in the chat (e.g., `/create-riemannian-metric`)

### Top-Level Library Skill

For a general introduction to the library, see `SKILL.md` in the project root. This provides:
- Library overview and capabilities
- Installation instructions
- Quick start example
- Links to all available skills

### Skill File Format

Each skill follows the Agent Skills standard with:
- YAML frontmatter containing `name` and `description`
- "When to Use" section with clear triggers
- Step-by-step instructions with code examples
- Common patterns and gotchas

You can also create your own skills by adding folders to `.cursor/skills/` following the same format.

---

## Known Limitations

The library has some known limitations to be aware of:

1. **Dimension mismatches in pullback_metric**: The `pullback_metric` function currently requires that all Jets involved have the same coordinate dimension. This means pullback computations work correctly for same-dimension mappings (e.g., polar to Cartesian in R^2), but mappings between different-dimensional spaces (e.g., R^2 → R^3 embeddings) require manual handling of the metric derivatives.

2. **Jet coordinate dimension consistency**: The `@jet_decorator` expects all Jet inputs to have consistent coordinate dimensions. When composing operations that involve Jets from different coordinate systems, ensure the coordinate dimensions match or use explicit Jacobian transformations.

3. **ODE solver for exponential map**: The `exponential_map_ode` function may require tuning of step sizes and tolerances for highly curved metrics. For local computations, `exponential_map_taylor` is faster and sufficient.
