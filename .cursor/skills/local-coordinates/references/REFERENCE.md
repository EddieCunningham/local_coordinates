# local_coordinates Reference

This skill provides an overview of the local_coordinates library for differential geometry computations.

## Primary Documentation

For comprehensive documentation with detailed examples, see:

- **[TUTORIAL.md](../../../TUTORIAL.md)** - Full tutorial with examples for all library features
- **[README.md](../../../README.md)** - Project overview and quick start guide

## Related Skills

For task-specific guidance, see the individual skills in `.cursor/skills/`:

| Skill | Use Case |
|-------|----------|
| `create-riemannian-metric` | Creating RiemannianMetric objects from metric functions |
| `compute-curvature` | Computing Levi-Civita connection, Riemann tensor, Ricci tensor |
| `pullback-metric` | Computing pullback metrics under coordinate transformations |
| `riemann-normal-coordinates` | Transforming objects to Riemann normal coordinates |
| `compute-geodesics` | Computing geodesics via exponential/logarithmic maps |
| `jet-differentiation` | Using Jets for second-order automatic differentiation |

## Test Examples

The `tests/` directory contains extensive examples demonstrating library usage:

- `tests/test_riemann_curvature_tensor.py` - Curvature computation examples
- `tests/test_pullback_metric_basis.py` - Pullback metric examples
- `tests/test_geodesic_derivation.py` - Geodesic computation examples
