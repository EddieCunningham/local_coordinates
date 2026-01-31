# Curvature Mathematical Reference

## Levi-Civita Connection

The **Levi-Civita connection** is the unique torsion-free connection that is compatible with a Riemannian metric.

### Koszul Formula

Given a local frame (E_1, ..., E_n), the connection is determined by the Koszul formula:

```
g(nabla_{E_i} E_j, E_k) = (1/2) [ E_i(g_{jk}) + E_j(g_{ik}) - E_k(g_{ij}) 
                                + g([E_i,E_j], E_k) - g([E_i,E_k], E_j) - g([E_j,E_k], E_i) ]
```

### Christoffel Symbols

The Christoffel symbols Gamma^m_{ij} are the components of the connection in a local frame. The formula is:

```
Gamma^m_{ij} = (1/2) [ E_i(g_{jk}) g^{km} + E_j(g_{ik}) g^{km} - E_k(g_{ij}) g^{km} 
                     + c^m_{ij} - c^l_{ik} g_{lj} g^{km} - c^l_{jk} g_{li} g^{km} ]
```

where:
- g_{ij} are the metric components
- g^{ij} is the inverse metric
- c^l_{ij} are the structure constants: [E_i, E_j] = c^l_{ij} E_l

**For a coordinate basis** (where [E_i, E_j] = 0), this simplifies to:

```
Gamma^k_{ij} = (1/2) g^{kl} (d_i g_{jl} + d_j g_{il} - d_l g_{ij})
```

### Key Properties

1. **Torsion-free**: nabla_X Y - nabla_Y X = [X, Y]
2. **Metric-compatible**: X(g(Y, Z)) = g(nabla_X Y, Z) + g(Y, nabla_X Z)
3. **Symmetry in coordinate basis**: Gamma^k_{ij} = Gamma^k_{ji}

## Riemann Curvature Tensor

The **Riemann curvature tensor** measures the failure of parallel transport to commute.

### Definition

```
R(E_i, E_j)E_k = nabla_{E_i} nabla_{E_j} E_k - nabla_{E_j} nabla_{E_i} E_k - nabla_{[E_i,E_j]} E_k
```

### Component Formula

The components R^m_{ijk} are defined by R(E_i, E_j)E_k = R^m_{ijk} E_m:

```
R^m_{ijk} = E_i(Gamma^m_{jk}) - E_j(Gamma^m_{ik}) + Gamma^l_{jk} Gamma^m_{il} - Gamma^l_{ik} Gamma^m_{jl} - c^l_{ij} Gamma^m_{lk}
```

### Derivation

Starting from the definition and using the product rule:

1. First term: nabla_{E_i} nabla_{E_j} E_k = E_i(Gamma^l_{jk}) E_l + Gamma^l_{jk} Gamma^m_{il} E_m
2. Second term: nabla_{E_j} nabla_{E_i} E_k = E_j(Gamma^l_{ik}) E_l + Gamma^l_{ik} Gamma^m_{jl} E_m
3. Third term: nabla_{[E_i,E_j]} E_k = c^l_{ij} Gamma^m_{lk} E_m

Combining gives the component formula above.

### Riemann Tensor Symmetries

The Riemann tensor (with all indices lowered: R_{ijkl} = g_{lm} R^m_{ijk}) satisfies:

1. **Skew symmetry in first pair**: R_{ijkl} = -R_{jikl}
2. **Skew symmetry in second pair**: R_{ijkl} = -R_{ijlk}
3. **Interchange symmetry**: R_{ijkl} = R_{klij}
4. **First Bianchi identity**: R_{ijkl} + R_{jkil} + R_{kijl} = 0

## Ricci Tensor

The **Ricci tensor** is the contraction of the Riemann tensor:

```
R_{ab} = R^i_{aib}
```

Properties:
- Symmetric: R_{ab} = R_{ba}
- In dimension 2: R_{ab} = (R/2) g_{ab} where R is the scalar curvature

## Scalar Curvature

The **scalar curvature** (Ricci scalar) is:

```
R = g^{ab} R_{ab}
```

This is a coordinate-invariant scalar that characterizes the local curvature.

## Key Invariants

1. **Flat space**: R^m_{ijk} = 0 for all indices (Euclidean metric)
2. **Ricci identity**: Symmetric Ricci tensor R_{ab} = R_{ba}
3. **Kretschmann scalar**: K = R_{abcd} R^{abcd} = 0 iff flat
4. **Scalar curvature invariance**: R = g^{ab} R_{ab} is basis-independent

## Implementation Notes

- Index convention: Gamma[i, j, k] = Gamma^k_{ij}
- Riemann convention: R[i, j, k, m] = R^m_{ijk}
- Use `get_levi_civita_connection(metric)` to compute connection
- Use `get_riemann_curvature_tensor(connection)` for Riemann tensor
- Use `get_ricci_tensor(connection)` for Ricci tensor
