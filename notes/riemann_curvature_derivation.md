# Derivation of the Riemann Curvature Tensor Components

The Riemann curvature tensor is defined by its action on three vector fields $E_i, E_j, E_k$:
$$
R(E_i, E_j)E_k = \nabla_{E_i} \nabla_{E_j} E_k - \nabla_{E_j} \nabla_{E_i} E_k - \nabla_{[E_i, E_j]} E_k
$$
We can find the components of the tensor in the basis $\{E_m\}$ by expanding this definition using the definition of the connection and the Lie bracket.

First, let's write the covariant derivatives and Lie bracket in terms of their components in the local frame:
- $\nabla_{E_j} E_k = \Gamma^l_{jk} E_l$
- $[E_i, E_j] = c^l_{ij} E_l$

Now, we can substitute these into the definition. Let's evaluate the first term, applying the product rule for connections:
$$
\begin{align}
\nabla_{E_i} \nabla_{E_j} E_k &= \nabla_{E_i} (\Gamma^l_{jk} E_l) \\
&= (\nabla_{E_i} \Gamma^l_{jk}) E_l + \Gamma^l_{jk} (\nabla_{E_i} E_l) \\
&= E_i(\Gamma^l_{jk}) E_l + \Gamma^l_{jk} \Gamma^m_{il} E_m
\end{align}
$$
Here, $E_i(\Gamma^l_{jk})$ is the directional derivative of the Christoffel symbol component (which is a scalar function) along the basis vector $E_i$.

By swapping the indices $i$ and $j$, we get the second term:
$$
\nabla_{E_j} \nabla_{E_i} E_k = E_j(\Gamma^l_{ik}) E_l + \Gamma^l_{ik} \Gamma^m_{jl} E_m
$$

The third term is:
$$
\nabla_{[E_i, E_j]} E_k = \nabla_{c^l_{ij}E_l} E_k = c^l_{ij} \nabla_{E_l} E_k = c^l_{ij} \Gamma^m_{lk} E_m
$$

Now, we combine all the terms. We must be careful with the dummy indices of summation. Let's relabel the index in the last two terms from $l$ to $m$ to match the final expression in the image.
$$
\begin{align}
R(E_i, E_j)E_k &= \left( E_i(\Gamma^m_{jk}) E_m + \Gamma^l_{jk} \Gamma^m_{il} E_m \right) - \left( E_j(\Gamma^m_{ik}) E_m + \Gamma^l_{ik} \Gamma^m_{jl} E_m \right) - \left( c^l_{ij} \Gamma^m_{lk} E_m \right) \\
&= \left( E_i(\Gamma^m_{jk}) - E_j(\Gamma^m_{ik}) + \Gamma^l_{jk}\Gamma^m_{il} - \Gamma^l_{ik}\Gamma^m_{jl} - c^l_{ij}\Gamma^m_{lk} \right) E_m
\end{align}
$$

The components of the resulting vector are defined as the components of the Riemann tensor, ${R_{ijk}}^m$. The standard convention for the component indices is that $R(E_i, E_j)E_k = {R_{ijk}}^m E_m$. Therefore, by directly matching the terms in our derived expression, we can identify the components.  This gives the final formula for the components of the Riemann curvature tensor in an arbitrary basis:
$$
{R_{ijk}}^m = E_i(\Gamma^m_{jk}) - E_j(\Gamma^m_{ik}) + \Gamma^l_{jk}\Gamma^m_{il} - \Gamma^l_{ik}\Gamma^m_{jl} - c^l_{ij}\Gamma^m_{lk}
$$
