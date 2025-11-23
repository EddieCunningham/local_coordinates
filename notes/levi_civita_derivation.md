# Levi-Civita connection derivation

The Levi-Civita connection is a connection is the torsion free connection that is compatible with a Riemannian metric.  Let $(E_1, \ldots, E_n)$ be a local frame on the manifold.  We can derive the components of the connection using the Kozul formula, which says that
$$
\begin{align}
  g(\nabla_{E_i} E_j, E_k) = \frac{1}{2}\left(E_i(g(E_j, E_k)) + E_j(g(E_i, E_k)) - E_k(g(E_i, E_j)) + g([E_i, E_j], E_k) - g([E_i, E_k], E_j)- g([E_j, E_k], E_i)\right)
\end{align}
$$
Plugging in coordinate expressions for the terms, we get
$$
\begin{align}
  g(\Gamma_{ij}^l E_l, E_k) = \frac{1}{2}\left(E_i(g_{jk}) + E_j(g_{ik}) - E_k(g_{ij}) + g(c_{ij}^l E_l, E_k) - g(c_{ik}^l E_l, E_j)- g(c_{jk}^l E_l, E_i)\right) \\
  \implies \Gamma_{ij}^l g_{lk} = \frac{1}{2}\left(E_i(g_{jk}) + E_j(g_{ik}) - E_k(g_{ij}) + c_{ij}^l g_{lk} - c_{ik}^l g_{lj} - c_{jk}^l g_{li}\right)
\end{align}
$$
Multiplying both sides by the inverse of the metric, $g^{km}$, we are left with the components of the connection as
$$
\begin{align}
  \Gamma_{ij}^m = \frac{1}{2}\left(E_i(g_{jk})g^{km} + E_j(g_{ik})g^{km} - E_k(g_{ij})g^{km} + c_{ij}^m - c_{ik}^l g_{lj}g^{km} - c_{jk}^l g_{li}g^{km}\right)
\end{align}
$$

# Levi-Civita connection change of basis
Let $(\epsilon^1, \ldots, \epsilon^n)$ be the co-basis associated with the frame $(E_1, \ldots, E_n)$.  Now, consider another basis $(\tilde{E}_1, \ldots, \tilde{E}_n)$ and the corresponding dual coframe $(\tilde{\epsilon}^1, \ldots, \tilde{\epsilon}^n)$ that is related to the original basis by a linear transformation $T = {\hat{T}}^{-1}$:
$$
\tilde{E}_i = T^j_i E_j
$$
$$
\tilde{\epsilon}^i = \hat{T}^i_j \epsilon^j
$$
To derive how the christoffel symbols transform, we can simply see how the connection 1-forms transform.
$$
\begin{align}
  d \epsilon^k = d(T^k_i \tilde{\epsilon}^i) = dT^k_i \wedge \tilde{\epsilon}^i + T^k_i d\tilde{\epsilon}^i = -\tilde{\epsilon}^j \wedge dT^k_j + T_i^k \tilde{\epsilon}^j \wedge \tilde{\omega}_j^i
\end{align}
$$
and also
$$
\begin{align}
  d \epsilon^k = \epsilon^i \wedge \omega_i^k = T^i_j \tilde{\epsilon}^j \wedge \omega_i^k
\end{align}
$$
Since both expressions start with $\tilde{\epsilon}^i \wedge \ldots$, we can equate the two expressions to relate $\omega$ and $\tilde{\omega}$ as
$$
\begin{align}
  T_i^k \tilde{\omega}_j^i = T_j^i \omega_i^k + dT_j^k
\end{align}
$$
which can be rearranged to give
$$
\begin{align}
  \tilde{\omega}_j^i = T_j^l \omega_l^k \hat{T}_k^i + dT_j^k \hat{T}_k^i
\end{align}
$$

### Putting it all together
Finally, to get back to the Christoffel symbols in the new basis, we can apply the one forms to a basis vector because $\tilde{\Gamma}^k_{ij} = \tilde{\omega}_j^k(\tilde{E}_i)$.  So, renaming indices, we have
$$
\begin{align}
  \tilde{\omega}_j^k(\tilde{E}_i) &= T_j^l \omega_l^m(\tilde{E}_i) \hat{T}_m^k + dT_j^m(\tilde{E}_i)\hat{T}_m^k \\
  &= T_j^l \omega_l^m(\hat{T}_i^aE_a) \hat{T}_m^k + \tilde{E}_i(T_j^m)\hat{T}_m^k \\
  &= T_j^l \hat{T}_i^a\omega_l^m(E_a) \hat{T}_m^k + \tilde{E}_i(T_j^m)\hat{T}_m^k \\
\end{align}
$$
Finally, this gives us the transformation law for the Christoffel symbols as
$$
\begin{align}
  \tilde{\Gamma}^k_{ij} = T_j^l \hat{T}_i^a \hat{T}_m^k \Gamma^m_{al} + \tilde{E}_i(T_j^m)\hat{T}_m^k
\end{align}
$$
