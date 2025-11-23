# Normal coordinates
Here we'll go over how to construct normal coordinates on a Riemannian manifold.  We will show precisely how to construct a `BasisVectors` object that represents the normal coordinates from a `RiemannianMetric` object.  This means that we will need to derive the components of the normal coordinates directly.

Here is the strategy that we will use to construct the normal coordinates.  Let $(\mathcal{M}, g)$ be an n-dimensional Riemannian manifold and let $p \in \mathcal{M}$.  We will want to define a function $\gamma(s_1,\dots, s_n): \mathcal{M} \to \mathbb{R}^n$ such that $\gamma(p) = 0$ that satisfies the following properties:
1) The basis vectors $V_i = \frac{\partial \gamma}{\partial s_i}$ are orthonormal, $\langle V_i, V_j \rangle_g = \delta_{ij}$.
2) With all $s_{j\neq i}$ held constant, $\gamma(s_i)$ is a geodesic in the direction of $V_i$.

The technical challenge associated with this is that we need to determine the `Jet` representation of the components of the basis vectors $V_i$.  It will turn out that if we specify the values of the components of $(V_1, \dots, V_n)$ at $p$, then using the Riemannian metric tensor, we will be able to determine the gradient and Hessian of the components of each $V_i$, which will allow us to construct the `BasisVectors` object associated with the normal coordinates at $p$.

## Basic properties
For the derivation, we will need to remember some basic properties.  First, the geodesic equation tells us that
$$
\begin{align}
  \nabla_{V_i} V_i = 0, \quad i=1,\dots,n.
\end{align}
$$
Also because each $V_i$ is a coordinate vector, we have that
$$
\begin{align}
  \nabla_{V_i} V_j = \Gamma_{ij}^k V_k = \Gamma_{ji}^k V_k = \nabla_{V_j} V_i
\end{align}
$$

By definition of the Riemann curvature endomorphism, we have that
$$
\begin{align}
  \nabla_{V_i} \nabla_{V_j} V_k - \nabla_{V_j} \nabla_{V_i} V_k = R(V_i, V_j)V_k
\end{align}
$$
which, combined with the geodesic equation condition and symmetry lemma yields:
$$
\begin{align}
  R(V_i, V_j)V_i &= \nabla_{V_i} \nabla_{V_j} V_i - \nabla_{V_j} \underbrace{\nabla_{V_i} V_i}_{0} \\
  &= \nabla_{V_i}^2 V_j
\end{align}
$$
Similarly,
$$
\begin{align}
  R(V_i, V_j)V_j &= \nabla_{V_i} \underbrace{\nabla_{V_j} V_j}_{0} - \nabla_{V_j} \nabla_{V_i} V_j \\
  &= -\nabla_{V_j}^2 V_i
\end{align}
$$

## Exponential map
Let $(E_1, \dots, E_n)$ be a basis of tangent vectors at $T_p\mathcal{M}$ and let $V_1 = V_1^k E_k$ be a unit length vector at $T_p\mathcal{M}$.  In a small enough neighborhood of $p$, the exponential map is a diffeomorphism defined by $\exp_p: \mathbb{R} \to \mathcal{M}$ such that $\exp_p(t) = \gamma(t)$ where $\gamma(t)$ is the geodesic that starts at $p$ in the direction of $V_1$ and has length $t$.  Note that $\gamma(0) = p$ and $\dot{\gamma}(0) = V_1$.


This statement is a bit abstract, but actually tells us what the second derivative information of the components of $V_1$ are.

The geodesic equation is given by
$$
\begin{align}
  \ddot{\gamma}(t) = -\Gamma_{ij}^k(p + t(V_1^i E_i + V_1^j E_j)) V_1^i V_1^j
\end{align}
$$