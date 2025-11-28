Title: Riemann normal coordinates
Date: 2025-11-14
Category: Blog
Slug: riemann-normal-coordinates
hidden: true
Summary: Riemann normal coordinates

In this post we'll go over Riemann normal coordinates and how to construct them from a given Riemannian manifold.  Let $(M, g)$ be a Riemannian manifold and let $p \in M$ be a point.  Suppose that $(\frac{\partial}{\partial x^1}, \dots, \frac{\partial}{\partial x^n})$ is a coordinate frame around $p$ in which we write the components of the metric as
$$
\begin{align}
  g = g_{ij}dx^i \otimes dx^j
\end{align}
$$
Additionally, suppose that we have a Jet to represent the derivatives of the components of the metric in terms of these coordinates as follows:
$$
\begin{align}
  \frac{\partial g_{ij}}{\partial x^k} &= J_{ij;k} \\
  \frac{\partial^2 g_{ij}}{\partial x^k \partial x^l} &= H_{ij;kl}
\end{align}
$$
With these expressions for the metric, we are able to the majority of the geometric quantities we might care about, such as the Christoffel symbols, the Riemann curvature tensor, and the Ricci curvature tensor.  However, all of these objects can be computed only at $p$ and nowhere else.  This is a problem because we might want to use the higher order derivatives of the metric to compute these objects at other, nearby, points.  This is where Riemann normal coordinates come in.  They are a convenient coordinate system that plays well with Riemannian geometry and allows us to sort of abstract away the information about the components of the metric that is not directly related to curvature.

In this post we will derive Riemann normal coordinates, including the expressions for the Jet coefficients of the metric, and Taylor expansions of geometric quantities we might care about in terms of these coordinates.  Around $p \in \mathcal{M}$, we will denote Riemann coordinates by $(v^1, \dots, v^n)$ and the corresponding coordinate frame by $(\frac{\partial}{\partial v^1}, \dots, \frac{\partial}{\partial v^n})$.  In this basis, we will see that the metric takes the form $g = \delta_{ij} dv^i \otimes dv^j$, and the Jet coefficients will be given by:
$$
\boxed{
\begin{aligned}
  \frac{\partial g_{ij}}{\partial v^k} &= 0 \\
  \frac{\partial^2 g_{ij}}{\partial v^k \partial v^l} &= \frac{1}{3}R_{kilj} + \frac{1}{3}R_{likj} \\
  \frac{\partial^2 \log \det(g)}{\partial v^i \partial v^j} &= -\frac{2}{3}(Rc)_{ij}
\end{aligned}
}
$$
where $R_{ijkl}$ are the components of the Riemann curvature tensor written in the normal coordinate basis and $Rc$ is the Ricci curvature tensor.

Additionally, we can derive the Taylor expansions of the coordinate functions $x^i(v)$ in terms of the normal coordinate basis:
$$
\boxed{
\begin{aligned}
  \frac{\partial x^i}{\partial v^j}
  &= J^i_j \quad \text{(orthonormal frame components)}, \\
  \frac{\partial^2 x^i}{\partial v^j \partial v^k}
  &= -\bar{\Gamma}_{ab}^i J^a_j J^b_k, \\
  \frac{\partial^3 x^i}{\partial v^a \partial v^b \partial v^c}
  &= \text{Sym}_{abc}\left(-\frac{\partial \bar{\Gamma}^i_{jk}}{\partial x^m} J^m_a J^j_b J^k_c + 2\bar{\Gamma}^i_{jk} \bar{\Gamma}^j_{mn} J^m_a J^n_b J^k_c\right)
\end{aligned}
}
$$

# Riemann normal coordinates
Let $(M, g)$ be a Riemannian manifold and let $p \in M$ be a point.  Let $U \subset M$ be a normal neighborhood of $p$ and $(E_1, \dots, E_n)$ be an orthonormal frame on $U$.  For any $q \in U$, there exists $T \in T_p M$ such that $q = \exp_p(T)$ uniquely.  Since $T_p M$ is Euclidean space, we can construct a chart on $U$ that computes the components of $T$ given $q$:
$$
\begin{align}
  v(q) = E^{-1}\circ \log_p(q), \quad \text{where } E(v) = v^i E_i
\end{align}
$$
where $\log_p(q)$ is the inverse of the exponential map at $p$.  The coordinate functions $v(q)$ are called the *Riemann normal coordinates* of $q$ with respect to the frame $(E_1, \dots, E_n)$.  In this basis, $\frac{\partial}{\partial v^i} = E_i$, so from the fact that $(E_1, \dots, E_n)$ is an orthonormal frame, we have that
$$
\begin{align}
  g = \delta_{ij}dv^i \otimes dv^j
\end{align}
$$
The christoffel symbols take a particularly simple form in this basis.  Let $\gamma(t) = \exp_p(t v^i E_i)$ be a geodesic on $U$.  By the geodesic equation, we have that
$$
\begin{align}
  \ddot{\gamma}(t) + \Gamma_{ij}^k \dot{\gamma}(t)^i \dot{\gamma}(t)^j = 0
\end{align}
$$
and we can compute the time derivatives as follows:
$$
\begin{align}
  \dot{\gamma}(t) &= \frac{d}{dt}|_{t=0}\; \exp_p(t v^i E_i) \\
  &= d\left(\exp_p\right)_{0}\left(\frac{d}{dt}|_{t=0}\; t v^i E_i\right) \\
  &= d\left(\exp_p\right)_{0}\left(v^i E_i\right) \\
  &= v^i E_i \\
  \ddot{\gamma}(t) &= 0
\end{align}
$$
and so we have that $\Gamma_{ij}^k \dot{\gamma}(0)^i \dot{\gamma}(0)^j = 0$, which implies that $\Gamma_{ij}^k = 0$ in normal coordinates.

# Jacobi fields

A *Jacobi field* is a vector field along a geodesic that commutes with the geodesic flow.  Let $S(t)$ be a Jacobi field along a geodesic $\gamma(t) = \exp_p(T_t)$ where $T_t = t T_0$ for some $T_0 \in T_pM$.  Then we will define $S(t)$ as a vector field that satisfies
$$
\begin{align}
  [T(t), S(t)] = 0
\end{align}
$$
A useful intepretation of this equation is that $S(t)$ is a variation field of the geodesic.  If we suppose that $\gamma(t;\theta)$ is a family of geodesics parameterized by $\theta$, then we could interpret $S(t) = \frac{\partial \gamma(t;\theta)}{\partial \theta}$ as the variation field of the geodesic with respect to $\theta$.  The fact that $S(t)$ takes a geodesic to another geodesic imposes constraints on the derivatives of $S$.

Here are some useful properties of Jacobi fields:
1. $\nabla_S T = \nabla_T S$ because $\nabla$ is torsion free (so $\nabla_S T - \nabla_T S = [S, T]$) and $[T, S] = 0$ for Jacobi fields.
2. $\nabla_T^2 S = R(T, S)T$ because $R(T,S)T = \nabla_T \nabla_S T - \nabla_S \cancel{\nabla_T T} = \nabla_T \nabla_T S$.
3. $S(0)$ and $\nabla_T S(0)$ uniquely determine $S(t)$ for all $t$ because of (2).
4. $f(t) = g(T(t), S(t))$ is a linear function of $t$ because
$$
\begin{align}
  \frac{d^2}{dt^2} f(t) = g(\nabla_T^2 S(t), T(t)) = g(R(T, S)T, T) = R(T, S, T, T) = 0
\end{align}
$$
5. If $g(T(0), S(0)) = g(T(0), \nabla_T S(0)) = 0$, then $g(T(t), S(t)) = 0$ for all $t$ because of (4).

## Simple examples of Jacobi fields
From (3), we know that $S(t)$ is uniquely determined by $S(0)$ and $\nabla_T S(0)$.  Therefore, we can construct simple examples of Jacobi fields by choosing $S(0)$ and $\nabla_T S(0)$.  A simple choice of Jacobi field is $(S(0), \nabla_T S(0)) = (0, \frac{\partial}{\partial v^i})$ for some $i$.  This arises from the family of geodesics $\gamma(t, s) = t(E_0 + s E_i)$, where $T = \frac{\partial \gamma}{\partial t}$ and $S = \frac{\partial \gamma}{\partial s}$.  By construction, $[T, S] = 0$ because mixed partial derivatives commute.

In fact, this choice has a closed form solution for $S(t)$ given by
$$
\begin{align}
  S(t) = t \frac{\partial}{\partial v^i}
\end{align}
$$
We can verify the initial conditions at $t=0$.  First, $S(0) = 0 \cdot E_i = 0$.  Second, using the covariant derivative formula $(\nabla_T S)^k = T^j \partial_j S^k + \Gamma^k_{jl} T^j S^l$ and the fact that $\Gamma^k_{jl}(0) = 0$ in RNC:
$$
\begin{align}
  (\nabla_T S)^k\big|_{t=0} = T^j \partial_j (t \delta^k_i)\big|_{t=0} + 0 = \delta^j_0 \delta^k_i \frac{\partial t}{\partial v^j}\bigg|_{t=0} = \delta^k_i
\end{align}
$$
where we used that $t = v^0$ along the geodesic $\gamma(t) = t E_0$, so $\frac{\partial t}{\partial v^0} = 1$.  Thus $\nabla_T S(0) = E_i = \frac{\partial}{\partial v^i}$.

**Note:** The Jacobi equation $\nabla_T^2 S = R(T, S)T$ holds exactly for all $t$ — this is not an approximation.  However, the simpler formula "$\nabla_T S = E_i$" only holds at $t=0$; for $t \neq 0$ there are corrections from the Christoffel symbols.  This subtlety does not affect the Taylor expansion calculations below since they are all evaluated at $t=0$.  See the section "RNC basis vectors as geodesics and Jacobi fields" for a detailed discussion of why *constant* vector fields (as opposed to proper Jacobi fields) fail to satisfy the Jacobi equation.

# Metric taylor expansion
Now suppose that we choose the Jacobi fields $(S_1(t), \dots, S_n(t)) = (t \frac{\partial}{\partial v^1}, \dots, t \frac{\partial}{\partial v^n})$.  Consider the function $h(t) = g(S_i(t), S_j(t))$.  We will compute the 4th order Taylor approximation of this function with respect to $t$ to help us get the coefficients of the Taylor expansion of the metric at $t=0$:
$$
\begin{align}
  h_{ij}(t) = h_{ij}(0) + h_{ij}'(0)t + \frac{h_{ij}''(0)}{2}t^2 + \frac{h_{ij}'''(0)}{6}t^3 + \frac{h_{ij}^{(4)}(0)}{24}t^4 + O(t^5)
\end{align}
$$

$$
\begin{align}
  h_{ij}(0) = g(S_i(0), S_j(0)) = 0
\end{align}
$$

$$
\begin{align}
  h_{ij}'(0) &= \frac{d}{dt}|_{t=0} g(S_i(t), S_j(t)) \\
  &\underset{\to 0}{=} g(\nabla_T S_i, S_j) + g(S_i, \nabla_T S_j) \\
  &= 0
\end{align}
$$

$$
\begin{align}
  h_{ij}''(0) &\underset{\to 0}{=} 2g(\nabla_T^2 S_i, S_j) + 2g(\nabla_T S_i, \nabla_T S_j) \\
  &= 2g(\frac{\partial}{\partial v^i}, \frac{\partial}{\partial v^j}) \\
  &= 2\delta_{ij}
\end{align}
$$

$$
\begin{align}
  h_{ij}'''(0) &\underset{\to 0}{=} \underbrace{6g(\nabla_T^3 S_i, S_j)}_{=0} + 6g(\nabla_T^2 S_i, \nabla_T S_j) \\
  &= 6g(R(T, S_i)T, \nabla_T S_j) \\
  &= 0
\end{align}
$$

$$
\begin{align}
  h_{ij}^{(4)}(0) &\underset{\to 0}{=} \underbrace{2g(\nabla_T^4 S_i, S_j)}_{=0} + 8g(\nabla_T^3 S_i, \nabla_T S_j) + \underbrace{6g(\nabla_T^2 S_i, \nabla_T^2 S_j)}_{=0} \\
  &= 8g(\nabla_T R(T, S_i)T, \nabla_T S_j) \\
  &= 8g(R(T, \nabla_T S_i)T, \nabla_T S_j) \\
  &= 8R(T, \frac{\partial}{\partial v^i}, T, \frac{\partial}{\partial v^j})
\end{align}
$$

So we have that
$$
\begin{align}
  h_{ij}(t) = t^2 g_{ij}(t \frac{\partial}{\partial v^i}, t \frac{\partial}{\partial v^i}) = \delta_{ij}t^2 + \frac{1}{3}R(T, \frac{\partial}{\partial v^i}, T, \frac{\partial}{\partial v^j})t^4 + O(t^5)
\end{align}
$$
So dividing by $t^2$ and letting $v_t = (t \frac{\partial}{\partial v^1}, \dots, t \frac{\partial}{\partial v^n})$ we have that
$$
\begin{align}
  g_{ij}(tv) &= \delta_{ij} + \frac{1}{3}R(v^a \frac{\partial}{\partial v^a}, \frac{\partial}{\partial v^i}, v^b \frac{\partial}{\partial v^b}, \frac{\partial}{\partial v^j})t^2 + O(t^3) \\
  &= \delta_{ij} + \frac{1}{3}R(\frac{\partial}{\partial v^a}, \frac{\partial}{\partial v^i}, \frac{\partial}{\partial v^b}, \frac{\partial}{\partial v^j})(tv^a)(tv^b) + O(t^3)
\end{align}
$$
Evaluating at $t=0$ gives us the final result:
$$
\boxed{
  \begin{align}
    g_{ij}(v) = \delta_{ij} + \frac{1}{3}R_{kilj}(p)v^k v^l + O(|v|^3)
  \end{align}
}
$$

## Derivation of the metric coefficients
Differentiating with respect to $v^k$ gives us the following expression for the coefficients of the metric:
$$
\begin{align}
  \frac{\partial g_{ij}}{\partial v^a} &= \frac{1}{3}\left(R_{aikj} + R_{kiaj}\right)v^k + O(|v|^2) \\
  \frac{\partial^2 g_{ij}}{\partial v^a \partial v^b} &= \frac{1}{3}\left(R_{aibj} + R_{biaj}\right) + O(|v|)
\end{align}
$$
Evaluating at $v=0$ gives us:
$$
\boxed{
  \begin{align}
    \frac{\partial^2 g_{ij}}{\partial v^a \partial v^b}(p) &= \frac{1}{3}\left(R_{aibj}(p) + R_{biaj}(p)\right)
  \end{align}
}
$$
This also yields the expression for the Christoffel symbols:
$$
\boxed{
  \begin{align}
    \frac{\partial \Gamma_{ij}^k}{\partial v^l} &= \frac{1}{3}\left(R_{kijl} + R_{kjil}\right) \\
    &= \frac{1}{3}\left(\bar{R}_{abcd}\frac{\partial x^a}{\partial v^k}\frac{\partial x^b}{\partial v^i}\frac{\partial x^c}{\partial v^j}\frac{\partial x^d}{\partial v^l} + \bar{R}_{abcd}\frac{\partial x^a}{\partial v^k}\frac{\partial x^b}{\partial v^j}\frac{\partial x^c}{\partial v^i}\frac{\partial x^d}{\partial v^l}\right)
  \end{align}
}
$$

## Second derivative of the log determinant of the metric
A useful quantity that we will care about is the second derivative of the log determinant of the metric.  Its expression can be easily derived from the other expressions that we have satisfied:
$$
\begin{align}
  \frac{\partial^2 \log \det(g)}{\partial v^a \partial v^b} &= \frac{\partial}{\partial v^b}\left(\frac{\partial \log \det(g)}{\partial v^a}\right) \\
  &= \frac{\partial}{\partial v^b}\left(g^{ij} \frac{\partial g_{ij}}{\partial v^a}\right) \\
  &= \frac{\partial g^{ij}}{\partial v^b}\underbrace{\frac{\partial g_{ij}}{\partial v^a}}_{=0} + \underbrace{g^{ij}}_{\delta_{ij}} \frac{\partial^2 g_{ij}}{\partial v^b \partial v^a} \\
  &= \frac{1}{3}\left(R_{aibi} + R_{biai}\right) \\
  &= -\frac{2}{3}(Rc)_{ab}
\end{align}
$$
where $Rc$ is the Ricci curvature tensor.


# Taylor series expansion
Suppose we have a coordinate system $(x^1, \dots, x^n)$ on a Riemannian manifold for which we have defined geometric quantities in terms of.  For example, suppose that we have parameterized the components of the metric as $g = g_{ij}dx^i \otimes dx^j$ as well as higher order derivatives of the components $\frac{\partial g_{ij}}{\partial x^k}$ and $\frac{\partial^2 g_{ij}}{\partial x^k \partial x^l}$.  With these, we can compute the majority of the geometric quantities we might care about, such as the Christoffel symbols and the Riemann curvature tensor.  However, suppose we construct a normal neighborhood around a point $p$ and want to perform a change of coordinates to Riemann normal coordinates, $(v^1, \dots, v^n)$.  This would allow us to probe the manifold near $p$ in a geometrically meaningful way, as each coordinate $v^i$ would represent a position on the geodesic that is tangent to the $\frac{\partial}{\partial v^i}$ direction at $p$.  In other words, we would like to have access to the function $x(v) = \exp_p(v^i \frac{\partial}{\partial v^i})$.  We can approximate this function using a Taylor series expansion.
$$
\begin{align}
  x^i(v) = x^i + \sum_{j=1}^n \frac{\partial x^i}{\partial v^j} v^j + \frac{1}{2}\sum_{j,k=1}^n \frac{\partial^2 x^i}{\partial v^j \partial v^k} v^j v^k + \frac{1}{6}\sum_{j,k,l=1}^n \frac{\partial^3 x^i}{\partial v^j \partial v^k \partial v^l} v^j v^k v^l + O(v^4)
\end{align}
$$
The only thing that we need to know then are the coefficients of the Taylor expansion, which we can solve for in a straightforward manner.

First, $\frac{\partial x^i}{\partial v^j}$ is simply the components of the normal coordinate basis, written in the $x$ coordinate system:
$$
\boxed{
  \frac{\partial}{\partial v^j} = \frac{\partial x^i}{\partial v^j} \frac{\partial}{\partial x^i}
}
$$
Note that $\frac{\partial x^i}{\partial v^j}$ is an orthonormal matrix by construction of normal coordinates.

### Second order coefficients
Next, the second order coefficients can be derived using the fact that the Christoffel symbols are zero in normal coordinates, and by using the change of coordinates formula for Christoffel symbols.  Suppose $\Gamma$ are the Christoffel symbols written in the $v$ (which we know equal $0$) coordinate system and $\bar{\Gamma}$ are the Christoffel symbols written in the $x$ coordinate system.  Then we have that
$$
\begin{align}
  \Gamma_{kl}^i = \frac{\partial v^i}{\partial x^m}\bar{\Gamma}_{np}^m \frac{\partial x^n}{\partial v^k}\frac{\partial x^p}{\partial v^l} + \frac{\partial^2 x^m}{\partial v^k \partial v^l}\frac{\partial v^i}{\partial x^m}
\end{align}
$$
For the moment, we will not make the simplification that $\Gamma_{kl}^i = 0$ because the derivatives of $\Gamma_{kl}^i$ with respect to $x^m$ will be non-zero.  We will compute these derivatives later.

Solving for $\frac{\partial^2 x^i}{\partial v^j \partial v^k}$ gives us the expression
$$
\boxed{
\begin{align}
  \frac{\partial^2 x^i}{\partial v^j \partial v^k} = -\bar{\Gamma}_{ab}^i \frac{\partial x^a}{\partial v^j}\frac{\partial x^b}{\partial v^k} + \Gamma_{jk}^a \frac{\partial x^i}{\partial v^a}
\end{align}
}
$$

#### Alternate derivation via geodesic equation
We can also derive the second-order coefficients directly from the geodesic equation.  The geodesic $\gamma(t) = x(tv)$ satisfies:
$$
\begin{align}
  \ddot{\gamma}^i + \bar{\Gamma}^i_{jk}(\gamma) \dot{\gamma}^j \dot{\gamma}^k = 0
\end{align}
$$
At $t=0$, using the same chain rule argument as before:
- $\dot{\gamma}^i(0) = J^i_a v^a$
- $\ddot{\gamma}^i(0) = H^i_{ab} v^a v^b$ where $H^i_{ab} = \frac{\partial^2 x^i}{\partial v^a \partial v^b}\big|_{v=0}$

Substituting into the geodesic equation at $t=0$:
$$
\begin{align}
  H^i_{ab} v^a v^b + \bar{\Gamma}^i_{jk}(p) (J^j_a v^a)(J^k_b v^b) &= 0 \\
  H^i_{ab} v^a v^b &= -\bar{\Gamma}^i_{jk}(p) J^j_a J^k_b v^a v^b
\end{align}
$$
Since this holds for all vectors $v$, and both sides are already symmetric in $(a,b)$ (the RHS because $\bar{\Gamma}^i_{jk}$ is symmetric in $(j,k)$), we can directly read off:
$$
\begin{align}
  \frac{\partial^2 x^i}{\partial v^a \partial v^b} = -\bar{\Gamma}^i_{jk} J^j_a J^k_b
\end{align}
$$
Note that no explicit symmetrization is needed here because the Christoffel symbols are already symmetric in their lower indices.

### Third order coefficients
For the third order coefficients, we use the geodesic equation directly.  The geodesic $\gamma(t) = x(tv)$ satisfies:
$$
\begin{align}
  \ddot{\gamma}^i + \bar{\Gamma}^i_{jk}(\gamma) \dot{\gamma}^j \dot{\gamma}^k = 0
\end{align}
$$
Differentiating with respect to $t$ and evaluating at $t=0$ gives:
$$
\begin{align}
  \dddot{\gamma}^i + \frac{\partial \bar{\Gamma}^i_{jk}}{\partial x^m} \dot{\gamma}^m \dot{\gamma}^j \dot{\gamma}^k + 2\bar{\Gamma}^i_{jk} \ddot{\gamma}^j \dot{\gamma}^k = 0
\end{align}
$$
At $t=0$, we have $\gamma(0) = p$.  To compute the time derivatives, we use the chain rule.  Let $u(t) = tv$ so that $\gamma(t) = x(u(t))$.  Then $\frac{du^a}{dt} = v^a$, and:
$$
\begin{align}
  \dot{\gamma}^i &= \frac{\partial x^i}{\partial u^a} \frac{du^a}{dt} = \frac{\partial x^i}{\partial u^a} v^a \\
  \ddot{\gamma}^i &= \frac{d}{dt}\left(\frac{\partial x^i}{\partial u^a} v^a\right) = \frac{\partial^2 x^i}{\partial u^a \partial u^b} \frac{du^b}{dt} v^a = \frac{\partial^2 x^i}{\partial u^a \partial u^b} v^a v^b \\
  \dddot{\gamma}^i &= \frac{d}{dt}\left(\frac{\partial^2 x^i}{\partial u^a \partial u^b} v^a v^b\right) = \frac{\partial^3 x^i}{\partial u^a \partial u^b \partial u^c} v^a v^b v^c
\end{align}
$$
Each time derivative adds one partial derivative and one factor of $v$.  Evaluating at $t=0$ (where $u=0$) and using our formulas for the Taylor coefficients:
- $\dot{\gamma}^i(0) = J^i_a v^a$
- $\ddot{\gamma}^i(0) = -\bar{\Gamma}^i_{jk} J^j_a J^k_b v^a v^b$ (from the second-order formula)
- $\dddot{\gamma}^i(0) = T^i_{abc} v^a v^b v^c$ where $T^i_{abc} = \frac{\partial^3 x^i}{\partial v^a \partial v^b \partial v^c}\big|_{v=0}$

Substituting these into the differentiated geodesic equation:
$$
\begin{align}
  T^i_{abc} v^a v^b v^c = \underbrace{\left(-\frac{\partial \bar{\Gamma}^i_{jk}}{\partial x^m} J^m_a J^j_b J^k_c + 2\bar{\Gamma}^i_{jk} \bar{\Gamma}^j_{mn} J^m_a J^n_b J^k_c\right)}_{C^i_{abc}} v^a v^b v^c
\end{align}
$$
where $T^i_{abc} = \frac{\partial^3 x^i}{\partial v^a \partial v^b \partial v^c}$.  Now here's the key observation:

- **Left side**: $T^i_{abc}$ is symmetric in $(a,b,c)$ because partial derivatives commute.
- **Right side**: The coefficient $C^i_{abc}$ is **not** symmetric in $(a,b,c)$ as written.

However, when we contract $C^i_{abc}$ with $v^a v^b v^c$, only the **symmetric part** of $C$ contributes, because $v^a v^b v^c$ is itself symmetric.  This is because for any tensor $A_{abc}$:
$$
A_{abc} v^a v^b v^c = \text{Sym}(A)_{abc} v^a v^b v^c
$$
where $\text{Sym}(A)_{abc} = \frac{1}{6}\sum_{\sigma \in S_3} A_{\sigma(a)\sigma(b)\sigma(c)}$ averages over all 6 permutations.

Since the equation $T^i_{abc} v^a v^b v^c = C^i_{abc} v^a v^b v^c$ holds for **all** vectors $v$, we can conclude:
$$
\begin{align}
  \frac{\partial^3 x^i}{\partial v^a \partial v^b \partial v^c} = T^i_{abc} = \text{Sym}_{abc}(C^i_{abc}) = \text{Sym}_{abc}\left(-\frac{\partial \bar{\Gamma}^i_{jk}}{\partial x^m} J^m_a J^j_b J^k_c + 2\bar{\Gamma}^i_{jk} \bar{\Gamma}^j_{mn} J^m_a J^n_b J^k_c\right)
\end{align}
$$

**Note:** An alternative derivation via the coordinate transformation formula gives a formula with an explicit curvature term (from $\frac{\partial \Gamma^m_{jk}}{\partial v^l} = \frac{1}{3}(R^m_{jkl} + R^m_{kjl})$), but this formula is only valid for a specific derivative ordering.  The geodesic-derived formula above is manifestly symmetric after applying $\text{Sym}$.

So to summarize, we have that
$$
\boxed{
\begin{aligned}
  \frac{\partial x^i}{\partial v^j}
  &= J^i_j \quad \text{(orthonormal frame components)}, \\
  \frac{\partial^2 x^i}{\partial v^j \partial v^k}
  &= -\bar{\Gamma}_{ab}^i J^a_j J^b_k, \\
  \frac{\partial^3 x^i}{\partial v^a \partial v^b \partial v^c}
  &= \text{Sym}_{abc}\left(-\frac{\partial \bar{\Gamma}^i_{jk}}{\partial x^m} J^m_a J^j_b J^k_c + 2\bar{\Gamma}^i_{jk} \bar{\Gamma}^j_{mn} J^m_a J^n_b J^k_c\right)
\end{aligned}
}
$$

# RNC basis vectors as geodesics and Jacobi fields

In Riemann normal coordinates, the coordinate basis vectors $\frac{\partial}{\partial v^i}$ have a natural interpretation in terms of geodesics and Jacobi fields.  However, there is a subtle issue when trying to verify the Jacobi equation numerically using constant vector fields.

## Geodesics as coordinate lines
The geodesic from $p$ with initial velocity $v = v^i E_i$ is simply the straight line $\gamma(t) = tv$ in v-coordinates.  The tangent vector to this geodesic is $T = v^i \frac{\partial}{\partial v^i}$, which is constant along the geodesic.  At the origin, $\nabla_T T = 0$ because $\Gamma^k_{ij}(0) = 0$.

## The naive expectation (what I expected)
Given the nice properties of RNC, one might expect to be able to verify the Jacobi equation $\nabla_T^2 S = R(T, S)T$ using constant coordinate basis vectors $T = E_0 = \frac{\partial}{\partial v^0}$ and $S = E_1 = \frac{\partial}{\partial v^1}$.  The reasoning would be:
1. $[T, S] = 0$ because coordinate basis vectors commute
2. $\nabla_T T = 0$ at the origin because $\Gamma^k_{ij}(p) = 0$
3. Therefore $\nabla_T^2 S = \nabla_T \nabla_T S = \nabla_T \nabla_S T$ (using $[T,S] = 0$ and torsion-free)

The Jacobi equation is derived from the curvature identity:
$$
\begin{align}
  R(T, S)T = \nabla_T \nabla_S T - \nabla_S \nabla_T T
\end{align}
$$
If $\nabla_T T = 0$ everywhere (i.e., $T$ is geodesic along the entire variation), then $\nabla_S \nabla_T T = 0$ and we get the Jacobi equation $\nabla_T^2 S = R(T, S)T$.

## Why it fails: $\nabla_S \nabla_T T \neq 0$ for constant $S$
The issue is that $\nabla_T T = 0$ holds **only at the origin**, not everywhere.  Let's compute $\nabla_S \nabla_T T$ explicitly at the origin.  In components:
$$
\begin{align}
  (\nabla_T T)^k = T^i \partial_i T^k + \Gamma^k_{ij} T^i T^j
\end{align}
$$
Since $T = E_0$ is constant ($T^k = \delta^k_0$), the first term vanishes.  So $(\nabla_T T)^k = \Gamma^k_{00}$.  Now we compute $\nabla_S (\nabla_T T)$:
$$
\begin{align}
  (\nabla_S \nabla_T T)^k &= S^m \partial_m (\nabla_T T)^k + \Gamma^k_{mj} S^m (\nabla_T T)^j \\
  &= S^m \partial_m \Gamma^k_{00} + \Gamma^k_{mj} S^m \Gamma^j_{00}
\end{align}
$$
At the origin, $\Gamma^j_{00}(p) = 0$, so the second term vanishes.  With $S = E_1$ (so $S^m = \delta^m_1$):
$$
\begin{align}
  (\nabla_S \nabla_T T)^k \big|_p = \partial_1 \Gamma^k_{00} \big|_p = \frac{\partial \Gamma^k_{00}}{\partial v^1}\bigg|_{v=0}
\end{align}
$$
Using our formula for the Christoffel gradient in RNC:
$$
\begin{align}
  \frac{\partial \Gamma^k_{ij}}{\partial v^l}\bigg|_{v=0} = \frac{1}{3}\left(R^k_{ijl} + R^k_{jil}\right)
\end{align}
$$
we get:
$$
\begin{align}
  (\nabla_S \nabla_T T)^k \big|_p = \frac{1}{3}\left(R^k_{001} + R^k_{001}\right) = \frac{2}{3} R^k_{001}
\end{align}
$$
This is **not zero** in curved space!  The problem is that $T = E_0$ is only geodesic at the origin; nearby parallel curves in the $E_1$ direction are not geodesics.

## The consequence: a factor of 1/3
From the curvature identity $R(T, S)T = \nabla_T \nabla_S T - \nabla_S \nabla_T T$, we have:
$$
\begin{align}
  \nabla_T \nabla_S T = R(T, S)T + \nabla_S \nabla_T T
\end{align}
$$
At the origin with constant $T = E_0$ and $S = E_1$:
$$
\begin{align}
  \nabla_T^2 S &= \nabla_T \nabla_S T \quad \text{(using } [T,S] = 0 \text{)} \\
  &= R(T, S)T + \nabla_S \nabla_T T \\
  &= R^k_{010} + \frac{2}{3} R^k_{001} \\
  &= -R^k_{001} + \frac{2}{3} R^k_{001} \quad \text{(antisymmetry)} \\
  &= -\frac{1}{3} R^k_{001}
\end{align}
$$
Meanwhile, $R(T, S)T = R^k_{010} = -R^k_{001}$.  So we get:
$$
\begin{align}
  \nabla_T^2 S = \frac{1}{3} R(T, S)T \neq R(T, S)T
\end{align}
$$
The factor of $\frac{1}{3}$ comes from the $\frac{2}{3}$ contribution of $\nabla_S \nabla_T T$.

## The fix: proper Jacobi fields with $S(0) = 0$
The Jacobi equation derivation assumes that $T$ is geodesic along the **entire variation**, which means $\nabla_S \nabla_T T = 0$.  For constant $S$, this fails because we're probing $\nabla_T T$ at nearby points where it's no longer zero.

The solution is to use a **proper Jacobi field** $S(t)$ that vanishes at the origin: $S(0) = 0$.  Consider a family of geodesics through the origin parameterized by $\epsilon$:
$$
\begin{align}
  \gamma(t, \epsilon) = t(E_0 + \epsilon E_1)
\end{align}
$$
The variation vector $S = \frac{\partial \gamma}{\partial \epsilon}$ is a Jacobi field along the central geodesic $\gamma(t, 0) = t E_0$.  At $t = 0$:
$$
\begin{align}
  S(0) = 0, \quad (\nabla_T S)(0) = E_1
\end{align}
$$
In v-coordinates, this Jacobi field has the simple form $S(t) = t \cdot E_1$, so along the geodesic $\gamma(t) = t E_0$:
$$
\begin{align}
  S^k(v) = v^0 \cdot \delta^k_1
\end{align}
$$
with gradient $\frac{\partial S^k}{\partial v^m} = \delta^k_1 \delta^m_0$.

Now at the origin, $S(0) = 0$, so:
$$
\begin{align}
  (\nabla_S \nabla_T T)^k \big|_p = S^m(0) \cdot \partial_m \Gamma^k_{00} = 0 \cdot \partial_m \Gamma^k_{00} = 0
\end{align}
$$
The Jacobi equation becomes:
$$
\begin{align}
  \nabla_T^2 S(0) = R(T(0), S(0))T(0) = R(T, 0)T = 0
\end{align}
$$
This is trivially satisfied because both sides are zero!  The non-trivial content of the Jacobi equation is encoded in the **gradient** of $S$, which determines how the Jacobi field evolves away from the origin.

## Summary
| Vector field | $S(0)$ | $\nabla_S \nabla_T T$ | Jacobi equation |
|--------------|--------|------------------------|-----------------|
| Constant $S = E_1$ | $E_1$ | $\frac{2}{3} R^k_{001} \neq 0$ | $\nabla_T^2 S = \frac{1}{3} R(T,S)T$ ❌ |
| Proper $S(t) = t E_1$ | $0$ | $0$ | $\nabla_T^2 S = R(T,S)T = 0$ ✓ |

The lesson: when verifying the Jacobi equation numerically, one must use a properly constructed Jacobi field with $S(0) = 0$, not a constant vector field.
