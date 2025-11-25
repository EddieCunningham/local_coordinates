# Detailed notes on coordinate changes
In this code base, we store geometric objects using `Jet` objects, whose gradient and Hessian are calculated with respect to a local coordinate system.  When we want to change coordinates, in addition to performing the usual coordinate change on the value of the geometric object, we also need to perform the coordinate change on the gradient and Hessian.  In this note, we will show how to do this for the different objects that we use in this code base.  We will assume that we are working in a local coordinate system $(x^1, \dots, x^n)$ and we want to change to a new coordinate system $(z^1, \dots, z^n)$.

# Jets
A `Jet` object is a tuple of the value of the geometric object, its gradient, and its Hessian.  When we want to change coordinates, we need to perform the coordinate change on the value, gradient, and Hessian.  Let $F$ be the value of the geometric object.  The Jet of $F$ in the $x$-coordinate system is given by
$$
J[F]_x = \left(F, \frac{\partial F}{\partial x^i}, \frac{\partial^2 F}{\partial x^i \partial x^j}\right)
$$
We want to compute the Jet of $F$ in the $z$-coordinate system, which is given by
$$
J[F]_z = \left(F, \frac{\partial F}{\partial z^i}, \frac{\partial^2 F}{\partial z^i \partial z^j}\right)
$$
The gradient in the new coordinate system can be computed using the chain rule. Let $J_i^a = \frac{\partial x^a}{\partial z^i}$ be the Jacobian of the inverse map $x(z)$. Then:
$$
\frac{\partial F}{\partial z^i} = \frac{\partial x^a}{\partial z^i} \frac{\partial F}{\partial x^a} = J_i^a \frac{\partial F}{\partial x^a}
$$
The Hessian is a bit more involved.  Starting from the chain rule, we have
$$
\begin{align}
\frac{\partial^2 F}{\partial z^i \partial z^j} &= \frac{\partial}{\partial z^j}\left(\frac{\partial x^a}{\partial z^i} \frac{\partial F}{\partial x^a}\right) \\
&= \frac{\partial^2 x^a}{\partial z^j \partial z^i} \frac{\partial F}{\partial x^a} + \frac{\partial x^a}{\partial z^i} \frac{\partial^2 F}{\partial x^a \partial x^b} \frac{\partial x^b}{\partial z^j}
\end{align}
$$
Let $H_{ij}^a = \frac{\partial^2 x^a}{\partial z^i \partial z^j}$ be the Hessian of the inverse map.  To compute $H_{ij}^a$, we can use the fact that $\frac{\partial x}{\partial z} = \left(\frac{\partial z}{\partial x}\right)^{-1}$.  Differentiating this identity, we get
$$
\begin{align}
  H_{ij}^a = \frac{\partial^2 x^a}{\partial z^i \partial z^j} &= \frac{\partial}{\partial z^j}\left(\frac{\partial x^a}{\partial z^i}\right) \\
  &= \frac{\partial x^b}{\partial z^j}\frac{\partial}{\partial x^b}\left(\frac{\partial x^a}{\partial z^i}\right) \\
  &= -\frac{\partial x^b}{\partial z^j}\frac{\partial x^a}{\partial z^c}\frac{\partial^2 z^c}{\partial x^b \partial x^d}\frac{\partial x^d}{\partial z^i} \\
  &= -J_j^b J_c^a \frac{\partial^2 z^c}{\partial x^b \partial x^d} J_i^d
\end{align}
$$
Substituting this back into the expression for the Hessian, we obtain
$$
\begin{align}
\frac{\partial^2 F}{\partial z^i \partial z^j} &= H_{ij}^a \frac{\partial F}{\partial x^a} + J_i^a \frac{\partial^2 F}{\partial x^a \partial x^b} J_j^b \\
&= -J_j^b J_c^a \frac{\partial^2 z^c}{\partial x^b \partial x^d} J_i^d \frac{\partial F}{\partial x^a} + J_i^a \frac{\partial^2 F}{\partial x^a \partial x^b} J_j^b \\
&= J_j^b\left(-\frac{\partial^2 z^c}{\partial x^b \partial x^d}J_c^a\frac{\partial F}{\partial x^a} + \frac{\partial^2 F}{\partial x^d \partial x^b}\right)J_i^d
\end{align}
$$
This formula allows us to compute the Hessian in the new coordinate system using only the Jacobian and Hessian of the coordinate transformation $z(x)$ and the gradient and Hessian of $F$ in the original coordinate system.

To summarize, the coordinate change for a `Jet` object is given by
$$
\boxed{
\begin{align}
\frac{\partial F}{\partial z^i} &= J_i^a \frac{\partial F}{\partial x^a} \\
\frac{\partial^2 F}{\partial z^i \partial z^j} &= J_j^b\left(\frac{\partial^2 F}{\partial x^b \partial x^d} - \frac{\partial^2 z^c}{\partial x^b \partial x^d}J_c^a\frac{\partial F}{\partial x^a} \right)J_i^d
\end{align}
}
$$
where $J_i^a = \frac{\partial x^a}{\partial z^i}$ is the Jacobian of the inverse map.

# Basis vectors
A `BasisVectors` object contains a matrix of `Jet` objects, each of which represents a single component of the basis vectors.  When we want to change coordinates, we need to perform the coordinate change on each of the components of the basis vectors.

Let $(E_1, \dots, E_n)$ be a set of basis vectors where
$$
E_j = E_j^i \frac{\partial}{\partial x^i}
$$
In the new coordinate system $(z^1,\dots,z^n)$ we can write
$$
E_j = \tilde{E}_j^a \frac{\partial}{\partial z^a}
$$
We can relate the two coordinate bases by
$$
E_j = E_j^i \frac{\partial}{\partial x^i} = E_j^i \frac{\partial z^a}{\partial x^i} \frac{\partial}{\partial z^a}
$$
which gives us the coordinate change rule
$$
\boxed{
\tilde{E}_j^a = E_j^i \frac{\partial z^a}{\partial x^i}
}
$$
where $\frac{\partial z^a}{\partial x^i}$ are the entries of the Jacobian of the forward coordinate map $z(x)$, i.e. the inverse matrix of $J_a^i = \frac{\partial x^i}{\partial z^a}$.

To differentiate these components with respect to the $z$–coordinates, it is convenient to write the inverse Jacobian explicitly.  Let
$$
J_a^i = \frac{\partial x^i}{\partial z^a}, \qquad G_i^a = \frac{\partial z^a}{\partial x^i} = (J^{-1})_i^{\ a}.
$$
Then
$$
\tilde{E}_j^a = E_j^i G_i^a.
$$

### Gradient
The gradient of the new components with respect to $z$ is obtained by the chain rule.  Using
$$
\begin{align}
  \frac{\partial \tilde{E}^i_j}{\partial z^k} &= \frac{\partial}{\partial z^k} (\frac{\partial z^i}{\partial x^a}E_j^a) \\
  &= -\frac{\partial z^i}{\partial x^b}\frac{\partial^2 x^b}{\partial z^k \partial z^l}\frac{\partial z^l}{\partial x^a}E_j^a + \frac{\partial z^i}{\partial x^a}\frac{\partial x^b}{\partial z^k}\frac{\partial E_j^a}{\partial x^b} \\
  &= -G^i_b H^b_{kl}G^l_a E^a_j + G_a^i J^b_k \frac{\partial E_j^a}{\partial x^b}
\end{align}
$$
So we have that
$$
\boxed{
  \frac{\partial \tilde{E}^i_j}{\partial z^k} = -G^i_b H^b_{km}G^m_a E^a_j + G_a^i J^b_k \frac{\partial E_j^a}{\partial x^b}
}
$$

### Hessian
We can take another derivative with respect to $z^l$ to get the Hessian.  First, we need the derivative of the forward Jacobian $G$ with respect to $z$.  From the identity $G^i_c J^c_a = \delta^i_a$, differentiating with respect to $z^l$ gives
$$
\frac{\partial G^i_c}{\partial z^l} J^c_a + G^i_c H^c_{al} = 0
$$
Multiplying by $G^a_n$ and using $J^c_a G^a_n = \delta^c_n$:
$$
\frac{\partial G^i_n}{\partial z^l} = -G^i_c H^c_{al} G^a_n
$$
We also introduce the third derivative of the coordinate transformation:
$$
T^b_{kml} = \frac{\partial^3 x^b}{\partial z^k \partial z^m \partial z^l}
$$
Now we differentiate the gradient formula
$$
\frac{\partial \tilde{E}^i_j}{\partial z^k} = -G^i_b H^b_{km}G^m_a E^a_j + G_a^i J^b_k \frac{\partial E_j^a}{\partial x^b}
$$
with respect to $z^l$.  For the first term, using the product rule on four factors:
$$
\begin{align}
\frac{\partial}{\partial z^l}\left(-G^i_b H^b_{km}G^m_a E^a_j\right)
&= -\frac{\partial G^i_b}{\partial z^l} H^b_{km}G^m_a E^a_j - G^i_b T^b_{kml}G^m_a E^a_j \\
&\quad - G^i_b H^b_{km}\frac{\partial G^m_a}{\partial z^l} E^a_j - G^i_b H^b_{km}G^m_a J^c_l\frac{\partial E^a_j}{\partial x^c}
\end{align}
$$
Substituting the derivative of $G$:
$$
\begin{align}
&= G^i_c H^c_{nl} G^n_b H^b_{km}G^m_a E^a_j - G^i_b T^b_{kml}G^m_a E^a_j \\
&\quad + G^i_b H^b_{km}G^m_c H^c_{nl} G^n_a E^a_j - G^i_b H^b_{km}G^m_a J^c_l\frac{\partial E^a_j}{\partial x^c}
\end{align}
$$
For the second term:
$$
\begin{align}
\frac{\partial}{\partial z^l}\left(G_a^i J^b_k \frac{\partial E_j^a}{\partial x^b}\right)
&= \frac{\partial G_a^i}{\partial z^l} J^b_k \frac{\partial E_j^a}{\partial x^b} + G_a^i H^b_{kl} \frac{\partial E_j^a}{\partial x^b} + G_a^i J^b_k J^c_l \frac{\partial^2 E_j^a}{\partial x^b \partial x^c} \\
&= -G^i_c H^c_{nl} G^n_a J^b_k \frac{\partial E_j^a}{\partial x^b} + G_a^i H^b_{kl} \frac{\partial E_j^a}{\partial x^b} + G_a^i J^b_k J^c_l \frac{\partial^2 E_j^a}{\partial x^b \partial x^c}
\end{align}
$$
Combining all terms and grouping by powers of $E$, we obtain
$$
\boxed{
\begin{align}
\frac{\partial^2 \tilde{E}^i_j}{\partial z^k \partial z^l}
&= G^i_c H^c_{nl} G^n_b H^b_{km}G^m_a E^a_j + G^i_b H^b_{km}G^m_c H^c_{nl} G^n_a E^a_j - G^i_b T^b_{kml}G^m_a E^a_j \\
&\quad - G^i_b H^b_{km}G^m_a J^c_l\frac{\partial E^a_j}{\partial x^c} - G^i_c H^c_{nl} G^n_a J^b_k \frac{\partial E_j^a}{\partial x^b} + G_a^i H^b_{kl} \frac{\partial E_j^a}{\partial x^b} \\
&\quad + G_a^i J^b_k J^c_l \frac{\partial^2 E_j^a}{\partial x^b \partial x^c}
\end{align}
}
$$
This formula expresses the Hessian of the transformed components in terms of the original components $E^a_j$ and their derivatives, along with the Jacobian $J$, its inverse $G$, and the second and third derivatives $H$ and $T$ of the inverse coordinate map $x(z)$.

# Tangent vectors
A `TangentVector` object represents a tangent vector at a point $p$, written in components of a chosen basis.  In local coordinates $(x^1,\dots,x^n)$, a coordinate basis is given by $\partial_{x^i}$ and a tangent vector can be written as
$$
X = X^i \frac{\partial}{\partial x^i}.
$$
After changing coordinates to $(z^1,\dots,z^n)$ we can also write
$$
X = \tilde{X}^a \frac{\partial}{\partial z^a}.
$$
The coordinate bases are related by
$$
\frac{\partial}{\partial z^a} = J_a^i \frac{\partial}{\partial x^i}, \qquad
J_a^i = \frac{\partial x^i}{\partial z^a},
$$
and the inverse Jacobian is
$$
G_i^a = \frac{\partial z^a}{\partial x^i} = (J^{-1})_i^{\ a}.
$$
Requiring that $X$ is the same geometric vector in both coordinate systems,
$$
X^i \frac{\partial}{\partial x^i}
  = \tilde{X}^a \frac{\partial}{\partial z^a}
  = \tilde{X}^a J_a^i \frac{\partial}{\partial x^i},
$$
we obtain
$$
X^i = \tilde{X}^a J_a^i
\quad\Longrightarrow\quad
\tilde{X}^a = X^i G_i^a.
$$
Thus the components of a tangent vector transform contravariantly with respect to the Jacobian of the coordinate change:
$$
\boxed{
  \tilde{X}^a = X^i \frac{\partial z^a}{\partial x^i}.
}
$$
If $X$ is extended to a local vector field $X(x)$, we can also track the change of its derivatives.  Writing $X^i(x)$ for the components as functions of $x$ and
$$
R_{k i}^a = \frac{\partial G_i^a}{\partial x^k},
$$
the first derivatives in the $z$–coordinates satisfy
$$
\begin{align}
\frac{\partial \tilde{X}^a}{\partial z^b}
  &= J_b^k \frac{\partial}{\partial x^k}\bigl(X^i G_i^a\bigr) \\
  &= J_b^k\left(
        \frac{\partial X^i}{\partial x^k} G_i^a
        + X^i R_{k i}^a
      \right),
\end{align}
$$
which is the exact analogue of the basis-vector gradient formula with $E_j^i$ replaced by $X^i$.  A second differentiation with respect to $z^c$ produces a Hessian expression of the same form as in the `Basis vectors` section, again with $E_j^i$ replaced by $X^i$, $R$ and $S$ encoding the first and second $x$–derivatives of $G$.

# Frames
A `Frame` object consists of a set of basis vectors (stored in a `BasisVectors` object) and a matrix of components (stored as a `Jet`) that express the frame vectors as linear combinations of the basis vectors.  Let $V_k$ be the $k$-th vector in the frame, and let $E_j$ be the $j$-th basis vector.  Then
$$
V_k = C_k^j E_j
$$
where $C_k^j$ are the component functions.

When we change coordinates from $x$ to $z$, we need to perform two operations:
1. Change the coordinates of the basis vectors $E_j$ to obtain $\tilde{E}_j$ (as described in the "Basis vectors" section).
2. Change the coordinates of the component functions $C_k^j$. Since these are scalar functions, we use the scalar Jet transformation rule (as described in the "Jets" section).

Let $J[C]_x$ be the Jet of the components in $x$-coordinates.  The Jet of the components in $z$-coordinates, $J[C]_z$, is obtained by applying the scalar coordinate change to each entry $C_k^j$.

To summarize, the coordinate change for a `Frame` object is given by:
$$
\boxed{
\begin{align}
\text{Basis:} \quad & \text{Transform } E \to \tilde{E} \text{ using BasisVectors rule} \\
\text{Components:} \quad & \text{Transform } C_k^j \to \tilde{C}_k^j \text{ using scalar Jet rule:} \\
& \frac{\partial \tilde{C}_k^j}{\partial z^i} = J_i^a \frac{\partial C_k^j}{\partial x^a} \\
& \frac{\partial^2 \tilde{C}_k^j}{\partial z^i \partial z^l} = J_l^b\left(\frac{\partial^2 C_k^j}{\partial x^b \partial x^d} - \frac{\partial^2 z^c}{\partial x^b \partial x^d}J_c^a\frac{\partial C_k^j}{\partial x^a} \right)J_i^d
\end{align}
}
$$

# Christoffel symbols
Christoffel symbols $\Gamma_{bc}^a$ are not tensors, so they do not transform like tensors. We can derive their transformation rule by considering the definition of the covariant derivative. In the $x$-coordinate system, the covariant derivative is defined by $\nabla_{\partial_{x^b}} \partial_{x^c} = \Gamma_{bc}^a \partial_{x^a}$.
We want to find the Christoffel symbols $\bar{\Gamma}_{ij}^k$ in the $z$-coordinate system, such that $\nabla_{\partial_{z^i}} \partial_{z^j} = \bar{\Gamma}_{ij}^k \partial_{z^k}$.

Recall that the basis vectors transform as $\partial_{z^i} = J_i^a \partial_{x^a}$, where $J_i^a = \frac{\partial x^a}{\partial z^i}$.
Substituting this into the definition:
$$
\begin{align}
\nabla_{\partial_{z^i}} \partial_{z^j} &= \nabla_{J_i^a \partial_{x^a}} (J_j^b \partial_{x^b}) \\
&= J_i^a \nabla_{\partial_{x^a}} (J_j^b \partial_{x^b}) \\
&= J_i^a \left( J_j^b \nabla_{\partial_{x^a}} \partial_{x^b} + \frac{\partial J_j^b}{\partial x^a} \partial_{x^b} \right)
\end{align}
$$
We know that $\nabla_{\partial_{x^a}} \partial_{x^b} = \Gamma_{ab}^c \partial_{x^c}$.
Also, using the chain rule, $J_i^a \frac{\partial}{\partial x^a} = \frac{\partial}{\partial z^i}$. Thus, the second term becomes
$$
J_i^a \frac{\partial J_j^b}{\partial x^a} \partial_{x^b} = \frac{\partial J_j^b}{\partial z^i} \partial_{x^b} = \frac{\partial^2 x^b}{\partial z^i \partial z^j} \partial_{x^b} = H_{ij}^b \partial_{x^b}
$$
Substituting these back:
$$
\nabla_{\partial_{z^i}} \partial_{z^j} = J_i^a J_j^b \Gamma_{ab}^c \partial_{x^c} + H_{ij}^c \partial_{x^c} = (J_i^a J_j^b \Gamma_{ab}^c + H_{ij}^c) \partial_{x^c}
$$
On the other hand, we have
$$
\nabla_{\partial_{z^i}} \partial_{z^j} = \bar{\Gamma}_{ij}^k \partial_{z^k} = \bar{\Gamma}_{ij}^k J_k^c \partial_{x^c}
$$
Equating the coefficients of $\partial_{x^c}$:
$$
\bar{\Gamma}_{ij}^k J_k^c = J_i^a J_j^b \Gamma_{ab}^c + H_{ij}^c
$$
To isolate $\bar{\Gamma}_{ij}^k$, we multiply by the inverse Jacobian $(J^{-1})_c^m = \frac{\partial z^m}{\partial x^c}$.
$$
\bar{\Gamma}_{ij}^m = (J^{-1})_c^m (J_i^a J_j^b \Gamma_{ab}^c + H_{ij}^c)
$$
Renaming indices to match the LHS ($m \to k$):
$$
\boxed{
\bar{\Gamma}_{ij}^k = (J^{-1})_c^k J_i^a J_j^b \Gamma_{ab}^c + (J^{-1})_c^k H_{ij}^c
}
$$
