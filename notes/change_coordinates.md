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
A `BasisVectors` object contains a matrix of `Jet` objects, each of which represents a single component of the basis vectors.  When we want to change coordinates, we need to perform the coordinate change on each of the components of the basis vectors.  Let $(E_1, \dots, E_n)$ be a set of basis vectors where $E_j = E_j^i \frac{\partial}{\partial x^i}$.  The components of these vectors in the new coordinate system are given by $E_j = \tilde{E}_j^a \frac{\partial}{\partial z^a}$.  Using the transformation law implemented in `local_coordinates/basis.py`, the components transform as
$$
\tilde{E}_j^a = E_j^i \frac{\partial x^i}{\partial z^a}
$$
To find the derivatives of the new components, we differentiate this expression with respect to $z$.  Let $J_a^i = \frac{\partial x^i}{\partial z^a}$, $H_{ab}^i = \frac{\partial^2 x^i}{\partial z^a \partial z^b}$, and $T_{abc}^i = \frac{\partial^3 x^i}{\partial z^a \partial z^b \partial z^c}$ denote the derivatives of the inverse map $x(z)$.

### Gradient
The gradient of the new components is given by
$$
\begin{align}
\frac{\partial \tilde{E}_j^a}{\partial z^b} &= \frac{\partial}{\partial z^b}\left(E_j^i \frac{\partial x^i}{\partial z^a}\right) \\
&= \frac{\partial E_j^i}{\partial x^k} \frac{\partial x^k}{\partial z^b} \frac{\partial x^i}{\partial z^a} + E_j^i \frac{\partial^2 x^i}{\partial z^a \partial z^b}
\end{align}
$$
Using our notation for the inverse Jacobian derivatives, this is
$$
\frac{\partial \tilde{E}_j^a}{\partial z^b} = \frac{\partial E_j^i}{\partial x^k} J_b^k J_a^i + E_j^i H_{ab}^i
$$

### Hessian
For the Hessian, we differentiate again with respect to $z^c$:
$$
\begin{align}
\frac{\partial^2 \tilde{E}_j^a}{\partial z^c \partial z^b} &= \frac{\partial}{\partial z^c}\left(\frac{\partial E_j^i}{\partial x^k} \frac{\partial x^k}{\partial z^b} \frac{\partial x^i}{\partial z^a} + E_j^i \frac{\partial^2 x^i}{\partial z^a \partial z^b}\right) \\
&= \left(\frac{\partial^2 E_j^i}{\partial x^k \partial x^l} \frac{\partial x^l}{\partial z^c} \frac{\partial x^k}{\partial z^b} \frac{\partial x^i}{\partial z^a} + \frac{\partial E_j^i}{\partial x^k} \frac{\partial^2 x^k}{\partial z^b \partial z^c} \frac{\partial x^i}{\partial z^a} + \frac{\partial E_j^i}{\partial x^k} \frac{\partial x^k}{\partial z^b} \frac{\partial^2 x^i}{\partial z^a \partial z^c}\right) \\
&\quad + \left(\frac{\partial E_j^i}{\partial x^l} \frac{\partial x^l}{\partial z^c} \frac{\partial^2 x^i}{\partial z^a \partial z^b} + E_j^i \frac{\partial^3 x^i}{\partial z^a \partial z^b \partial z^c}\right)
\end{align}
$$
Using the compact notation:
$$
\frac{\partial^2 \tilde{E}_j^a}{\partial z^c \partial z^b} = \frac{\partial^2 E_j^i}{\partial x^k \partial x^l} J_c^l J_b^k J_a^i + \frac{\partial E_j^i}{\partial x^k} H_{bc}^k J_a^i + \frac{\partial E_j^i}{\partial x^k} J_b^k H_{ac}^i + \frac{\partial E_j^i}{\partial x^l} J_c^l H_{ab}^i + E_j^i T_{abc}^i
$$

To summarize, the coordinate change for `BasisVectors` is given by:
$$
\boxed{
\begin{align}
\tilde{E}_j^a &= E_j^i J_a^i \\
\frac{\partial \tilde{E}_j^a}{\partial z^b} &= \frac{\partial E_j^i}{\partial x^k} J_b^k J_a^i + E_j^i H_{ab}^i \\
\frac{\partial^2 \tilde{E}_j^a}{\partial z^c \partial z^b} &= \frac{\partial^2 E_j^i}{\partial x^k \partial x^l} J_c^l J_b^k J_a^i + \frac{\partial E_j^i}{\partial x^k} H_{bc}^k J_a^i + \frac{\partial E_j^i}{\partial x^k} J_b^k H_{ac}^i + \frac{\partial E_j^i}{\partial x^l} J_c^l H_{ab}^i + E_j^i T_{abc}^i
\end{align}
}
$$
where $J, H, T$ represent the first, second, and third derivatives of the inverse map $x(z)$.

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
