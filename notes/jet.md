# Jet notes
Let $\mathcal{M}$ be a smooth, $d$ dimensional manifold and let $p \in \mathcal{M}$ and let $F: \mathcal{M} \to \mathbb{R}^n$ be a smooth function.  Suppose that $(\frac{\partial}{\partial x^1}, \ldots, \frac{\partial}{\partial x^d})$ is a local coordinate system around $p$.  The second order Jet of $F$ at $p$, expressed in this coordinate system, is the tuple
$$
J[F]_p = \left(F_p^k, \frac{\partial F_p^k}{\partial x^i}, \frac{\partial^2 F_p^k}{\partial x^i \partial x^j}\right), \quad i,j=1,\ldots,d, \quad k=1,\ldots,n
$$
The Jet of $F$ at $p$ represents the second order Taylor expansion of $F$ at $p$ in this coordinate system.  That is,
$$
F(q)^k \approx F_p^k + \sum_{i=1}^d \frac{\partial F_p^k}{\partial x^i} (q^i - p^i) + \frac{1}{2}\sum_{i,j=1}^d \frac{\partial^2 F_p^k}{\partial x^i \partial x^j} (q^i - p^i)(q^j - p^j)
$$
for $q \in \mathcal{M}$ in a neighborhood of $p$.  To work with Jets, there are two operations that we need to be able to perform.  The first is a change of coordinates and the second is pushforward through a smooth map.

## Change of coordinates
Suppose that $(\frac{\partial}{\partial z^1}, \ldots, \frac{\partial}{\partial z^d})$ is another local coordinate system around $p$.  Then, we can express the Jet of $F$ at $p$ in this new coordinate system as
$$
J[F]_p = \left(F_p^k, \frac{\partial F_p^k}{\partial z^i}, \frac{\partial^2 F_p^k}{\partial z^i \partial z^j}\right), \quad i,j=1,\ldots,d, \quad k=1,\ldots,n
$$
where the partial derivatives are now taken with respect to the new coordinate system.  The values of the gradient and Hessian can be computed using the chain rule as follows:
$$
\frac{\partial F_p^k}{\partial z^i} = \frac{\partial x^a}{\partial z^i} \frac{\partial F_p^k}{\partial x^a}
$$
and
$$
\begin{align}
\frac{\partial^2 F_p^k}{\partial z^i \partial z^j} &= \frac{\partial}{\partial z^j}\left(\frac{\partial x^a}{\partial z^i} \frac{\partial F_p^k}{\partial x^a}\right) \\
&= \frac{\partial^2 x^a}{\partial z^j \partial z^i} \frac{\partial F_p^k}{\partial x^a} + \frac{\partial x^a}{\partial z^i} \frac{\partial^2 F_p^k}{\partial x^a \partial z^j} \\
&= \frac{\partial^2 x^a}{\partial z^j \partial z^i} \frac{\partial F_p^k}{\partial x^a} + \frac{\partial x^a}{\partial z^i} \frac{\partial^2 F_p^k}{\partial x^a \partial x^b} \frac{\partial x^b}{\partial z^j}
\end{align}
$$

This expression is not too practical though as it requires the inverse of the coordinate transformation.  Instead, we can write the result in terms of the Jacobian of the coordinate transformation, $z(x)$ by using the fact that $\frac{\partial x}{\partial z} = \left(\frac{\partial z}{\partial x}\right)^{-1}$.  This means that
$$
\frac{\partial x^a}{\partial z^i} = {\left(\frac{\partial z}{\partial x}\right)^{-1}}_i^a
$$
and
$$
\begin{align}
  \frac{\partial^2 x^a}{\partial z^i \partial z^j} &= \frac{\partial}{\partial z^j}\left(\frac{\partial x^a}{\partial z^i}\right) \\
  &= \frac{\partial x^b}{\partial z^j}\frac{\partial}{\partial x^b}\left(\frac{\partial x^a}{\partial z^i}\right) \\
  &= -\frac{\partial x^b}{\partial z^j}\frac{\partial x^a}{\partial z^c}\frac{\partial^2 z^c}{\partial x^b \partial x^d}\frac{\partial x^d}{\partial z^i}
\end{align}
$$
and so we have
$$
\begin{align}
\frac{\partial^2 F_p^k}{\partial z^i \partial z^j} &= -\frac{\partial x^b}{\partial z^j}\frac{\partial^2 z^c}{\partial x^b \partial x^d}\frac{\partial x^d}{\partial z^i}\frac{\partial x^a}{\partial z^c}\frac{\partial F_p^k}{\partial x^a} + \frac{\partial x^a}{\partial z^i} \frac{\partial^2 F_p^k}{\partial x^a \partial x^b} \frac{\partial x^b}{\partial z^j} \\
&= \frac{\partial x^b}{\partial z^j}\left(-\frac{\partial^2 z^c}{\partial x^b \partial x^d}\frac{\partial x^a}{\partial z^c}\frac{\partial F_p^k}{\partial x^a} + \frac{\partial^2 F_p^k}{\partial x^d \partial x^b}\right)\frac{\partial x^d}{\partial z^i}
\end{align}
$$


## Pushforward through a smooth map
Next, suppose that $T: \mathbb{R}^n \to \mathbb{R}^m$ is a smooth transformation of the Jet value.  Let $G(p) = T\circ F(p)$.  Then, the Jet of $G$ at $p$ is given by
$$
J[G]_p = \left(G_p^k, \frac{\partial G_p^k}{\partial x^i}, \frac{\partial^2 G_p^k}{\partial x^i \partial x^j}\right), \quad i,j=1,\ldots,d, \quad k=1,\ldots,m
$$
The gradients and Hessians can be computed using the chain rule as follows:
$$
\frac{\partial G_p^k}{\partial x^i} = \frac{\partial T^k}{\partial F^a} \frac{\partial F_p^a}{\partial x^i}
$$
and
$$
\begin{align}
\frac{\partial^2 G_p^k}{\partial x^i \partial x^j} &= \frac{\partial}{\partial x^j}\left(\frac{\partial T^k}{\partial F^a} \frac{\partial F_p^a}{\partial x^i}\right) \\
&= \frac{\partial}{\partial x^j}\left(\frac{\partial T^k}{\partial F^a} \right)\frac{\partial F_p^a}{\partial x^i} + \frac{\partial T^k}{\partial F^a} \frac{\partial^2 F_p^a}{\partial x^i \partial x^j} \\
&= \frac{\partial F_p^b}{\partial x^j}\frac{\partial^2 T^k}{\partial F^b \partial F^a} \frac{\partial F_p^a}{\partial x^i} + \frac{\partial T^k}{\partial F^a} \frac{\partial^2 F_p^a}{\partial x^i \partial x^j}
\end{align}
$$
