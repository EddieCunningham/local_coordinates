# Jacobian derivatives

Suppose we have a function $z(x)$ that has the first three derivatives of $z$ at $x = p$ given by
$$
\begin{align}
  \frac{\partial z^i}{\partial x^j},  \frac{\partial^2 z^i}{\partial x^j \partial x^k},  \frac{\partial^3 z^i}{\partial x^j \partial x^k \partial x^l}
\end{align}
$$
and we want to use these expressions to compute the first three derivatives of the inverse function $x(z)$,
$$
\begin{align}
  \frac{\partial x^i}{\partial z^j},  \frac{\partial^2 x^i}{\partial z^j \partial z^k},  \frac{\partial^3 x^i}{\partial z^j \partial z^k \partial z^l}
\end{align}
$$

We get the first derivative trivially by inverting the Jacobian.  For the other two, we can use the chain rule.

### Second derivative
The second derivative of the inverse function is given by:
$$
\begin{align}
  \frac{\partial^2 x^i}{\partial z^j \partial z^k} &= \frac{\partial x^m}{\partial z^k}\frac{\partial}{\partial x^m}\left(\frac{\partial x^i}{\partial z^j}\right) \\
  &= -\frac{\partial x^m}{\partial z^k}\frac{\partial x^i}{\partial z^a}\frac{\partial^2 z^a}{\partial x^m \partial x^b}\frac{\partial x^b}{\partial z^j} \\
  &= -\frac{\partial^2 z^a}{\partial x^m \partial x^b}\frac{\partial x^i}{\partial z^a}\frac{\partial x^b}{\partial z^j}\frac{\partial x^m}{\partial z^k}
\end{align}
$$

### Third derivative
The third derivative of the inverse function is given by:
$$
\begin{align*}
  \frac{\partial^3 x^i}{\partial z^j \partial z^k \partial z^l} &= \frac{\partial x^n}{\partial z^l}\frac{\partial}{\partial x^n}\left(\frac{\partial^2 x^i}{\partial z^j \partial z^k}\right) \\
  &= -\frac{\partial x^n}{\partial z^l}\left(\frac{\partial}{\partial x^n}(\frac{\partial^2 z^a}{\partial x^m \partial x^b})\frac{\partial x^i}{\partial z^a}\frac{\partial x^b}{\partial z^j}\frac{\partial x^m}{\partial z^k} + \frac{\partial^2 z^a}{\partial x^m \partial x^b}\frac{\partial}{\partial x^n}(\frac{\partial x^i}{\partial z^a}\frac{\partial x^b}{\partial z^j}\frac{\partial x^m}{\partial z^k})\right) \\
  &= -\frac{\partial^3 z^a}{\partial x^n \partial x^m \partial x^b}\frac{\partial x^n}{\partial z^l}\frac{\partial x^i}{\partial z^a}\frac{\partial x^b}{\partial z^j}\frac{\partial x^m}{\partial z^k} - \frac{\partial^2 z^a}{\partial x^m \partial x^b}\frac{\partial}{\partial z^l}(\frac{\partial x^i}{\partial z^a}\frac{\partial x^b}{\partial z^j}\frac{\partial x^m}{\partial z^k}) \\
  &= -\frac{\partial^3 z^a}{\partial x^n \partial x^m \partial x^b}\frac{\partial x^n}{\partial z^l}\frac{\partial x^i}{\partial z^a}\frac{\partial x^b}{\partial z^j}\frac{\partial x^m}{\partial z^k} \\
  &\qquad\qquad - \frac{\partial^2 z^a}{\partial x^m \partial x^b}\frac{\partial^2 x^i}{\partial z^a \partial z^l}\frac{\partial x^b}{\partial z^j}\frac{\partial x^m}{\partial z^k} \\
  &\qquad\qquad\qquad - \frac{\partial^2 z^a}{\partial x^m \partial x^b}\frac{\partial x^i}{\partial z^a}\frac{\partial^2 x^b}{\partial z^j \partial z^l}\frac{\partial x^m}{\partial z^k} \\
  &\qquad\qquad\qquad\qquad - \frac{\partial^2 z^a}{\partial x^m \partial x^b}\frac{\partial x^i}{\partial z^a}\frac{\partial x^b}{\partial z^j}\frac{\partial^2 x^m}{\partial z^k \partial z^l} \\
  &= -\frac{\partial x^i}{\partial z^a}\left(\frac{\partial^3 z^a}{\partial x^n \partial x^m \partial x^b}\frac{\partial x^n}{\partial z^l}\frac{\partial x^b}{\partial z^j}\frac{\partial x^m}{\partial z^k} + \frac{\partial^2 z^a}{\partial x^m \partial x^b}\left(\frac{\partial^2 x^m}{\partial z^j \partial z^k}\frac{\partial x^b}{\partial z^l} + \frac{\partial^2 x^m}{\partial z^k \partial z^l}\frac{\partial x^b}{\partial z^j} + \frac{\partial^2 x^m}{\partial z^l \partial z^j}\frac{\partial x^b}{\partial z^k}\right)\right)
\end{align*}
$$

To summarize, we have that
$$
\boxed{
  \begin{align}
    \frac{\partial x^i}{\partial z^j} &= \frac{\partial z^i}{\partial x^j}, \\
    \frac{\partial^2 x^i}{\partial z^j \partial z^k} &= -\frac{\partial^2 z^a}{\partial x^m \partial x^b}\frac{\partial x^i}{\partial z^a}\frac{\partial x^b}{\partial z^j}\frac{\partial x^m}{\partial z^k}, \\
    \frac{\partial^3 x^i}{\partial z^j \partial z^k \partial z^l} &= -\frac{\partial x^i}{\partial z^a}\left(\frac{\partial^3 z^a}{\partial x^n \partial x^m \partial x^b}\frac{\partial x^n}{\partial z^l}\frac{\partial x^b}{\partial z^j}\frac{\partial x^m}{\partial z^k} + \frac{\partial^2 z^a}{\partial x^m \partial x^b}\left(\frac{\partial^2 x^m}{\partial z^j \partial z^k}\frac{\partial x^b}{\partial z^l} + \frac{\partial^2 x^m}{\partial z^k \partial z^l}\frac{\partial x^b}{\partial z^j} + \frac{\partial^2 x^m}{\partial z^l \partial z^j}\frac{\partial x^b}{\partial z^k}\right)\right)
  \end{align}
}
$$