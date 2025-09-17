# 1-D atom simulation
## H atom

$$
i \frac{\partial}{\partial t}\psi(x,t) = \hat{H}\psi(x,t)
$$

where,

$$
\begin{aligned}
\hat{H} &= -\frac{1}{2}\nabla^2 + \hat{V}(x) + \hat{V}_{\rm{ext}}(x) \\
        &= -\frac{1}{2}\frac{d^2}{dx^2} - \frac{1}{\sqrt{1+x^2}} + E(t)x
\end{aligned}
$$

## He atom

$$
i \frac{\partial}{\partial t}\psi(x,t) = \hat{H}\psi(x,t)
$$

where,

$$
\hat{H} = \hat{H}_1 (t) + \hat{H}_2
$$

and, $\hat{H}_1 (t)$, $\hat{H}_2$ are represented as

$$
\begin{aligned}
\hat{H}_1(t) &= -\frac{1}{2}\nabla^2 + \hat{V}(x) + \hat{V}_{\rm{ext}}(x) \\
        &= -\frac{1}{2}\frac{d^2}{dx^2} - \frac{2}{\sqrt{1+x^2}} + E(t)x
\end{aligned}
$$

$$
\hat{H}_2 (t) = \int dx'\ \frac{\psi^*(x')\psi(x')}{\sqrt{(x-x')^2+1}}
$$

## Imaginary time relaxation to the electronic ground state
With $t \rightarrow -i\tau$ transformation to TDSE, we can obtain the ground state for the problem.

$$
i\frac{d}{dt}\Psi(t) = H_0\Psi(t) \longrightarrow
\frac{d}{d\tau}\Psi(-i\tau) = -H_0\Psi(-i\tau), 
$$