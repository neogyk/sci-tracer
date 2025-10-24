### Proximal policy optimization:

$$
\begin{align}
L_{PPO} = E_{q~P(Q), p~\pi_{old}(p,q)} \frac{1}{|p|} \sum_{t=1} min [\frac{\pi(p_{t}|q, p_{<t})}{\pi_{old}(p_{t}|q, p_{<t})}A_{t}, clip(\frac{\pi(p_{t}|q, p_{t})}
{\pi_{old}(p_{t}|q, p_{<t})}, 1-\eps, 1+\eps)A_{t}])
\end{align}
$$

### Group relative policy optimization

### Informational gain policy optimization

### Refference List:
