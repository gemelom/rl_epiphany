# Policy Gradient

## Policy

Policy 𝜋 is a network with parameter 𝜃 

- Input: the observation of machine represented as a vector or a matrix 
- Output: each action corresponds to a neuron in output layer

## Trajectory

- Trajectory $$\tau = \{{s_1, a_1, s_2, a_2, \cdots, s_T, a_T}\} $$

- $$ p_{\theta}(\tau)= p(s_1)p_{\theta}(a_1|s_1)p(s_2|s_1, a_1)p_{\theta}(a_2|s_2)p(s_3|s_2, a_2)\cdots = p(s_1)\prod_{t=1}^{T} p_{\theta}(a_t|s_t)p(s_{t+1}|s_t, a_t) $$






