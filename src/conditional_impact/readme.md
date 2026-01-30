We rely on a class MultiExponentialHawkes that sets that the set of parameters of a 1-dimensional Hawkes whose kernel is a sum of exponential functions with decays beta's and scaled by alpha's: this is a process Markovian in k states where k is the number of exponential components.

For such a kernel, we are looking for the exact form of the propagator operator and the corresponding closed formula for the conditional expectation of the Hawkes intensities as a function of the markovian factors.

- propagator.rs : creating with Propagator::new(hawkes_parameters) produces a structure with new objects lambda and c, such that the propagator associated to the Hawkes is $\delta_0$ + a sum of k exponential components with decays given by lambda's and scaled by c's.

- tail_intensity.rs : creating TailIntensity::new(hawkes_parameters, $c_\lambda$) produces a structure relying on the Propagator with new object the factors $(F_i)_{i=1}^k$ by which we have to multiply the Markovian factors $(R^i_t)_{i=1}^k$ to compute $\int_t^\infty e^{-c_\lambda (s-t)} \mathbb{E}_t[\lambda_s] ds = \sum_{i=1}^k F_i R^i_t + \mu (\frac{1}{c_\lambda} + \sum_{j=1}^N \frac{c_j}{\lambda_j} (\frac{1}{c_\lambda} - \frac{1}{\lambda_j + c_\lambda}))$. It is equiped with a function compute that takes as input the current Markov states and returns this integrated intensity with decay.

- impact_factors.rs : creates TailImpact that takes as input a sequence of events from a Hawkes and returns tail_impact_events which is the sequence of $\int_t^\infty \mathbb{E}_t[\lambda_s] e^{-c_\lambda (s-t)} ds$ for the timestamps $t$ corresponding to the jump times of the Hawkes.

- impact_path.rs : It puts the blocks together, we input the queue path and the queue_bar path, as well as the Hawkes event sequence, and it produces the time series: $$\int_0^t (\bar{q}_s q_s) dN_s + (\bar{q}_t - q_t) \int_t^{\infty} e^{-c_\lambda (s-t)} \mathbb{E}_t[\lambda_s] ds$$