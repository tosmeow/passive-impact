This is an implementation of a framework for market impact relying on the hypothesis that the queue dynamics are inhomogeneous Poisson processes with intensities that are functionals of the limit, cancel and market orders trajectories.

This is implemented here in the specific case where $\lambda^L_t = \lambda^L(q_{t-})$, $\lambda^C_t = \lambda^C(q_{t-})$ and $N$ is a Hawkes process. It can be further generalized to more complicated intensity functionals, at the expense of computational efficiency of the impact.

This crate is organized with the following modules:
- models contains abstractions of:
    - queues events in an order book.
    - paths of queues events, consisting in a full reconstitution of a time period.
    - point processes, with methods to compute intensities and appropriate bounds to apply Ogata for simulation.
    - hawkes process with kernel as sum of exponentials, and implementing for this point process the efficient markovian computation of intensity.

- simulation implements simulators of queues, hawkes and conditional queues. I need to check if all can be seemlessly integrated as 1: conditional sampler be an abstract class with a markovian state; 2: same thing for queue simulator and thinning together.

- conditional_impact implements in the case of a propagator with weights $\kappa(q)$ linear in the queue and with $\lambda^L - \lambda^C$ the close formula that we obtained for $$\mathbb{E}_t[\int_t^\infty (\kappa(\bar{q}^a_s) - \kappa(q^a_s))dN^a_s]$$
    This term is computed in constant time, and we implemented in impact_path the computation of the impact timeseries if we input the timeseries of values of $\bar{q}^a_s$,  $q^a_s$ and $N^a_s$.

    This relies on a computation of $\int_t^\infty e^{-c_{\lambda}(s-t)} \mathbb{E}_t[\lambda^a_s]ds$ explicitely and in constant time when the Hawkes kernel is a sum of exponentials. This computation relies on computing the associated propagator kernel explicitely parametrized by a sum of exponentials whose coefficients are computed using finite_difference.rs and ivt.rs in utils.

- utils implements the two elementary functions:
    - ivt is a custom bisection method to search zeros in a particular setting where we only know that the function f is defined on an open interval (a,b) and that the limits of f at a and b are of opposite sign (but possibly infinite). This implements first a search of appropriate endpoints where the function is defined but has opposite sign to start the bisection.
    - finite_difference implements a finite difference method with a custom parameterization to produce low tolerance for errors in the obtained derivative: we say we are close enough to our derivative if the difference between two "finite differences" is lower than a set tolerance, and we pick the average of these two candidates as our derivative.