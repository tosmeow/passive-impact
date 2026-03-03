In this case here, I will want to do the agressive market impact experiment under a different price model.

Our queues are in the single_queue experiment setup, with L, C and N (last one is a Hawkes).

Now, corresponding to this Hawkes parameter setup as a sum of exponentials, we have a propagator in conditional_impact/impact_utils equal to Id + sum of exponnetials that we get in this propagator.rs

We call xi this operator.

Then, our new price model is P_t = \int_0^t kappa(q^a_s) \xi(t-s) dN^a_s - \int_0^t kappa(q^b_s) \xi(t-s) dN^b_s

In terms of market impact, it is now giving: we add a market order N^o, that changes that queues into \bar{q}^a_s = q^a_0 + \bar{L}^a_s - \bar{C}^a_s - N^a_s - N^o_s (only L, C are impacted by us adding the metaorder, not the hawkes).

Then, the corresponding price impact is:

MI_t = \int_0^t (kappa(\bar{q}^a_s) - kappa(q^a_s)) \xi(t-s) dN^a_s + \int_0^t kappa(\bar{q}^a_s) dN^o_s

This is what we want to implement now: we pick a series of external events for N^o, generate first a path of q with L, C, N;
Then, we generate a counterfactual path for L, C impacted by this N^o acting as reducing the queue.

We then aggregate these on each path to get a path MI_t and store the distributions in data, and in python/experiments, create /agressive_impact and make then inside a notebook to display the corresponding queue dynamic, impact etc.

We will choose the same parameters for queues and hawkes as in single_queue, and the kappa function to be kappa(q) = cq + d with d > 0, q < 0.