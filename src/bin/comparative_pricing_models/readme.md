On one hand, with agressive_impact, we introduced a different price model from what we had in single_queue

Now, i want to do something different: i use the price model in agressive_impact but for passive impact as well, that is impact will be given just as int_0^t xi(t-s) (kappa(\bar{q}^a_s) - kappa(q^a_s))dN^a_s for this new model; i want that we compare this to the single_queue case where we use also the same underlying queue, hawkes, conditional queue simulations for both so that they're really comparable

the outputs from single_queue were as if we had kappa(q) = q; we can do it with this as well in the new model as well so that it's cleanly comparable