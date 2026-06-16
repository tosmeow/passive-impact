# Custom passive impact experiment

Edit the `config = pi.PassiveImpactConfig(...)` block in `main.py`, then run:

    python experiments/passive_impact/custom_experiment/main.py

Set `counterfactual=False` for with-us conditioning, or `counterfactual=True` for the without-us counterfactual.
For single-queue runs, set `side="ask"` for the current sign convention or `side="bid"` to flip the impact sign for bid-posted buy orders.
Set `c_kappa_effective` to the final price-impact scale. Use `1.0` for normalized passive impact, or the calibrated reduced-form slope such as `-0.00001713` to match `impact_cost`.

Outputs land in `output/with_us/` or `output/without_us/` (gitignored).
