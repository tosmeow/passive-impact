# Custom passive impact experiment

Edit the `config = pi.PassiveImpactConfig(...)` block in `main.py`, then run:

    python experiments/passive_impact/custom_experiment/main.py

Set `counterfactual=False` for with-us conditioning, or `counterfactual=True` for the without-us counterfactual.

Outputs land in `output/with_us/` or `output/without_us/` (gitignored).
