# Custom aggressive impact experiment

Edit `config` in `main.py`, then:

    python experiments/agressive_impact/custom_experiment/main.py

Set `counterfactual=False` for with-us conditioning, or `counterfactual=True` for the without-us counterfactual. For the hybrid model, set `model="hybrid"` and provide `bar_kappa`. Outputs land in `output/` (gitignored).
