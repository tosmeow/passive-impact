# Custom aggressive impact experiment

Edit `config` in `main.py`, then:

    python experiments/agressive_impact/custom_experiment/main.py

Set `counterfactual=False` for with-us conditioning, or `counterfactual=True` for the without-us counterfactual. Tune `bar_kappa` and `kappa` directly in the config. Outputs land in `output/with_us/` or `output/without_us/` (gitignored).
