# Custom queue simulation experiment

Edit `config` in `main.py`, then:

    python experiments/queue_simulation/custom_experiment/main.py

Set `counterfactual=False` for with-us conditioning, or `counterfactual=True` for the without-us counterfactual. Outputs (`times.npy`, `queue_paths.npy`) land in `output/with_us/` or `output/without_us/` (gitignored).
