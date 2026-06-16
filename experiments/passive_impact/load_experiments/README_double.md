# Double Queue Impact

Extends passive impact to a full bid-ask queue pair. A limit-order metaorder is executed on the ask side only; we observe how impact propagates to both sides and compute the net price impact (ask $-$ bid).

## Setup

- **Bid-ask queues**: Independent dynamics on each side, same affine intensities as single queue
- **Market orders**: Independent Hawkes processes $N^a$, $N^b$ with identical parameters
- **Metaorder**: Limit orders on the ask side only
- **Impact scale**: normalized passive impact multiplied by `c_kappa_effective = -0.00001713`
- **Paths**: 500 conditional simulations, horizon $T = 100$s

## Results

### Queue dynamics

<p align="center">
  <img src="images/bidask_queue_given_q.png" width="48%" alt="Queues given q"/>
  <img src="images/bidask_queue_given_qbar.png" width="48%" alt="Queues given qbar"/>
</p>

*Left*: Counterfactual queues given the baseline $q$. Ask (blue) and bid (orange) evolve independently; the metaorder only affects the ask side. *Right*: Same, conditioning on the impacted queue $\bar{q}$.

### Per-side impact

<p align="center">
  <img src="images/ask_impact_given_q.png" width="48%" alt="Ask impact given q"/>
  <img src="images/bid_impact_given_q.png" width="48%" alt="Bid impact given q"/>
</p>

*Left*: Ask-side impact $I^a(t)$ (directly affected by the metaorder). *Right*: Bid-side impact $I^b(t)$ (indirectly affected through queue coupling).

### Net price impact (ask $-$ bid)

<p align="center">
  <img src="images/bidask_impact_given_q.png" width="48%" alt="Net impact given q"/>
  <img src="images/bidask_impact_given_qbar.png" width="48%" alt="Net impact given qbar"/>
</p>

*Left*: Total price impact given baseline $q$. *Right*: Total price impact given impacted $\bar{q}$. Green band shows $\pm 1$ standard deviation.

## How to Run

```bash
cargo run --release --bin double_queue_efficient_with_us
cargo run --release --bin double_queue_efficient_without_us
python plot_utils.py
```

Override the impact scale with `C_KAPPA_EFFECTIVE=<value> cargo run --release --bin ...`.
