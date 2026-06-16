# Lifecycle Impact-Cost Formulas

This note maps the current `config.toml` fields to the formulas used by
`lifecycle_passive_cost.py`. The experiment is fixed to `tail_propagator`.

## Config Key Map

| Key | Where it enters |
| --- | --- |
| `aggregated_path` | Empirical event/queue input for $\bar q_k$, $t_k$, sides, and quantities. |
| `output_dir` | CSV/JSON output directory. |
| `image_dir` | PNG output directory. |
| `episode_spacing_seconds` | Candidate episode spacing bucket. |
| `max_episodes` | Cap on selected episode candidates. |
| `randomize_episodes` | Randomized episode cap selection. |
| `horizon_seconds` | Episode horizon $H$. |
| `output_step_seconds` | Output grid step $\Delta$. |
| `warmup_seconds` | Pre-origin warmup $w$. |
| `n_policy_paths` | Synthetic lifecycle paths per episode. |
| `min_market_events` | Minimum consuming market events after episode origin. |
| `start_time`, `end_time` | Calendar-time candidate filters. |
| `seed` | Base random seed; path seed is `seed + 100003e + p`. |
| `raw_side` | Own passive limit/cancel side used for candidate rows. |
| `queue_col` | Queue column for $\bar q_k$, no-us queue, and price sign. |
| `market_side` | Consuming market-order process $N$. |
| `n_cycles` | Number of lifecycle cycles. |
| `orders_per_cycle` | Number of orders per cycle, $n$. |
| `order_qty` | Quantity added by posts and removed by fills/cancels. |
| `posting_spacing_seconds` | Time spacing between posts. |
| `fill_count_model` | Fill count law. |
| `fill_probability` | Binomial fill probability. |
| `fixed_filled_orders` | Fixed fill count, when set. |
| `fill_selection` | Which slots fill. |
| `fill_time_model` | Fill-time law. |
| `fill_wait_mean_seconds` | First/independent fill exponential mean. |
| `fill_gap_mean_seconds` | Clustered fill-gap exponential mean. |
| `min_resting_seconds` | Minimum rest time before cancels. |
| `cancel_delay_seconds` | Delay from latest fill to cancels. |
| `cancel_jitter_seconds` | Uniform cancel-time jitter. |
| `repost_delay_seconds` | Delay before the next cycle. |
| `propagator_kappa` | Permanent propagator level $\kappa_s$. |
| `propagator_gamma` | Reduced-form queue slope $\kappa_1$. |
| `propagator_weights` | Transient propagator weights $w_i$. |
| `propagator_beta` | Transient propagator rates $\beta_i$. |
| `propagator_tail_zeta` | Baseline tail term $\zeta$. |
| `b_l`, `b_c` | Queue mean-reversion scale $C_\lambda=b_c-b_l$. |

## Time And Episodes

For an episode starting at empirical time $T_e$:

$$
W_e = [T_e - w,\; T_e + H]
$$

- $w=\texttt{warmup\_seconds}$
- $H=\texttt{horizon\_seconds}$

The output grid is:

$$
\mathcal G = \{0,\Delta,2\Delta,\ldots,H\},
\qquad \Delta=\texttt{output\_step\_seconds}.
$$

Episode candidates are empirical limit rows with `side = raw_side` between
`start_time` and `end_time`. `episode_spacing_seconds` keeps the first
candidate in each spacing bucket. `max_episodes` caps the selected candidates;
if `randomize_episodes = true`, the cap is sampled with `seed`, otherwise the
first candidates are used. `min_market_events` filters out windows with too few
post-origin consuming market events. `n_policy_paths` is the number of
synthetic lifecycle paths generated per accepted episode.

Path $p$ in episode $e$ uses:

$$
\text{path\_seed}_{e,p} = \texttt{seed} + 100003e + p.
$$

## Queue And Active Quantity

Market rows are selected with `side = market_side` and sampled from
`queue_col`. The observed factual queue is:

$$
\bar q_k = \texttt{queue\_col}(t_k)
$$

at consuming market times $t_k$. The price sign is:

$$
s =
\begin{cases}
-1, & \texttt{queue\_col}=\texttt{q\_b},\\
+1, & \texttt{queue\_col}=\texttt{q\_a}.
\end{cases}
$$

`raw_side` labels own passive post/cancel rows in the data convention.

Own active displayed quantity is the cumulative lifecycle displacement:

$$
A(t)=\max\left(0,\sum_m \delta_m \mathbf 1_{\{u_m \le t\}}\right),
$$

where posts have $\delta_m=+\texttt{order\_qty}$, fills and cancels have
$\delta_m=-\texttt{order\_qty}$. The no-us queue used for impact is:

$$
q_k=\max(\bar q_k-A(t_k),0),
\qquad d_k=\bar q_k-q_k.
$$

## Lifecycle

For cycle $c=0,\ldots,\texttt{n\_cycles}-1$, let $R_c$ be the cycle start
and $n=\texttt{orders\_per\_cycle}$. Order slot $j$ posts at:

$$
p_{c,j}=R_c+j\,\texttt{posting\_spacing\_seconds},
\qquad j=0,\ldots,n-1.
$$

The posting block ends at:

$$
P_c=R_c+n\,\texttt{posting\_spacing\_seconds}.
$$

The number of fills in the cycle is:

$$
N_c =
\begin{cases}
\min(\texttt{fixed\_filled\_orders}, n), &
\texttt{fill\_count\_model}=\texttt{fixed}\ \text{and fixed\_filled\_orders is set},\\
n, & \texttt{fill\_count\_model}=\texttt{fixed}\ \text{and fixed\_filled\_orders is unset},\\
\operatorname{Binomial}(n,\texttt{fill\_probability}), & \texttt{fill\_count\_model}=\texttt{binomial}.
\end{cases}
$$

`fill_selection = "oldest"` fills the first $N_c$ slots; `"random"` samples
$N_c$ slots uniformly without replacement.

For `fill_time_model = "clustered_exponential"`:

$$
f_{c,0}=P_c+X_0,\qquad
X_0\sim \operatorname{Exp}(\texttt{fill\_wait\_mean\_seconds}),
$$

$$
f_{c,r}=f_{c,r-1}+X_r,\qquad
X_r\sim \operatorname{Exp}(\texttt{fill\_gap\_mean\_seconds}),\quad r\ge 1.
$$

For `fill_time_model = "independent_exponential"`:

$$
f_{c,j}=p_{c,j}+X_j,\qquad
X_j\sim \operatorname{Exp}(\texttt{fill\_wait\_mean\_seconds}).
$$

Unfilled orders cancel after:

$$
B_c=\max\left(P_c+\texttt{min\_resting\_seconds},
\max_r f_{c,r}+\texttt{cancel\_delay\_seconds}\right),
$$

with cancel times:

$$
a_{c,j}=B_c+U_j,\qquad
U_j\sim \operatorname{Uniform}(0,\texttt{cancel\_jitter\_seconds}).
$$

If `cancel_jitter_seconds = 0`, all unfilled orders cancel at $B_c$. The next
cycle starts at:

$$
R_{c+1}=\max(\text{cycle fills and cancels})+\texttt{repost\_delay\_seconds}.
$$

If a cycle has no fills, the second term in $B_c$ is omitted, so
$B_c=P_c+\texttt{min\_resting\_seconds}$. Fills, cancels, and active-quantity
events after `horizon_seconds` are generated but excluded from the plotted/cost
window.

## Tail Propagator

Write the fitted constant-sensitivity propagator as:

$$
G(u)
= \kappa_s + \sum_{i=1}^m w_i e^{-\beta_i u}
= \kappa_s\,\xi(u),
$$

with

$$
\xi(u)=1+\sum_{i=1}^m a_i e^{-\beta_i u},
\qquad a_i=\frac{w_i}{\kappa_s}.
$$

The config mapping is:

- $\kappa_s=\texttt{propagator\_kappa}$
- $w_i=\texttt{propagator\_weights}_i$
- $\beta_i=\texttt{propagator\_beta}_i$

The affine queue mean-reversion scale used in the continuation term is:

$$
C_\lambda = \texttt{b\_c}-\texttt{b\_l}.
$$

As in the reference-paper notation, the normalized propagator defines the
history-to-future response kernel:

$$
r(a)=-\xi'(a)
=\sum_{i=1}^m \beta_i a_i e^{-\beta_i a}.
$$

The passive continuation kernel discounts this future response by the queue
reversion:

$$
K_C(a)
=\int_0^\infty e^{-C_\lambda u}r(a+u)\,du
=\sum_{i=1}^m \eta_i e^{-\beta_i a},
$$

$$
\eta_i =
\frac{\beta_i a_i}{\beta_i+C_\lambda}
=
\frac{\beta_i w_i}{\kappa_s(\beta_i+C_\lambda)}.
$$

Let $N$ be the empirical consuming-side market-order process selected by
`market_side`. The tail term is:

$$
\mathcal I_t
= \zeta + \int_0^t K_C(t-s)\,dN_s,
$$

with $\zeta=\texttt{propagator\_tail\_zeta}$. In signed single-queue notation:

$$
U_t=s(\bar q_t-q_t),
$$

where $s=-1$ for `q_b` and $s=+1$ for `q_a`. The passive impact has the same
realized-plus-continuation structure as the reference passive-impact formula:

$$
MI(t)
= \kappa_1\int_0^t U_s\,dN_s
+ \kappa_1 U_t \mathcal I_t,
$$

where $\kappa_1=\texttt{propagator\_gamma}$.

At market times $t_k$, the implementation evaluates the same formula with
exponential states:

$$
S_{i,k}
=e^{-\beta_i(t_k-t_{k-1})}S_{i,k-1}+\eta_i,
\qquad S_{i,0^-}=0,
$$

$$
\mathcal I_k=\zeta+\sum_{i=1}^m S_{i,k},
$$

$$
MI_k
=\kappa_1\left(\sum_{r=1}^k U_r+U_k\mathcal I_k\right).
$$

The native helper computes the unsigned one-queue contribution; Python applies
the sign convention through $U_k=s(\bar q_k-q_k)$.

## Cost

For a fill at time $\tau_j$, the runner uses the left-limit impact:

$$
k(j)=\max\{k:t_k<\tau_j\}.
$$

The fill cost jump and running impact cost are:

$$
\Delta C_j=\texttt{qty}_j\,\Delta P_{k(j)},
$$

$$
C(t)=\sum_{\tau_j\le t}\Delta C_j.
$$

`output_dir` receives the CSV/JSON outputs for these paths. `image_dir`
receives the PNG figures. `aggregated_path` is the processed parquet containing
the empirical $\bar q_k$, event times, event sides, and quantities.
