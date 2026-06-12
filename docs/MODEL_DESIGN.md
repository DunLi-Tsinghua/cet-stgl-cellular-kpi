# MODEL_DESIGN

Model name for draft: **CET-STGL**: Causal Event-Token Spatio-Temporal Graph Learning.

The model is a technical architecture for sparse alarm-driven KPI degradation prediction. It is not a telecom foundation model.

## Inputs

For each cell `c`, time `t`, lookback window `L`, and horizon `h`:

- KPI matrix `K_{c,t-L+1:t}` with selected KPI columns
- Alarm matrix `E_{c,t-L+1:t}` with non-constant alarm/fault columns
- Cell ID and optional inferred neighbor cells
- Time features: hour-of-day, day-of-week, absolute/relative position

## Token Types

### KPI Patch Token

Continuous KPI history is split into fixed-length patches.

For KPI `m` and patch `p`:

- raw values in the patch
- patch statistics: mean, std, min, max, slope
- KPI ID embedding
- cell embedding
- temporal position embedding

Purpose:

- preserve local time dynamics
- reduce sequence length
- separate KPI identity from cell/time context

Default patch lengths:

- `P = 3` hours
- `P = 6` hours as sensitivity

### Alarm Event Token

Each non-zero alarm occurrence becomes an event token:

- alarm type ID
- count/value
- cell ID
- event timestamp
- parsed domain/layer suffix if available, for example `ENODEB_L1`, `RRU_L1`, `CELL_L1`
- time-since-event
- event intensity feature from Hawkes-style decay

Sparse all-zero alarm columns are excluded from training-time token construction but retained in schema metadata.

### Cell/Time Positional Encoding

Cell encoding:

- learned cell ID embedding
- optional site-prefix embedding inferred from cell ID
- graph positional feature from inferred cell graph degree/centrality

Time encoding:

- hour-of-day sine/cosine
- day-of-week sine/cosine
- relative position inside lookback window
- gap mask so windows crossing the 21-day gap are removed

## Dynamic Event Influence Graph

The graph is **causal-inspired**, not strictly causal.

Nodes:

- cell nodes
- alarm type nodes
- optional KPI nodes for interpretability

Edges:

- alarm type to cell/KPI degradation risk
- alarm type to alarm type, if temporal precedence is stable
- cell to cell, inferred from same-site heuristic and/or train-only KPI correlation

Edge weights:

`w_{a -> d}^{(h)} = f(lagged_lift, decayed_event_count, temporal_precedence, optional_graph_prior)`

Default train-only lagged lift:

`lift(a, h) = log((P(y_deg_h=1 | alarm_a in lookback) + eps) / (P(y_deg_h=1) + eps))`

Hawkes-style intensity:

`lambda_a(t) = mu_a + sum_{tau<t} alpha_a exp(-(t - tau) / eta)`

The first implementation may use a simplified decayed-count intensity. A later version can estimate event-specific `alpha_a` or multi-type excitation parameters.

## Encoder

### KPI Temporal Encoder

Encodes KPI patch tokens using one of:

- temporal convolution
- Transformer encoder
- gated recurrent block

Output:

- per-cell temporal state `H_kpi[c,t]`

### Event Encoder

Encodes alarm event tokens using:

- alarm type embedding
- time decay/intensity embedding
- event-to-cell cross-attention

Output:

- event state `H_event[c,t]`

### Spatio-Temporal Graph Encoder

Combines KPI states, event states, and graph messages.

Message passing:

- cross-cell graph messages among cell states
- event influence messages from alarm nodes to cell/KPI states
- temporal update across windows

Recommended update:

`H_c^l = TemporalBlock(H_c^{l-1}) + GraphMessage(A_cell, H^{l-1}) + EventMessage(A_event(t), H_event)`

Dynamic graph weights are recomputed or updated per window using event intensities and train-learned influence priors.

## Heads

### Forecasting Head

Multi-horizon regression:

- output `Y_hat_kpi[c,t+h]` for `h in {1,3,6,12}`
- one shared trunk, horizon-specific projection layers

Loss:

- Huber or MSE
- optional KPI-wise normalization

### Degradation Classification Head

Binary risk score:

- output `p_deg[c,t+h]`
- threshold-free evaluation with AUPRC/AUROC

Loss:

- weighted BCE or focal loss
- class weights fitted on train split

### Root-Cause Ranking Head

Ranks candidate alarm types for a future degradation event.

Candidate set:

- alarms observed in same cell and inferred neighbor cells within recent window

Score:

`s(a,c,t,h) = g(H_event_a, H_cell_c, A_event[a,c,t], lambda_a(t))`

Loss:

- pairwise ranking loss against weak top-k labels
- or listwise cross entropy over weak candidate distribution

Evaluation:

- MRR, Hits@k, nDCG@k using weak labels

## Training Objective

`L = L_forecast + beta * L_degradation + gamma * L_ranking + rho * L_graph_sparse`

Suggested starting weights:

- `beta = 1.0`
- `gamma = 0.2`
- `rho = 1e-4`

For the first working version, train forecasting and degradation heads first; add ranking after weak-label quality is inspected.

## Interpretability Outputs

The paper should expose:

- top event tokens contributing to degradation risk
- dynamic event influence graph snapshots
- cross-cell propagation heatmap
- event-conditioned forecast examples

These are explanations of model association and learned influence, not proof of physical causality.

## Implementation Notes

Current local environment lacks PyTorch. The first committed code skeleton therefore provides:

- data loading and panel validation
- token/feature preparation interfaces
- weak label generation
- sklearn smoke baselines
- model interface placeholders

When PyTorch is available, implement CET-STGL under the same data interface.
