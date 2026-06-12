# PAPER_ANALYSIS

## Experimental Setting

All experiments are based on the real `510957.csv` cellular panel in the project workspace. The dataset contains 33 cells, 3,151 hourly timestamps, 48 KPI/business columns and 137 alarm/fault columns, of which 11 alarm columns are active in this slice. Windows crossing the detected long time gap are removed. Forecasting uses real future KPI values; degradation classification uses weak KPI-threshold labels fitted from the training split; root-cause ranking uses weak alarm-lift labels and is not expert-verified.

## Forecasting Results

For the 1-hour horizon, `CET-STGL-sklearn` obtains primary-KPI MAE/RMSE of 2.183/4.635, compared with 2.219/4.698 for the Ridge/Logistic feature baseline. Event-only methods are substantially weaker for forecasting: `Hawkes-only` has 1-hour primary MAE/RMSE of 4.578/7.981. This suggests that sparse alarm events alone do not contain enough information to reconstruct continuous KPI trajectories; KPI temporal context remains the dominant signal.

Best forecasting rows by primary MAE are:

| model_label | horizon | primary_mae | primary_rmse | primary_mape_floor1_pct |
| --- | --- | --- | --- | --- |
| CET w/o alarm tokens | 1 | 2.1569 | 4.6016 | 42.4111 |
| CET w/o alarm tokens | 3 | 2.4904 | 4.9744 | 46.8069 |
| CET w/o alarm tokens | 6 | 3.1447 | 5.6052 | 56.8431 |
| CET w/o alarm tokens | 12 | 2.5147 | 5.0508 | 47.2830 |


The ablation results should be interpreted carefully. In several horizons, removing alarm tokens or using the no-alarm baseline does not degrade forecasting and sometimes improves averaged forecasting error. This is plausible because only about 1.62% of rows contain active alarm events, whereas many forecasted KPI/business values are governed by regular temporal patterns and load dynamics. Therefore, the paper should emphasize event-conditioned analyses and degradation/ranking tasks rather than claiming universal forecasting gains from alarms.

## Degradation Classification

At the 1-hour horizon, `CET-STGL-sklearn` reaches AUPRC/F1 of 0.466/0.479. The Ridge/Logistic baseline obtains AUPRC/F1 of 0.471/0.477, while the no-alarm Ridge/Logistic variant obtains 0.476/0.480. Thus, under the current weak degradation labels, alarm tokens do not provide a uniformly significant classification gain. This is an important negative/nuanced result: the weak labels are primarily defined by future KPI thresholds, so lagged KPI dynamics can already be highly predictive.

Best classification rows by AUPRC are:

| model_label | horizon | auprc | f1 | precision | recall | auroc |
| --- | --- | --- | --- | --- | --- | --- |
| Ridge/Logistic w/o alarms | 1 | 0.4756 | 0.4801 | 0.3956 | 0.6104 | 0.7971 |
| CET w/o alarm tokens | 3 | 0.3990 | 0.3481 | 0.5197 | 0.2617 | 0.7587 |
| CET w/o alarm tokens | 6 | 0.3344 | 0.3313 | 0.1996 | 0.9750 | 0.7167 |
| Ridge/Logistic w/o alarms | 12 | 0.3959 | 0.3348 | 0.2018 | 0.9828 | 0.7420 |


Hawkes-only and event-attention-only baselines underperform KPI-history baselines in AUPRC and AUROC. Their limitation is structural: active alarms are sparse and many KPI degradation events are weakly labeled from KPI thresholds rather than confirmed operational incidents. Event-only models are therefore useful as root-cause candidate generators and event sensitivity probes, but not sufficient as standalone KPI degradation predictors in this data slice.

## CET-STGL Proxy and Ablation Interpretation

`CET-STGL-sklearn` should be described as a runnable proxy implementation of the proposed feature blocks: KPI patch tokens, alarm event features, train-only alarm-lift event influence features, time/cell encodings and an inferred cross-cell graph. It is not the final neural spatio-temporal graph encoder. The current result supports the feasibility of the data pipeline and ablation framework. It does not yet prove that the full neural CET-STGL architecture outperforms all baselines.

The ablation trends indicate that tokenization and cross-cell context can affect classification and forecasting, but their gains depend on horizon and metric. Removing Hawkes/event intensity has little effect in several rows, implying that the present decayed-intensity signal is either redundant with rolling alarm counts or too sparse to dominate. This motivates a stronger event-influence learning module in the next neural implementation.

## Weak Root-Cause Ranking

Root-cause ranking results are based on weak labels derived from train-only alarm-to-degradation lift and recent event intensity. Some models score highly because they use the same alarm-lift signal used to define the weak labels. These results should be framed as weak-label consistency and sanity checking, not as expert-verified root-cause localization. The correct paper wording is: "the ranking head recovers weak root-cause candidates induced by temporal alarm-degradation association." Avoid the phrase "ground-truth root cause" unless expert annotations are later added.

## Neural Baseline Status

The current runtime does not include PyTorch, so LSTM, TCN, Informer, Autoformer, STGCN and GAT were not executed. They are recorded in `model_status.csv` as `not_run_backend_missing`; no results are fabricated for them. The sklearn baselines and CET-STGL proxy are fully runnable and their outputs are saved under `experiments/causal_event_token_stgl/results/`.

## Limitations for the Paper

1. The dataset is a private project slice, not a public benchmark.
2. Degradation and root-cause labels are weak labels derived from observed KPI/alarm patterns.
3. The event influence graph is causal-inspired / causal discovery-based; the observational setup does not justify strict causal claims.
4. The current CET-STGL result is a sklearn proxy over the proposed representation blocks, not the final neural graph encoder.
5. Alarm sparsity is severe: only 11 of 137 alarm/fault columns are active and only about 1.62% of rows contain active alarms.
6. Do not claim telecom foundation-model training.

## Suggested Experimental Paragraph for the Paper

"We evaluate alarm-driven KPI degradation prediction on a private cellular cell-time panel extracted from `510957.csv`, containing 33 cells, 3,151 hourly timestamps, 48 KPI/business indicators and 137 alarm/fault indicators. Since no expert root-cause annotations are available, degradation labels are weakly derived from training-split KPI thresholds, and root-cause ranking labels are weak candidates induced by lagged alarm-to-degradation lift. Across 1/3/6/12-hour horizons, KPI-history baselines remain strong, while event-only baselines are weaker due to sparse alarms. The CET-STGL proxy validates the proposed representation pipeline and enables ablation over alarm tokens, event intensity, tokenization and inferred cross-cell graph features. The current results should be interpreted as evidence for the feasibility of causal-inspired event-token modeling under weak supervision, not as proof of strict causal discovery or telecom foundation-model training."
