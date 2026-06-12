# REVIEWER_RISK_NOTES

This file anticipates reviewer concerns and recommends honest responses. The goal is to make the paper credible, not to over-defend unsupported claims.

## Risk 1. Data Are Private and Not Publicly Releasable

Likely reviewer concern:

> The dataset is private, so results may not be reproducible by external researchers.

Honest response:

- Acknowledge that the data are an enterprise cellular operations slice and cannot be released in raw form.
- Provide detailed schema, statistics, split protocol, weak-label definitions and code-level feature construction.
- Avoid calling the work a benchmark.
- State that the contribution is a technical case study and method formulation on real enterprise structured operations data.

Suggested manuscript wording:

> Due to confidentiality constraints, raw cellular operations data cannot be released. We therefore report detailed schema statistics, split rules, feature definitions and weak-label construction to support methodological transparency.

## Risk 2. Neural Baselines Were Not Run

Likely reviewer concern:

> LSTM, TCN, Informer, Autoformer, STGCN and GAT are listed in the plan but not evaluated.

Honest response:

- State explicitly that the current runtime lacks PyTorch.
- Do not include these models in result comparisons.
- Put `model_status.csv` in the experiment section.
- Describe current experiments as sklearn baselines plus a runnable CET-STGL proxy.

Suggested manuscript wording:

> Neural baselines are not reported because the current runtime did not include the required PyTorch backend. We include this as an experimental limitation rather than fabricating or approximating neural results.

## Risk 3. Alarm Sparsity Makes Gains Unstable

Likely reviewer concern:

> Only 11 of 137 alarm columns are active and active alarms appear in about 1.62% of rows; why should alarm tokens help?

Honest response:

- Agree that alarm sparsity is a central empirical finding.
- Report that event-only/Hawkes-only are weak.
- Avoid claiming uniform alarm-token gains.
- Reframe event modeling as useful for weak ranking and event-conditioned probing, not as a universal forecasting improvement.

Suggested manuscript wording:

> The sparse event regime is precisely why event-only models are insufficient in our experiments. KPI history remains the dominant signal for many weak degradation labels, while event features provide a structured way to construct and inspect weak root-cause candidates.

## Risk 4. Weak Root-Cause Ranking Is Not True Root-Cause Validation

Likely reviewer concern:

> The ranking labels are generated from the same alarm-degradation association used by some methods.

Honest response:

- Acknowledge that ranking evaluates weak-label consistency only.
- Do not use "ground truth root cause".
- Avoid treating perfect Hit@K as operational proof.
- State that expert work-order labels are future work.

Suggested manuscript wording:

> Because expert root-cause annotations are unavailable, ranking is evaluated against weak candidates induced by train-only alarm-to-degradation lift. These results measure consistency with a proxy signal rather than expert-verified fault localization.

## Risk 5. Causal Language Is Too Strong

Likely reviewer concern:

> The paper claims causality from observational data.

Honest response:

- Use "causal-inspired" throughout.
- Explain that the graph is based on temporal precedence, event intensity and train-only association.
- Do not claim identification, interventions or confirmed causal propagation.

Suggested manuscript wording:

> We use causal-inspired to denote a structured event influence prior motivated by temporal precedence and alarm-degradation association. The observational data do not support strict causal identification.

## Risk 6. Single File / Single Region / Single Time Period

Likely reviewer concern:

> Results may not generalize beyond one region or time period.

Honest response:

- Acknowledge limited external validity.
- Describe as an enterprise case study.
- Suggest multi-region, longer-period validation as future work.

Suggested manuscript wording:

> The current evaluation is limited to one private cellular data slice. Additional regions, longer periods and different vendors/operators are needed to assess generalization.

## Risk 7. CET-STGL-sklearn Is Not the Full Neural Architecture

Likely reviewer concern:

> The method section describes a spatio-temporal graph learning model, but experiments run a sklearn proxy.

Honest response:

- Clearly distinguish proposed architecture from runnable proxy.
- Say the current experiments validate the representation blocks and data protocol.
- Do not claim the full neural graph encoder has been trained.

Suggested manuscript wording:

> The current implementation is a runnable proxy over the proposed representation blocks. It validates the data pipeline, weak-label construction and ablation logic, while the full neural spatio-temporal graph encoder remains future implementation work.

## Risk 8. No-Alarm Ablation Sometimes Performs Best

Likely reviewer concern:

> If removing alarms improves metrics, why is this an alarm-driven paper?

Honest response:

- Explain weak labels are KPI-threshold based and strongly predictable from KPI history.
- Alarm events are sparse and may not cover all KPI degradations.
- Keep the title/motivation but avoid claiming alarm tokens improve aggregate metrics.
- Focus on representation and proxy ranking under sparse event supervision.

Suggested manuscript wording:

> The no-alarm ablation being competitive reveals that KPI dynamics dominate aggregate weak-label prediction in this slice. Rather than obscuring this result, we use it to motivate event-conditioned evaluation and future expert-labeled root-cause validation.

## Risk 9. MAPE Is Unstable for KPI Data

Likely reviewer concern:

> MAPE can be misleading when true values are zero.

Honest response:

- Use `MAPE_floor1_pct`.
- Explain denominator floor.
- Report MAE/RMSE as primary metrics.

Suggested manuscript wording:

> Because several KPI/business values are zero or near zero, we report MAPE with denominator floor 1.0 and use MAE/RMSE as the primary forecasting metrics.

## Risk 10. Overly Broad Related Work

Likely reviewer concern:

> The paper reads like another telecom LLM/foundation-model survey.

Honest response:

- Keep related work short.
- Center technical contribution on structured operations data.
- Do not use foundation-model framing.

Suggested manuscript wording:

> This work does not train a telecom foundation model; it studies structured event-token representations for KPI degradation prediction on enterprise cellular operations data.

