# CLAIM_SAFETY_CHECK

Scope: migrated CET-STGL manuscript in the Hyper-IIoT IEEEtran-style template.

| Check item | Status | Evidence / location |
|---|---|---|
| Foundation model claim | PASS | The manuscript explicitly says CET-STGL is not a foundation-model paper. |
| Benchmark claim | PASS | Data are described as private cellular operational records, not a public benchmark. |
| Strict causality claim | PASS | Event graph is described as causal-inspired / causal discovery-based / event influence prior, not confirmed causality. |
| True root-cause validation claim | PASS | Ranking is described as weak-label/proxy consistency, not expert-confirmed root-cause localization. |
| Neural baseline fabricated claim | PASS | LSTM, TCN, Informer, Autoformer, STGCN and GAT are marked not run because PyTorch is unavailable. |
| sklearn proxy boundary | PASS | The current runnable implementation is described as an sklearn proxy, not a completed neural CET-STGL backend. |
| Alarm-token overclaim | PASS | Results and Discussion state that no-alarm ablations can be competitive or better, so alarm tokens do not uniformly improve performance. |
| SOTA / excessive improvement overclaim | PASS | The manuscript uses calibrated language such as competitive and restrained claim, not excessive superiority language. |
| Data availability statement | PASS | Data Availability and Reproducibility states that the raw enterprise data cannot be publicly released in its current form. |

## Search Notes

Sensitive terms were checked across `main.tex`, `sections/`, `tables/`, `appendix/`, and `FIGURE_CAPTIONS.md`. Hits occur in boundary-setting or negative-claim contexts.

## Preserved Required Boundaries

- Current runnable implementation is an sklearn proxy.
- PyTorch backend is unavailable; neural baselines are not run.
- Ranking is weak-label/proxy consistency only.
- Event graph is causal-inspired / causal discovery-based, not confirmed causality.
- Data are private cellular operational records, not a public benchmark.
