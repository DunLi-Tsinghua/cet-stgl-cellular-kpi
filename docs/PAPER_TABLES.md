# PAPER_TABLES

All runnable rows are computed from the private `510957.csv` experiment outputs. Classification and ranking use weak labels defined from this dataset. Neural baselines marked `not_run_backend_missing` were not executed because the current runtime lacks PyTorch.

MAPE is reported as `MAPE_floor1_pct`, using denominator `max(abs(y_true), 1.0)` because several KPI/business columns contain zero or near-zero true values.

## Table 1. KPI Forecasting

| model_label | horizon | primary_mae | primary_rmse | primary_mape_floor1_pct | all_kpi_mae | all_kpi_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| CET w/o alarm tokens | 1 | 2.1569 | 4.6016 | 42.4111 | 30.1717 | 53.1923 |
| CET w/o cross-cell graph | 1 | 2.1791 | 4.6351 | 42.7362 | 31.1928 | 62.4168 |
| CET w/o dynamic event graph | 1 | 2.1831 | 4.6351 | 42.6425 | 31.2506 | 62.1414 |
| CET w/o Hawkes intensity | 1 | 2.1831 | 4.6351 | 42.6425 | 31.2506 | 62.1414 |
| CET w/o tokenization | 1 | 2.2244 | 4.7059 | 43.4812 | 31.6755 | 61.3319 |
| CET-STGL-sklearn | 1 | 2.1830 | 4.6351 | 42.6425 | 31.2510 | 62.1427 |
| Event-attention-only | 1 | 4.6305 | 8.2757 | 82.5194 | 163.9535 | 271.1688 |
| Hawkes-only | 1 | 4.5779 | 7.9814 | 82.2610 | 160.0211 | 227.2084 |
| Ridge/Logistic | 1 | 2.2191 | 4.6984 | 43.5993 | 31.4426 | 58.4626 |
| Ridge/Logistic w/o alarms | 1 | 2.2034 | 4.6712 | 43.3820 | 30.6931 | 53.7885 |
| CET w/o alarm tokens | 3 | 2.4904 | 4.9744 | 46.8069 | 39.4850 | 67.8885 |
| CET w/o cross-cell graph | 3 | 2.5354 | 5.0786 | 47.2707 | 42.1875 | 103.4692 |
| CET w/o dynamic event graph | 3 | 2.5368 | 5.0829 | 47.1621 | 42.1587 | 102.7291 |
| CET w/o Hawkes intensity | 3 | 2.5368 | 5.0829 | 47.1621 | 42.1587 | 102.7291 |
| CET w/o tokenization | 3 | 3.0549 | 5.6840 | 56.1056 | 50.7988 | 101.8653 |
| CET-STGL-sklearn | 3 | 2.5367 | 5.0829 | 47.1624 | 42.1585 | 102.7298 |
| Event-attention-only | 3 | 4.6346 | 8.3256 | 82.7764 | 164.6453 | 289.4013 |
| Hawkes-only | 3 | 4.5809 | 8.0385 | 82.5149 | 159.7122 | 227.2520 |
| Ridge/Logistic | 3 | 3.0709 | 5.6842 | 56.3815 | 50.7191 | 93.6261 |
| Ridge/Logistic w/o alarms | 3 | 3.0361 | 5.5858 | 56.0146 | 48.9354 | 78.7454 |
| CET w/o alarm tokens | 6 | 3.1447 | 5.6052 | 56.8431 | 50.7871 | 82.1649 |
| CET w/o cross-cell graph | 6 | 3.1904 | 5.8644 | 57.2840 | 54.4509 | 133.8707 |
| CET w/o dynamic event graph | 6 | 3.1970 | 5.8527 | 57.2636 | 54.4014 | 132.2650 |
| CET w/o Hawkes intensity | 6 | 3.1970 | 5.8527 | 57.2636 | 54.4014 | 132.2650 |
| CET w/o tokenization | 6 | 3.3631 | 6.0530 | 61.5916 | 65.4517 | 131.9830 |
| CET-STGL-sklearn | 6 | 3.1971 | 5.8527 | 57.2638 | 54.4004 | 132.2636 |
| Event-attention-only | 6 | 4.6287 | 8.2769 | 83.1170 | 164.8499 | 295.9310 |
| Hawkes-only | 6 | 4.5785 | 7.8735 | 82.8707 | 159.5017 | 225.5720 |
| Ridge/Logistic | 6 | 3.3380 | 5.9402 | 61.5904 | 65.1005 | 118.3171 |
| Ridge/Logistic w/o alarms | 6 | 3.3177 | 5.8838 | 61.2722 | 62.6208 | 97.5729 |
| CET w/o alarm tokens | 12 | 2.5147 | 5.0508 | 47.2830 | 42.0699 | 74.8636 |
| CET w/o cross-cell graph | 12 | 2.6040 | 5.4528 | 47.7833 | 46.2547 | 127.1682 |
| CET w/o dynamic event graph | 12 | 2.6041 | 5.4421 | 47.6820 | 46.4562 | 126.0803 |
| CET w/o Hawkes intensity | 12 | 2.6041 | 5.4421 | 47.6820 | 46.4562 | 126.0803 |
| CET w/o tokenization | 12 | 2.6796 | 5.5472 | 49.0619 | 47.8638 | 124.7888 |
| CET-STGL-sklearn | 12 | 2.6041 | 5.4421 | 47.6819 | 46.4562 | 126.0802 |
| Event-attention-only | 12 | 4.6714 | 8.6216 | 83.3406 | 164.9266 | 287.0291 |
| Hawkes-only | 12 | 4.5719 | 7.8884 | 83.0608 | 160.6519 | 238.1456 |
| Ridge/Logistic | 12 | 2.6400 | 5.3913 | 49.1403 | 47.3654 | 122.6165 |
| Ridge/Logistic w/o alarms | 12 | 2.5928 | 5.1347 | 48.8058 | 43.2913 | 76.8817 |

## Table 2. KPI Degradation Classification

| model_label | horizon | auprc | f1 | precision | recall | auroc |
| --- | --- | --- | --- | --- | --- | --- |
| CET w/o alarm tokens | 1 | 0.4702 | 0.3609 | 0.2215 | 0.9746 | 0.7975 |
| CET w/o cross-cell graph | 1 | 0.4482 | 0.3980 | 0.5244 | 0.3207 | 0.7827 |
| CET w/o dynamic event graph | 1 | 0.4662 | 0.4784 | 0.3948 | 0.6070 | 0.7943 |
| CET w/o Hawkes intensity | 1 | 0.4662 | 0.4785 | 0.3947 | 0.6074 | 0.7944 |
| CET w/o tokenization | 1 | 0.4578 | 0.4583 | 0.4634 | 0.4534 | 0.7892 |
| CET-STGL-sklearn | 1 | 0.4663 | 0.4787 | 0.3949 | 0.6078 | 0.7944 |
| Event-attention-only | 1 | 0.2573 | 0.3310 | 0.2257 | 0.6201 | 0.6358 |
| Hawkes-only | 1 | 0.2465 | 0.3274 | 0.2181 | 0.6562 | 0.6314 |
| Ridge/Logistic | 1 | 0.4709 | 0.4774 | 0.4182 | 0.5563 | 0.7946 |
| Ridge/Logistic w/o alarms | 1 | 0.4756 | 0.4801 | 0.3956 | 0.6104 | 0.7971 |
| CET w/o alarm tokens | 3 | 0.3990 | 0.3481 | 0.5197 | 0.2617 | 0.7587 |
| CET w/o cross-cell graph | 3 | 0.3870 | 0.4010 | 0.2700 | 0.7787 | 0.7527 |
| CET w/o dynamic event graph | 3 | 0.3760 | 0.1475 | 0.5751 | 0.0846 | 0.7430 |
| CET w/o Hawkes intensity | 3 | 0.3759 | 0.1476 | 0.5780 | 0.0846 | 0.7429 |
| CET w/o tokenization | 3 | 0.3419 | 0.3121 | 0.1849 | 0.9985 | 0.7236 |
| CET-STGL-sklearn | 3 | 0.3752 | 0.1468 | 0.5696 | 0.0842 | 0.7427 |
| Event-attention-only | 3 | 0.2494 | 0.3310 | 0.2183 | 0.6836 | 0.6328 |
| Hawkes-only | 3 | 0.2331 | 0.3265 | 0.2181 | 0.6492 | 0.6230 |
| Ridge/Logistic | 3 | 0.3409 | 0.3201 | 0.1907 | 0.9959 | 0.7231 |
| Ridge/Logistic w/o alarms | 3 | 0.3365 | 0.2118 | 0.4454 | 0.1389 | 0.7256 |
| CET w/o alarm tokens | 6 | 0.3344 | 0.3313 | 0.1996 | 0.9750 | 0.7167 |
| CET w/o cross-cell graph | 6 | 0.3195 | 0.3318 | 0.1997 | 0.9792 | 0.7178 |
| CET w/o dynamic event graph | 6 | 0.3203 | 0.3894 | 0.2853 | 0.6129 | 0.7131 |
| CET w/o Hawkes intensity | 6 | 0.3203 | 0.3892 | 0.2849 | 0.6136 | 0.7131 |
| CET w/o tokenization | 6 | 0.3026 | 0.3033 | 0.3729 | 0.2556 | 0.6842 |
| CET-STGL-sklearn | 6 | 0.3181 | 0.3887 | 0.2835 | 0.6181 | 0.7127 |
| Event-attention-only | 6 | 0.2146 | 0.3523 | 0.2455 | 0.6234 | 0.6165 |
| Hawkes-only | 6 | 0.2940 | 0.3183 | 0.2152 | 0.6113 | 0.6326 |
| Ridge/Logistic | 6 | 0.3087 | 0.3152 | 0.4025 | 0.2590 | 0.6909 |
| Ridge/Logistic w/o alarms | 6 | 0.2840 | 0.3625 | 0.2448 | 0.6983 | 0.6830 |
| CET w/o alarm tokens | 12 | 0.3769 | 0.3337 | 0.2007 | 0.9885 | 0.7526 |
| CET w/o cross-cell graph | 12 | 0.3777 | 0.3213 | 0.1915 | 0.9981 | 0.7499 |
| CET w/o dynamic event graph | 12 | 0.3728 | 0.3243 | 0.1937 | 0.9966 | 0.7463 |
| CET w/o Hawkes intensity | 12 | 0.3729 | 0.3243 | 0.1937 | 0.9966 | 0.7463 |
| CET w/o tokenization | 12 | 0.3641 | 0.3633 | 0.4503 | 0.3045 | 0.7382 |
| CET-STGL-sklearn | 12 | 0.3732 | 0.3240 | 0.1934 | 0.9966 | 0.7465 |
| Event-attention-only | 12 | 0.2405 | 0.3396 | 0.2383 | 0.5904 | 0.6332 |
| Hawkes-only | 12 | 0.2467 | 0.3221 | 0.2190 | 0.6083 | 0.6107 |
| Ridge/Logistic | 12 | 0.3806 | 0.3339 | 0.2011 | 0.9828 | 0.7413 |
| Ridge/Logistic w/o alarms | 12 | 0.3959 | 0.3348 | 0.2018 | 0.9828 | 0.7420 |

## Table 3. Weak Root-Cause Ranking

| model_label | horizon | ranking_samples | hit_at_1 | hit_at_3 | hit_at_5 | mrr | ndcg_at_5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CET w/o alarm tokens | 1 | 0.0000 |  |  |  |  |  |
| CET w/o cross-cell graph | 1 | 2682.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CET w/o dynamic event graph | 1 | 2682.0000 | 0.8084 | 0.9952 | 0.9959 | 0.8948 | 0.9204 |
| CET w/o Hawkes intensity | 1 | 611.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CET w/o tokenization | 1 | 2682.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CET-STGL-sklearn | 1 | 2682.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Event-attention-only | 1 | 611.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Hawkes-only | 1 | 2682.0000 | 0.8084 | 0.9952 | 0.9959 | 0.8948 | 0.9204 |
| Ridge/Logistic | 1 | 611.0000 | 0.9296 | 1.0000 | 1.0000 | 0.9604 | 0.9706 |
| Ridge/Logistic w/o alarms | 1 | 0.0000 |  |  |  |  |  |
| CET w/o alarm tokens | 3 | 0.0000 |  |  |  |  |  |
| CET w/o cross-cell graph | 3 | 2671.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CET w/o dynamic event graph | 3 | 2671.0000 | 0.8068 | 0.9944 | 0.9951 | 0.8937 | 0.9193 |
| CET w/o Hawkes intensity | 3 | 609.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CET w/o tokenization | 3 | 2671.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CET-STGL-sklearn | 3 | 2671.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Event-attention-only | 3 | 609.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Hawkes-only | 3 | 2671.0000 | 0.8068 | 0.9944 | 0.9951 | 0.8937 | 0.9193 |
| Ridge/Logistic | 3 | 609.0000 | 0.9228 | 1.0000 | 1.0000 | 0.9576 | 0.9685 |
| Ridge/Logistic w/o alarms | 3 | 0.0000 |  |  |  |  |  |
| CET w/o alarm tokens | 6 | 0.0000 |  |  |  |  |  |
| CET w/o cross-cell graph | 6 | 2645.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CET w/o dynamic event graph | 6 | 2645.0000 | 0.8049 | 0.9940 | 0.9947 | 0.8926 | 0.9183 |
| CET w/o Hawkes intensity | 6 | 614.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CET w/o tokenization | 6 | 2645.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CET-STGL-sklearn | 6 | 2645.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Event-attention-only | 6 | 614.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Hawkes-only | 6 | 2645.0000 | 0.8049 | 0.9940 | 0.9947 | 0.8926 | 0.9183 |
| Ridge/Logistic | 6 | 614.0000 | 0.9169 | 1.0000 | 1.0000 | 0.9555 | 0.9670 |
| Ridge/Logistic w/o alarms | 6 | 0.0000 |  |  |  |  |  |
| CET w/o alarm tokens | 12 | 0.0000 |  |  |  |  |  |
| CET w/o cross-cell graph | 12 | 2617.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CET w/o dynamic event graph | 12 | 2617.0000 | 0.8040 | 0.9935 | 0.9943 | 0.8920 | 0.9177 |
| CET w/o Hawkes intensity | 12 | 649.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CET w/o tokenization | 12 | 2617.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CET-STGL-sklearn | 12 | 2617.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Event-attention-only | 12 | 649.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Hawkes-only | 12 | 2617.0000 | 0.8040 | 0.9935 | 0.9943 | 0.8920 | 0.9177 |
| Ridge/Logistic | 12 | 649.0000 | 0.9276 | 1.0000 | 1.0000 | 0.9617 | 0.9717 |
| Ridge/Logistic w/o alarms | 12 | 0.0000 |  |  |  |  |  |

## Table 4. Neural Baseline Status

| model | status | reason |
| --- | --- | --- |
| lstm | not_run_backend_missing | PyTorch is not installed in the current runtime; sklearn proxy and interfaces are runnable. |
| tcn | not_run_backend_missing | PyTorch is not installed in the current runtime; sklearn proxy and interfaces are runnable. |
| informer | not_run_backend_missing | PyTorch is not installed in the current runtime; sklearn proxy and interfaces are runnable. |
| autoformer | not_run_backend_missing | PyTorch is not installed in the current runtime; sklearn proxy and interfaces are runnable. |
| stgcn | not_run_backend_missing | PyTorch is not installed in the current runtime; sklearn proxy and interfaces are runnable. |
| gat | not_run_backend_missing | PyTorch is not installed in the current runtime; sklearn proxy and interfaces are runnable. |

## Best Runnable Rows by Horizon

Forecasting, lower `primary_mae` is better:

| model_label | horizon | primary_mae | primary_rmse | primary_mape_floor1_pct | all_kpi_mae | all_kpi_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| CET w/o alarm tokens | 1 | 2.1569 | 4.6016 | 42.4111 | 30.1717 | 53.1923 |
| CET w/o alarm tokens | 3 | 2.4904 | 4.9744 | 46.8069 | 39.4850 | 67.8885 |
| CET w/o alarm tokens | 6 | 3.1447 | 5.6052 | 56.8431 | 50.7871 | 82.1649 |
| CET w/o alarm tokens | 12 | 2.5147 | 5.0508 | 47.2830 | 42.0699 | 74.8636 |

Classification, higher AUPRC is better:

| model_label | horizon | auprc | f1 | precision | recall | auroc |
| --- | --- | --- | --- | --- | --- | --- |
| Ridge/Logistic w/o alarms | 1 | 0.4756 | 0.4801 | 0.3956 | 0.6104 | 0.7971 |
| CET w/o alarm tokens | 3 | 0.3990 | 0.3481 | 0.5197 | 0.2617 | 0.7587 |
| CET w/o alarm tokens | 6 | 0.3344 | 0.3313 | 0.1996 | 0.9750 | 0.7167 |
| Ridge/Logistic w/o alarms | 12 | 0.3959 | 0.3348 | 0.2018 | 0.9828 | 0.7420 |

Ranking, higher MRR is better; weak-label consistency only:

| model_label | horizon | ranking_samples | hit_at_1 | hit_at_3 | hit_at_5 | mrr | ndcg_at_5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CET w/o cross-cell graph | 1 | 2682.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CET w/o cross-cell graph | 3 | 2671.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CET w/o cross-cell graph | 6 | 2645.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CET w/o cross-cell graph | 12 | 2617.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Footnotes

- `CET-STGL-sklearn` is a runnable proxy implementation over the proposed feature blocks, not the final neural CET-STGL graph encoder.
- LSTM, TCN, Informer, Autoformer, STGCN and GAT are listed as non-run neural backends in `model_status.csv`; no metrics are fabricated for them.
- Root-cause ranking uses weak alarm-lift labels and should be described as weak-label consistency, not expert-verified root-cause localization.
- The proposed graph is causal-inspired / causal discovery-based; the observational data do not support strict causal claims.