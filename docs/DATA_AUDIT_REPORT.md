# DATA_AUDIT_REPORT

Project line: **Causal Event-Token Spatio-Temporal Graph Learning for Alarm-Driven KPI Degradation Prediction in Cellular Networks**

This audit is restricted to the current project assets. It does not assume public telecom data and does not claim a telecom foundation model.

## Data Source

Audited file pattern: `510957.csv`.

Duplicates observed in the workspace:

- `第一期\20240221会议\...\ref_data\510957.csv`
- `第一期\20250313会议\...\ref_data\510957.csv`
- `第二期\ref_data\510957.csv`
- `第二期\20240221会议\...\ref_data\510957.csv`

The largest copies have the same size, `86,027,324` bytes. The current runnable code discovers `510957.csv` automatically and prefers the largest copy. Future camera-ready experiments should pin one canonical copy and record its checksum.

## Basic Shape

- Rows: `103,983`
- Columns: `187`
- Numeric columns: `187`
- Missing values: none detected
- Constant columns: `126`, mainly alarm/fault columns with no non-zero occurrence in this slice
- Memory footprint in pandas: about `148.35 MB`

## Time and Cell Coverage

- Time column: `Time`, format `%Y%m%d%H%M%S`
- Cell column: `Cell`
- Time range: `2024-01-10 00:00:00` to `2024-06-10 06:00:00`
- Unique timestamps: `3,151`
- Nominal granularity: hourly
- Timestamp gaps: `3,149` hourly gaps plus one `21 days 01:00:00` gap
- Number of cells: `33`
- Rows per cell: exactly `3,151` rows for every cell

The data are therefore a balanced cell-time panel with one long time gap. Experiments must use chronological splits and must not randomly shuffle row-level samples across time.

## Field Groups

Column split rule used for the first audit:

- Metadata: `Time`, `Cell`
- Alarm/fault columns: names containing telecom alarm signatures such as `_RAN_WL_`, `DEFAULT_RAN_WL`, `TRANS_RAN_WL`, `POWER_RAN_WL`, `S1_RAN_WL`, `X2_RAN_WL`
- KPI/business columns: all remaining non-metadata columns

Counts:

- KPI/business-like columns: `48`
- Alarm/fault columns: `137`

Core KPI candidates include:

- `RRC Conn Users Avg`
- `Average TA`
- `DL Activated Users Avg`
- `ERAB Setup Success Rate(%)`
- `ERAB_Setup_Failure_Times`
- `RRC_Setup_Att_Times`
- `RRC Setup Success Rate(%)`
- `RRC_Setup_Failure_Times`
- `RRC ReEst Succ Rate(%)`
- `RRC_ReEst_Failure_Times`
- `S1Sig Setup Success Rate(%)`
- `Call Drop Rate(%)`
- `Call Drop Times`
- `CQI AVG`
- `LTE_UL PRB Utilizing Rate(%)`
- `L.UL.Interference.Avg(dBm)`
- `LTE_User UL Average Throughput(Mbps)`
- `LTE_DL PRB Utilizing Rate(%)`
- `LTE_User DL Average Throughput(Mbps)`
- `LTE_UL Traffic Volume(GB)`
- `LTE_DL Traffic Volume(GB)`
- handover and RACH counters/rates

## Alarm/Fault Sparsity

Alarm/fault columns: `137`.

Columns with any non-zero occurrence: `11`.

Rows with at least one non-zero alarm/fault column: `1,681`, about `1.62%`.

Top non-zero alarm/fault columns:

| Alarm/fault column | Non-zero rows |
|---|---:|
| `External Clock Reference Problem_DEFAULT_RAN_WL_ENODEB_L1` | 579 |
| `RF Unit Runtime Topology Error_DEFAULT_RAN_WL_RRU_L1` | 536 |
| `NE Is Disconnected_DEFAULT_RAN_WL_ENODEB_L3` | 516 |
| `4G-NE Down_DEFAULT_RAN_WL_ENODEB_L3` | 399 |
| `Cell Unavailable_TRANS_RAN_WL_CELL_L1` | 204 |
| `Remote Maintenance Link Failure_DEFAULT_RAN_WL_ENODEB_L1` | 204 |
| `BBU CPRI Interface Error_DEFAULT_RAN_WL_BBU_L1` | 81 |
| `Cell Unavailable_POWER_RAN_WL_CELL_L1` | 24 |
| `RF Unit Maintenance Link Failure_DEFAULT_RAN_WL_RRU_L1` | 24 |
| `Cell Unavailable_RF_RAN_WL_CELL_L1` | 6 |
| `RF Unit Clock Problem_DEFAULT_RAN_WL_RRU_L1` | 2 |

Implication: the alarm-driven part of the paper should emphasize sparse event influence, not dense event modeling.

## Target KPI Statistics

| KPI | Mean | Std | p05 | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| `Call Drop Rate(%)` | 0.137609 | 0.799348 | 0.000000 | 0.032234 | 0.528169 | 1.271308 |
| `Call Drop Times` | 2.947386 | 6.345737 | 0.000000 | 1.000000 | 11.000000 | 27.000000 |
| `RRC Setup Success Rate(%)` | 97.648051 | 14.285877 | 98.465473 | 99.934297 | 100.000000 | 100.000000 |
| `ERAB Setup Success Rate(%)` | 97.765184 | 13.817150 | 98.639804 | 99.854567 | 100.000000 | 100.000000 |
| `LTE_User DL Average Throughput(Mbps)` | 13.604742 | 9.824259 | 3.100685 | 10.992967 | 32.771322 | 44.213016 |
| `LTE_User UL Average Throughput(Mbps)` | 2.207431 | 1.295616 | 0.810215 | 1.968859 | 4.466001 | 6.503749 |
| `LTE_DL Traffic Volume(GB)` | 2.192346 | 1.994607 | 0.040777 | 1.613235 | 5.927501 | 8.158014 |
| `LTE_UL Traffic Volume(GB)` | 0.228477 | 0.227035 | 0.004851 | 0.162928 | 0.624765 | 0.929881 |
| `LTE_DL PRB Utilizing Rate(%)` | 31.243459 | 19.543420 | 3.433467 | 28.460000 | 67.694500 | 82.899733 |
| `LTE_UL PRB Utilizing Rate(%)` | 19.229711 | 8.793796 | 6.666667 | 18.666667 | 36.000000 | 44.000000 |
| `CQI AVG` | 9.432297 | 1.764754 | 7.244808 | 9.543134 | 11.757670 | 12.477115 |
| `L.UL.Interference.Avg(dBm)` | -114.475328 | 2.391305 | -118.000000 | -115.000000 | -110.000000 | -109.000000 |

## Available Labels

No explicit manually verified root-cause labels were found in the audited CSV. No incident ticket ID, work order ID, or expert root-cause column was observed.

Available supervised signals:

- Future KPI values for multi-step forecasting
- Weak KPI degradation labels derived from future KPI thresholds
- Weak root-cause ranking labels derived from alarm occurrence before degradation

Initial weak degradation prevalence using simple global thresholds:

| Weak label | Rule | Positive rows | Rate |
|---|---|---:|---:|
| `drop_high_gt_1pct` | `Call Drop Rate(%) > 1` | 1,663 | 1.60% |
| `rrc_low_lt_98pct` | `RRC Setup Success Rate(%) < 98` | 4,119 | 3.96% |
| `erab_low_lt_98pct` | `ERAB Setup Success Rate(%) < 98` | 3,467 | 3.33% |
| `dl_thr_low_q10` | `LTE_User DL Average Throughput(Mbps)` below global q10 | 10,399 | 10.00% |
| `cqi_low_q10` | `CQI AVG` below global q10 | 10,399 | 10.00% |
| `any_degradation` | OR of the above | 20,896 | 20.10% |

Final experiments should estimate quantile thresholds on the training split only to avoid leakage. Cell-specific thresholds should be reported as the primary setting; global thresholds can be a robustness check.

## Data Limitations

- Alarm labels are sparse; most alarm columns are constant zero in this slice.
- Cross-cell physical topology is not included. A cross-cell graph must therefore be inferred from same-site cell ID heuristics and/or training-set KPI correlation, and should be described as an inferred graph, not ground-truth topology.
- Root-cause labels are weak labels only unless Huawei/expert annotations are added later.
- One long time gap exists. Windows crossing this gap should be removed.
- All conclusions are restricted to this private cellular slice; do not claim public benchmark generality.

## Recommended Canonical Splits

Use chronological split by unique timestamp:

- Train: first 70%, ending around `2024-05-01 20:00:00`
- Validation: next 15%, ending around `2024-05-21 13:00:00`
- Test: final 15%

All threshold fitting, scaler fitting, inferred graph construction, and weak root-cause lift estimation must use train data only.
