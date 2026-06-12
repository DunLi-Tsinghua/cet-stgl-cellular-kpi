# Data README

## Files

- `synthetic_toy_510957.csv`: synthetic toy data for code smoke tests. It is not the real dataset and is not used for the paper's reported metrics.
- `schema_510957_public.json`: public schema/statistics metadata for the private `510957.csv` panel.

## Private Raw Data

The real `510957.csv` file is not included in this repository because it contains private cellular operational records and business-sensitive KPI/alarm information.

The paper reports:

- 103,983 rows
- 187 original columns
- 33 cells
- 3,151 hourly timestamps
- 48 KPI/business columns
- 137 alarm/fault columns
- 11 active alarm/fault columns
- no missing values in the audited slice

To reproduce private-data metrics, obtain proper authorization and place the raw CSV locally as:

```text
data_private/510957.csv
```

`data_private/` is excluded from git.

## Weak Labels

Forecasting labels are real future KPI values. Degradation and root-cause ranking labels are weak labels fitted from the training split only. Ranking labels should be interpreted as weak-label/proxy consistency, not expert-confirmed root cause.

