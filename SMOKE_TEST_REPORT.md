# SMOKE_TEST_REPORT

## Command

```powershell
python run_smoke.py --csv data\synthetic_toy_510957.csv --horizons 1,3 --max-train-samples 2000 --max-test-samples 1000 --max-feature-kpis 12
```

## Status

- Smoke test: PASS
- Dataset used: `data/synthetic_toy_510957.csv`
- Dataset shape: 5,544 rows x 187 columns
- Cells: 33
- Unique hourly timestamps: 168
- KPI columns: 48
- Alarm/fault columns: 137
- Active alarm/fault columns in toy data: 11

## Notes

- This smoke test validates code execution and data-interface compatibility only.
- Synthetic toy results are not paper results.
- Exact paper metric reproduction requires authorized access to the private `510957.csv` file.
