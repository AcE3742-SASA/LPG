# RQ1 / RQ4 validation execution, 2026-10-28

Actual execution: `runs/v2`. `runs/v1` is an unexecuted protocol draft retained for history. The original dataset and role files were not changed. No final-test action or locked timing reader is implemented.

The runner reads labels only from `main_complete_unlocked_conditions.csv`, filtered to allowed shape groups, modes and `legacy_train` / `legacy_validation`. `prepare` uses identifiers only. `run` refuses to change frozen code/input hashes or overwrite an existing execution. Selected source learners are saved before target-label access. Three seeded A/C samples are nested and common across methods and dtype/mode strata.

The environment is isolated in `.venv`; `requirements.txt` pins installed versions. Console output is ASCII English. All RF fits use one thread and numerical-library threads are fixed in the commands below.

```bash
cd /Users/studyhard/Claude/자율연구
python3 -m venv analysis/integrated_rq14_2026-10-28/.venv
analysis/integrated_rq14_2026-10-28/.venv/bin/python -m pip install -r analysis/integrated_rq14_2026-10-28/requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B analysis/integrated_rq14_2026-10-28/test_rq14.py
# Use a NEW output directory for a future replay; existing v2 must be preserved.
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B analysis/integrated_rq14_2026-10-28/run_rq14.py prepare --output analysis/integrated_rq14_2026-10-28/runs/replay
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B analysis/integrated_rq14_2026-10-28/run_rq14.py run --output analysis/integrated_rq14_2026-10-28/runs/replay
```

`export_selected.py` and `summarize_rq14.py` intentionally point to the actual `runs/v2` artifacts. The export has already run and refuses to overwrite selected model files. Summarization can be repeated after preserving the current report if manual edits were made.

```bash
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B analysis/integrated_rq14_2026-10-28/summarize_rq14.py
```

Primary results: `RQ1_RQ4_실행결과_2026-10-28.md`, `runs/v2/metrics.csv`, `aggregate_metrics.csv`, `paired_bootstrap.csv`, `target_label_costs.csv`, `selected_configurations_validation_only.csv`. Model weights are in `models/` (source) and `selected_models/` (selected configuration envelopes). `validation_predictions.csv` contains only development validation predictions and labels.

The RQ4 zero-transfer curve repeats the same zero-calibration baseline at each k for comparison. Its effective target calibration budget is zero, plus 30 validation shapes. Reported times separate model fitting, source-prior generation, correction prediction and whole-pipeline validation prediction. Historical GPU collection wall time is UNKNOWN. These retrospective label budgets do not establish prospective GPU collection savings.

Cost accounting correction: use `selected_configurations_cost_aware_validation_only.csv` and `selected_models_cost_aware/` for the adopted minimum-cost configuration. The original selection and model envelopes remain preserved. `selection_correction.md` explains the zero-baseline grid-position bug; `correct_selection.py` derives the correction solely from saved validation aggregates. The zero source uses k=0 even though its comparison row appears at grid n=5. Export replay for the corrected choice used `export_selected.py --cost-aware` and passed exact prediction / serialization checks for all 30 envelopes. It has already run and also refuses to overwrite artifacts.
