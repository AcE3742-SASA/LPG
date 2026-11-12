# Matched learner comparison

This version adds MLP(32,32) and MLP(64,64) to RF100/depth8 and Ridge(alpha=1), without replacing the earlier run. It reuses the original feature builder, allowed-label loader, shape subsets and metric definitions. No locked/test action exists.

RQ1 compares log(T) and log(T/L) at 100/300/1000/2652 A shapes, then adds a separate full legacy-train endpoint of 3345 complete shapes. The latter combines A and C only for RQ1; RQ4 sources always use A=2652. All six dtype/mode strata are separate. Each comparison uses three common seeds and 716 validation shapes.

RQ4 also compares all four learners. Each learner chooses its own source target using allocate validation only. Its fixed source is evaluated with zero, constant-ratio, residual, target-direct and prior-direct calibration at 5/10/30/100/300 common C shapes and the same fixed 30-shape validation. Residual, direct and prior-direct use that same learner family. The source is not fine-tuned. Target labels from the original development validation have already been examined historically; this is a development comparison, not a new independent evaluation.

Both MLPs have two ReLU hidden layers (1601 and 5249 parameters at 15 inputs). They use Adam, learning rate 0.001, alpha=0.0001, batch size min(128,n), at most 3000 epochs, training-loss tolerance 1e-6 and patience 100. No outer-validation early stopping or parameter search is performed. X and the log-response are standardized using only the current training subset. RF uses its original raw features; Ridge standardizes X. No equal training-time budget is claimed. Fit times, training errors, stopping epochs and convergence warnings are saved. The fixed settings test these configurations, not the best possible member of each model family.

Run with the existing isolated Python environment:

```bash
cd /Users/studyhard/Claude/자율연구
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B analysis/learner_comparison_2026-10-28/selfcheck.py
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B analysis/learner_comparison_2026-10-28/compare_learners.py prepare
analysis/integrated_rq14_2026-10-28/.venv/bin/python -u -B analysis/learner_comparison_2026-10-28/compare_learners.py run
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B analysis/learner_comparison_2026-10-28/verify.py
analysis/integrated_rq14_2026-10-28/.venv/bin/python -u -B analysis/learner_comparison_2026-10-28/audit_convergence.py
analysis/integrated_rq14_2026-10-28/.venv/bin/python -B analysis/learner_comparison_2026-10-28/summarize.py
```

Use a new `--output` directory for replays. The runner refuses to overwrite an execution or change a frozen source. It saves 1920 validation configurations (720 RQ1, 1200 RQ4), endpoint models with scalers, all predictions, training diagnostics, shape subgroup errors and normalization paired bootstraps. This comparison does not start new GPU collection or modify previous selected source weights.

Postprocessing commands intentionally refer to the actual `runs/v1` comparison. The convergence audit refits every MLP configuration that hit the primary 3000-epoch cap, regardless of its validation error, with a separate 10000-epoch cap. All other settings remain fixed, and the primary results are preserved. The audit refuses to overwrite its own output. This extra optimization budget is reported separately.

The user subsequently chose minimizing error as far as practical rather than stopping at an error threshold. The frozen 15/30 minimum-cost selection remains available for comparison with the prior run. `accuracy_first_selection_validation_only.csv` is a separate derived selection by mean development MAPE, then p90, within the evaluated grid; it does not claim optimal performance across untested settings. The new 10000-shape GPU release and its standalone-source/admission checks are documented in `expansion_verification.json` and `../../gemm_expand_20261028_v1/README.md`.

The [scikit-learn 1.8 MLP documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.neural_network.MLPRegressor.html) defines training-loss stopping and epoch limits. The [scaling guidance](https://scikit-learn.org/1.8/modules/neural_networks_supervised.html#tips-on-practical-use) supports train-only preprocessing.
