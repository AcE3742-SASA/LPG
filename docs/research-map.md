# Research map

The research developed through clock and timing checks, GEMM collection, layer collection, model selection, and a frozen test evaluation. The original folder layout under `research/` keeps those stages connected to their source snapshots and manifests. The Git tree contains code, documents, and smaller results. Release assets supply the larger data and model files; restore them before following links into those parts of the archive.

## Current results and source code

| Stage | Code or data | Result |
| --- | --- | --- |
| Roofline calibration at 720 MHz | [roofline_720_v1](../research/roofline_720_v1/) | [Calibration report](../research/analysis/roofline_720_result_2026-09-22/report.md) |
| Original GEMM model comparison | [learner_comparison_2026-10-28](../research/analysis/learner_comparison_2026-10-28/) | [Model comparison](../research/analysis/learner_comparison_2026-10-28/README.md) |
| Additional 10,000 GEMM shapes | [gemm_expand_20261028_v1](../research/gemm_expand_20261028_v1/) | [Returned data and fitting](../research/analysis/gemm_expansion_result_2026-11-06/README.md) |
| Layer collection | [layer_full_20261110_v5](../research/layer_full_20261110_v5/) | [Returned-data verification](../research/analysis/layer_full_result_2026-11-10/README.md) |
| Layer fitting and selection | [pipeline.py](../research/analysis/layer_full_fit_2026-11-10_v5/pipeline.py) | [Development report](../research/analysis/layer_full_development_2026-11-11/README.md) |
| Frozen test evaluation | [final_test_2026-11-11](../research/analysis/final_test_2026-11-11/) | [Evaluation report](../research/analysis/final_test_2026-11-11/README.md) |
| Prior-model comparison | [prior_model_audit_2026-11-11](../research/analysis/prior_model_audit_2026-11-11/) | [Corrected comparison](../research/analysis/prior_model_audit_2026-11-11/README.md) |
| Label and collection costs | [cost_data_efficiency_2026-11-11](../research/analysis/cost_data_efficiency_2026-11-11/) | [Cost report](../research/analysis/cost_data_efficiency_2026-11-11/README.md) |


## Data and model provenance

Keep original archives, checksum sidecars, extracted measurements, and embedded dependencies together. Several historical collection packages import neighboring packages, and recorded manifests refer to their original names.

The GEMM expansion adds 8,153 accepted training shapes and 895 validation shapes to the previous 3,345 and 716. The selected expanded models therefore use 11,498 training shapes and 1,611 validation shapes. The eligibility table and frozen selection are under [fit_v1](../research/analysis/gemm_expansion_result_2026-11-06/fit_v1/).

The complete layer campaign is stored at `research/analysis/layer_full_result_2026-11-10/raw/` in the [release archive](https://github.com/AcE3742-SASA/LPG/releases/tag/v1.0.0). It retains the returned archives and campaign records across collection versions. The accepted training, validation, and test measurements, correctness exclusions, and failed attempts are described in the adjacent verification report.

The [prepared layer directory](../research/analysis/layer_full_fit_2026-11-10_v5/prepared_layer_full/) holds the frozen protocol, selected model weights, and cost report. Its `selection_frozen.json` SHA-256 is:

```text
beed9b4314c3f258f0fcd6f86d19b15d271fac2d1abe705875aa50a94e99cbd4
```

The final test directory stores predictions and metrics together with [preflight checks](../research/analysis/final_test_2026-11-11/preflight.json), [independent verification](../research/analysis/final_test_2026-11-11/independent_verification.json), and an [artifact manifest](../research/analysis/final_test_2026-11-11/artifact_manifest.json). `PASS`, `REVIEW`, and correctness exclusions retain their original meanings throughout the archive.

## Running the code

Collection runs on the Ubuntu research PC under `/home/eu/gemm-research/`. Each package's README defines its hardware settings and launch command. New measurements need a new run directory; preserve the source and manifest that produced each earlier run. Terminal output for new collection scripts uses English ASCII.

The current layer collector is `research/layer_full_20261110_v5/collect_all.py`, with `selfcheck.py` for its local checks. It imports the earlier campaign before continuing collection. Read that package's README and the completed campaign report together: the package README records the restart procedure, while the report records its eventual completion.

CPU analysis uses the pinned [requirements](../research/analysis/integrated_rq14_2026-10-28/requirements.txt). The frozen layer protocol records Python 3.14.3, NumPy 2.5.4, scikit-learn 1.8.0, and joblib 1.6.0. GPU collection has its own environment metadata. Analysis scripts commonly expect single-threaded BLAS settings:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

Use these entry points after restoring the data paths and matching the recorded environment:

| Task | Entry point |
| --- | --- |
| Check GEMM expansion inputs | `research/analysis/gemm_expansion_result_2026-11-06/verify_expansion.py` |
| Fit GEMM expansion candidates | `research/analysis/gemm_expansion_result_2026-11-06/compare_expansion.py` |
| Prepare or fit layer models | `research/analysis/layer_full_fit_2026-11-10_v5/pipeline.py` |
| Recheck saved final predictions and metrics | `research/analysis/final_test_2026-11-11/verify_results.py` |
| Recheck label and collection costs | `research/analysis/cost_data_efficiency_2026-11-11/verify.py` |

The final-test verifiers validate frozen input hashes and sealed-access records. Some records contain the original absolute output path, so moving the archive requires an explicit path mapping in a separate rerun setup. Keep the archived records intact. Use a new output directory for new training or evaluation; the original evaluation has already consumed its reserved test set.

## Reading the comparisons

For NeuralPower results, use `prior_model_audit_2026-11-11`; the earlier `prior_model_comparison_2026-11-11` directory records the first implementation. The audit corrects cross-validation preprocessing and separates comparisons on equal training pools from the existing selections that use GEMM preparation data.

The cost report uses validation curves to count the labels needed to meet a target error. Its reduction estimates depend on the tested shape sets, shared GEMM preparation, and the stated reuse model. The actual collection ledger includes failed and superseded attempts, which belong to the cost of conducting the research.
