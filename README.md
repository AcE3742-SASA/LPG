# LPG

Latency Prediction Through GEMM Building Blocks

LPG studies how far GEMM timing models can carry over to layer latency prediction. It predicts individual matrix multiplications, adds their predicted times, and learns a layer-level correction from measured layer times.

This repository contains the measurement code, analysis results, and the data used in the study. Measurement data, trained models, and larger archives are stored as release assets with a path and SHA-256 manifest. The current experiments cover FP16 and FP32 GEMM, Linear, FFN, Conv1×1, and Conv3×3 on an NVIDIA RTX 4000 SFF Ada Generation. The measurement target is warmed-up GPU execution time, measured with CUDA Events. The main campaigns use a 720 MHz SM clock target.

## Method

The GEMM models use matrix dimensions and derived static features to predict either `log(T)` or `log(T/L)`, where `L` is a measured Roofline-based lower bound. The comparison includes Random Forest, Ridge, and two small MLPs.

For a layer with predicted constituent GEMM times, let `S` be their sum. The residual model learns `log(T_layer/S)` from static layer features. The experiments also compare direct layer prediction, adding `log(S)` as an input feature, a constant correction, and sums of separately predicted component operations. Standard framework execution and explicit decomposition have separate timing targets.

## Results

The final evaluation on November 11, 2026 used frozen model selections and weights. Values below are mean test MAPE across three seeds.

| Target | FP16 | FP32 |
| --- | ---: | ---: |
| GEMM, output allocation | 12.44% | 6.68% |
| GEMM, preallocated output | 12.25% | 6.68% |
| GEMM, CUDA Graph replay | 12.95% | 6.85% |
| Linear, standard | 11.02% | 6.71% |
| FFN, standard | 8.47% | 5.39% |
| Conv1×1, standard | 10.71% | 8.82% |
| Conv3×3, standard | 11.80% | 14.48% |

GEMM uses 1,000 reserved test shapes per precision and execution mode. Layer conditions use 50 test shapes, with 49 for Conv3×3 FP32 after the recorded correctness exclusion. Each layer row uses the method selected on validation data; Conv3×3 FP16 selects direct prediction.

The [evaluation report](research/analysis/final_test_2026-11-11/README.md) gives model choices, training sizes, p90 errors, decomposition results, and uncertainty estimates. The table values come from [GEMM](research/analysis/final_test_2026-11-11/summary/gemm_expanded.csv) and [layer](research/analysis/final_test_2026-11-11/summary/layers.csv) CSVs.

## Files

| Start here | Contents |
| --- | --- |
| [Research map](docs/research-map.md) | Data, code, models, and their dependencies |
| [Final evaluation](research/analysis/final_test_2026-11-11/) | Frozen test predictions, metrics, figures, and verification records |
| [Layer development](research/analysis/layer_full_development_2026-11-11/README.md) | Validation results and learning curves |
| [Prior-model comparison](research/analysis/prior_model_audit_2026-11-11/README.md) | Corrected NeuralPower, GEMMify, Paleo, and NeuSight comparisons |
| [Data efficiency](research/analysis/cost_data_efficiency_2026-11-11/README.md) | Label budgets, collection records, and conditional reuse estimates |

The dated directories retain the experiment history, including failed runs, quality flags, embedded source snapshots, and SHA-256 manifests. Their names and relative layout are part of that history. In particular, `abstract_sweep_725_v1` contains the 720 MHz collection.

## Working with the archive

Start with the saved CSVs and reports under `research/`. Restore the release assets to obtain the complete research tree; the manifests record each file's original relative path and checksum. For code, the [research map](docs/research-map.md#running-the-code) lists the collection and analysis entry points and their environment requirements. Historical protocols contain fixed hashes, software versions, and some absolute paths; reruns need a separate output directory and explicit path setup.

With Python 3.10+ and an authenticated [GitHub CLI](https://cli.github.com/):

```bash
gh repo clone AcE3742-SASA/LPG
cd LPG
python3 scripts/artifacts.py fetch
python3 scripts/artifacts.py restore --path-prefix research/analysis/final_test_2026-11-11
python3 scripts/artifacts.py verify --path-prefix research/analysis/final_test_2026-11-11
```

Omit `--path-prefix` to restore or verify the full archive. Full restoration needs space for the research files and downloaded packs; the script checks available space before writing. See [archive storage](docs/archive-storage.md) for the file manifest and verification procedure.

The completed evaluation covers GEMM and individual layers on the recorded GPU and software stack. The cost analysis counts labels under its stated reuse assumptions, with measured collection time reported separately.

## Citation and reuse

Cite [LPG v1.0.0](https://github.com/AcE3742-SASA/LPG/releases/tag/v1.0.0) when using the code or data. Author metadata is in [CITATION.cff](CITATION.cff); a reference and BibTeX entry are in [docs/citing.md](docs/citing.md).

Original code is licensed under [MIT](LICENSE), and the research team's data under [CC BY 4.0](LICENSE-DATA). See [license scope](LICENSE-SCOPE.md) for the covered materials and [third-party notices](THIRD_PARTY.md) for bundled dependencies.
