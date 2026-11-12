# Third-party materials

Pinned upstream files are stored under `research/analysis/prior_model_comparison_2026-11-11/sources/`. Their source URLs and commit hashes are recorded in [repositories.json](research/analysis/prior_model_comparison_2026-11-11/sources/repositories.json).

| Project | Upstream | License retained in this release |
| --- | --- | --- |
| NeuralPower | [enyac-group/NeuralPower](https://github.com/enyac-group/NeuralPower) | [Apache 2.0](research/analysis/prior_model_comparison_2026-11-11/sources/neuralpower/LICENSE) |
| Paleo | [TalwalkarLab/paleo](https://github.com/TalwalkarLab/paleo) | [Apache 2.0](research/analysis/prior_model_comparison_2026-11-11/sources/paleo/LICENSE) |
| NeuSight | [neusight/README](research/analysis/prior_model_comparison_2026-11-11/sources/neusight/README.md) | [MIT](research/analysis/prior_model_comparison_2026-11-11/sources/neusight/LICENSE) |

The NeuSight source subtree includes upstream data and pretrained weights. Those files retain the upstream license. Original imported files remain byte-for-byte copies of the recorded versions. The comparison adapters and audit scripts describe the local ports, calibration, and preprocessing changes.

Measurement code also uses separately installed PyTorch, CUDA, CUTLASS, and BabelStream components. Source headers and notices preserved in the measurement snapshots remain applicable to their respective files.

