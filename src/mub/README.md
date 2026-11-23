# MUB30 steering

The three model folders are `qwen` (Qwen2.5-VL-7B-Instruct), `llava`
(LLaVA-NeXT Mistral-7B), and `internvl` (InternVL3-8B). Each contains its adapter,
configuration, requirements, prompts, fitting/calibration, evaluation, and tests.

Use a separate environment for each model's requirements. Install the repository
with `python -m pip install -e .`, install
`src/mub/<model>/requirements.txt`, and set `model_path` in its `config.json`
or pass `--model-path` on the command line.

From the repository root:

```bash
python -B src/mub/qwen/run.py --stage validate
python -B src/mub/qwen/run.py --stage all --output results/mub/qwen_reproduction
```

Replace `qwen` with `llava` or `internvl`. The installed commands are
`vds-mub-qwen`, `vds-mub-llava`, and `vds-mub-internvl`. Validation checks local
data/configuration without loading weights. Full inference requires the model's
weights and dependencies.

The `v1` stage collects paired states from the 30 development examples, selects
a layer, fits a direction, and calibrates its sign. The `v2` stage evaluates the
frozen intervention. The saved summary separates `train`, `test`, and `all`;
the paper reports the 478-example `test` subset. Direction construction uses
the training-only option overrides documented in `data/mub/README.md`.

The release retains `axis.npy`, `crossfit_axes.npz`, `folds.json`, calibration
selection summaries, evaluation summaries, and run manifests. Per-question
answers, activation archives, and `crossfit.jsonl` are omitted. Use a new output
directory to regenerate a complete run. The native MUB experiment generates
records for all 508 examples but reports the held-out 478 separately.

`results/mub/qwen/v2/audit_summary.json` retains aggregate audit statistics;
question-specific repair lists and output details are omitted. These are saved
historical checks, not claims that a new model run has been executed locally.

The VDS90 transfer experiment in `src/vds_mub/` reuses this implementation and
fits a separate direction for each model. It does not reuse a direction from a
different model architecture.

CPU checks (from the repository root, after installation):

```bash
python -B -m unittest discover -s src/mub/qwen/tests -p "test_*.py"
python -B -m unittest discover -s src/mub/llava/tests -p "test_*.py"
python -B -m unittest discover -s src/mub/internvl/tests -p "test_*.py"
```

Tests requiring PyTorch/Transformers are skipped when those packages are absent.
