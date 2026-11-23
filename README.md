# VDS-Bench: core research release

This release contains the 450-pair VDS-Bench dataset and API evaluation, the
100-image visual decoy removal experiment, paired-state steering for four
open-weight vision-language models, and the supplemented MUB dataset with
MUB30 steering and VDS90-to-MUB478 transfer for three additional models.

## Directory map

| Component | Code | Data | Saved results |
| --- | --- | --- | --- |
| VDS-Bench / API evaluation | `src/api_eval/` | `data/benchmark/`, `data/images/` | `results/api_eval/` |
| Visual decoy removal | `src/visual_decoy_removal/` | `data/visual_decoy_removal/` | `results/visual_decoy_removal/` |
| Paired-state steering | `src/steering/` | shared VDS-Bench data and `data/splits/` | `results/steering/` |
| MUB30 steering | `src/mub/` | `data/mub/` | `results/mub/` |
| VDS90-to-MUB478 transfer | `src/vds_mub/` | shared VDS-Bench and MUB data | `results/vds_mub/` |

The two benchmark JSONL files contain the D+ and D- conditions for the same
450 images/questions, including answer options, gold answers, feedback targets,
and textual evidence. Match conditions by `pair_id`. The steering split contains
90 development and 360 test questions; `crossfit_folds.json` records the three
development folds. The 100-image experiment includes both original and edited
images and keeps questions, answer options, and feedback targets fixed.

## Installation

Run commands from the repository root, using Python 3.10 or newer:

```bash
python -m pip install -e .
```

This installs the package and lightweight analysis dependencies. For local model
inference, install the relevant model's requirements in a separate environment:

```bash
python -m pip install -r src/steering/qwen/requirements.txt
```

Replace `qwen` with `gemma`, `deepseek`, or `muse`. DeepSeek uses a different
Transformers stack and also requires the `deepseek_vl2` implementation; follow
the instructions in its requirements file. Obtain the model weights separately
and set `model_path` in `src/steering/<model>/config.json`.

## 1. VDS-Bench and API evaluation

Copy `.env.example` to `.env` and fill in the relevant provider credentials.
Supported profiles are Qwen, Kimi, MiMo, MiniMax, and an OpenAI-compatible
generic endpoint. The saved main-experiment runs are Qwen3.7-Plus, Kimi-K3,
MiMo-V2.5, and MiniMax-M3.

Preview a complete run without sending requests:

```bash
vds-api-eval --provider qwen --model qwen3.7-plus --full --dry-run
```

Run both feedback conditions with a new output name:

```bash
vds-api-eval --provider qwen --model qwen3.7-plus --full --run-prefix qwen_reproduction
```

Analyze saved results without calling an API:

```bash
vds-analyze-api --dplus results/api_eval/qwen3.7-plus/dplus --dminus results/api_eval/qwen3.7-plus/dminus --output results/api_eval/qwen3.7-plus/analysis
```

The primary paired comparison includes only questions initially answered
correctly in both arms with all response rounds parsed successfully. MiniMax
has 449 recorded pairs and 413 pairs eligible under this rule. Other saved API
runs contain 450 recorded pairs. Per-arm summaries and the paired analysis use
different eligibility sets; use `paired_primary` for the paired comparison.

## 2. Visual decoy removal on 100 images

Copy `configs/visual_decoy_removal/.env.example` to a `.env` file in the same
directory and configure credentials. This experiment uses MiMo-V2.6-Pro rather
than the main benchmark's MiMo-V2.5.

```bash
vds-decoy-removal --check
vds-decoy-removal --provider qwen --model qwen3.7-plus --run-name qwen_reproduction
python -B src/visual_decoy_removal/analyze.py --run results/visual_decoy_removal/qwen3.7-plus
```

The four conditions are original/edited images crossed with D+/D- feedback.
Use `all_four_matched` in each `analysis.json` for a fixed cohort that is parsed
successfully and initially correct in all four conditions. The existing
100-image dataset can be checked locally without credentials or API requests.
See `src/visual_decoy_removal/README.md` for the data fields and bootstrap command.

## 3. Paired-state steering

The supported model folders are `qwen`, `gemma`, `deepseek`, and `muse`.
Each has its own adapter, configuration, requirements, and tests. The `v1` and
`v2` directories are consecutive pipeline stages, both required:

| Stage | Code / outputs | Purpose |
| --- | --- | --- |
| `collect` | `v1/`, `v1/activations/` | Collect paired D+/D- development states |
| `fit` | `v1/`, `subspace.pt`, `crossfit_axes.pt` | Select a layer and fit the pooled direction |
| `calibrate` | `v1/calibration/` | Select the sign using development option-text scores |
| `evaluate` | `v2/records/`, `v2/summary.json` | Evaluate frozen interventions on 360 test questions |

Run all stages in the model's environment:

```bash
vds-qwen-paired-steering --stage all --output results/steering/qwen_reproduction
```

The other commands are `vds-gemma-paired-steering`,
`vds-deepseek-paired-steering`, and `vds-muse-paired-steering`. Alternatively,
use `python -B src/steering/<model>/run.py` with the same arguments. Inspect
`--help` for individual stages and selecting a sign from saved calibration records.

All steering conditions use the unmodified initial response as feedback-dialogue
history. Sycophancy uses the fixed baseline-initial-correct cohort; initial-answer
accuracy is evaluated separately for each intervention condition. The evaluation
includes magnitudes 0.1, 0.2, and 0.3 in both directions. The release retains
fitted directions, selection summaries, evaluation summaries, and run metadata.
Per-question steering outputs and activations are omitted; reproduce all stages
in a fresh output directory when those records are needed.

## 4. Supplemented MUB and steering

`data/mub/` contains a fixed 30-example development set and a 478-example test
set, with exactly the 508 referenced images. The test set retains 231 original
explicit incorrect targets and supplements 247 targets. All 30 development
examples have D+/D- labels; 21 have training-only incorrect-option overrides.
Original evaluation options remain in `options`. See `data/mub/README.md` for
the annotation fields, original-source references, and supplementation scope.

The models are Qwen2.5-VL-7B-Instruct, LLaVA-NeXT Mistral-7B, and InternVL3-8B.
Install the corresponding `src/mub/<model>/requirements.txt` in a separate
environment and configure its local model path.

Validate the MUB data without loading weights:

```bash
python -B src/mub/qwen/run.py --stage validate
python -B src/mub/llava/run.py --stage validate
python -B src/mub/internvl/run.py --stage validate
```

Run MUB30 extraction and evaluation, or VDS90-to-MUB478 transfer:

```bash
python -B src/mub/qwen/run.py --stage all --output results/mub/qwen_reproduction
python -B src/vds_mub/run.py --model qwen --stage all --output results/vds_mub/qwen_reproduction
```

Replace the model with `llava` or `internvl` as appropriate. MUB30 fits on the
30 MUB development examples; transfer fits on the fixed VDS development set of
90 examples. Both report held-out MUB478 performance. The native MUB summaries
also contain separate development/all-example statistics; use the `test`
section for the paper comparison.

The retained results include fitted directions, calibration and evaluation
summaries, run manifests, and aggregate comparison reports. They exclude
per-question generated answers, activations, and cross-validation scores.
Existing comparison reports can be read directly; recalculating comparisons
requires fresh per-question results for both source settings. See
`src/mub/README.md` and `src/vds_mub/README.md` for details.

## Local checks and release contents

The steering tests use synthetic or saved inputs and do not load full model
weights. Run each model's suite, for example:

```bash
python -B -m unittest discover -s src/steering/qwen/tests -p "test_*.py"
```

Real PyTorch numerical tests are skipped when PyTorch is unavailable.

The public files are this README, `pyproject.toml`, `.gitignore`, the credential
examples, and the retained `src/`, `configs/`, `data/`, and `results/` trees.
Local `.env` files, manuscript directories (`cvpr2027*`), `.explore/`, editor
settings, model checkpoints, and temporary files are excluded by `.gitignore`.
When assembling a ZIP instead of using Git, use the same public-file list and
exclude the real `.env` files explicitly.

This is a scoped release of the components above. MMBench, MM-SY, D-N, and
visual-token pruning experiments are outside this release. Historical
run manifests retain their original source hashes and runtime paths as provenance;
new inference runs should use a fresh output directory.
