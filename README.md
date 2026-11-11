# TRACE-MEC

**Injection-Robust Trust Transfer Across Deadline-Constrained Handovers in Vehicular Edge Computing**

TRACE-MEC is a research implementation for studying how vehicular misbehaviour evidence can be transferred between Multi-access Edge Computing (MEC) servers during network handovers. The study focuses on limited handover time and communication budgets, unreliable transfer, evidence suppression, and adversarially injected reassuring evidence.

The implementation combines evidence scoring, deadline- and byte-budget-aware transfer policies, destination-side belief aggregation, and a **simulated** position-consistency probe. It is a trace-driven simulation and evaluation framework; it is not a deployed MEC service or a live vehicular-network testbed.

## Research scope

The primary research question is how a destination edge server can make a more reliable trust decision when only a limited amount of evidence can be transferred before a vehicle's handover deadline.

The code supports experiments involving:

- Vehicle misbehaviour evidence and handover episodes.
- Evidence ranking and selection under per-session byte and time budgets.
- Transfer faults, including loss, duplication and delay.
- Adversarial evidence suppression and reassuring-item injection.
- Destination-side belief aggregation and decision policies.
- Mobility, path-fold and adversarial-profile evaluation.
- Sensitivity analyses, ablations and bootstrap-based statistical summaries.

The post-quantum cryptography (PQC) experiment models the additional item size associated with ML-DSA-44 signatures. **It does not implement cryptographic signing or verification.**

## Repository contents

| Path | Purpose |
|---|---|
| `src/preprocess.py` | Input-data preparation and split handling |
| `src/model.py` | Evidence model, transfer simulation, belief aggregation and decision logic |
| `src/train.py` | LSTM training workflow |
| `src/evaluate.py` | Evaluation orchestration |
| `src/stats.py` | Statistical analysis and bootstrap summaries |
| `src/ablation.py` | Ablation and sensitivity analyses |
| `src/a20.py`, `src/a20_lstm.py` | Position-attack extension workflows |
| `src/fig/` | Figure-generation source |
| `config/config.yaml` | Experiment settings and path configuration |
| `results/` | Selected compact CSV summaries from the original runs |
| `release.py` | Supported primary-evaluation release wrapper |
| `SOURCE_SHA256.json` | Source-integrity manifest used by the release workflow |
| `REPRODUCIBILITY.md` | Supported workflows, scope and scientific limitations |
| `CITATION.cff` | Software citation metadata |

The public release does not include raw datasets, trained model checkpoints, generated evidence or episode-level Parquet files, full run logs, local development history, or the complete figure-generation inputs. The manuscript and internal editorial materials are also not included.

## Data sources

The experiments use two separately generated data sources. They are not jointly recorded observations from a single real-world vehicular network.

1. **VeReMi NextGen** — vehicle messages and attack labels. Obtain the dataset from the [VeReMi NextGen project](https://veremi-dataset.github.io/veremi-nextgen). Place the original outer ZIP archives under `data/raw/veremi_nextgen/`. The primary scenarios are `InTAS_highway_2`, `InTAS_highway_7`, `InTAS_urban_2` and `InTAS_urban_7`. Each scenario uses the baseline, `timeDelayAttack`, `dataReplay` and `dosAttack` variants. Preserve the original nested ZIP layout and ground-truth filenames described in `config/config.yaml` and `src/preprocess.py`.
2. **Dataset for 5G HetNet Handover Evaluation, version 1** — radio and mobility traces. Obtain it from [Zenodo](https://doi.org/10.5281/zenodo.22251564) and place the supplied `dataset.zip` under `data/raw/hetnet_2026/`. The underlying simulator is documented at [Zenodo](https://doi.org/10.5281/zenodo.18313308) and [GTI's 5G handover simulator repository](https://github.com/gti-uvigo/5g-handover-sim).

Preprocessing maps vehicle sessions to disjoint network-path pools. The dataset owners' licences apply to the source data; the datasets are not redistributed in this repository.

## Software environment

The recorded main environment is Python 3.14.7. The LSTM workflow uses a separate Python 3.12.14 environment with PyTorch 2.2.2. Exact dependency records are provided in `requirements.txt` and `requirements_torch.txt`. These are archival environment records; installation has not been verified on every operating system or hardware configuration.

Create the environments from the repository root:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

python3.12 -m venv .venv_torch
.venv_torch/bin/python -m pip install -r requirements_torch.txt

.venv/bin/python release.py verify
```

Use equivalent interpreter paths on other platforms. Git is required by some retained reporting utilities.

## Primary reproduction workflow

### 1. Prepare the datasets

```bash
.venv/bin/python src/preprocess.py --stage clean --scenarios all
.venv/bin/python src/preprocess.py --stage traces --scenarios all
```

This prepares the derived inputs and keeps the Test split in a separate held-out location.

### 2. Train and prepare each scenario/seed

For each of the four primary scenarios and each seed from 0 to 4, run the LSTM training step and the model preparation/smoke workflow. For example:

```bash
.venv_torch/bin/python src/train.py --stage lstm --scenario InTAS_highway_2 --seed 0
.venv/bin/python src/model.py --smoke --scenario InTAS_highway_2 --seed 0
```

Repeat for every scenario and seed before the corresponding Test evaluation.

### 3. Validate and run the primary Test evaluation

```bash
.venv/bin/python release.py dryrun --seed 0
.venv/bin/python release.py final --seed 0
.venv/bin/python release.py seeds --seeds 1 2 3 4
```

The release wrapper uses SHA-256 source verification in place of private Git-commit checks. The numerical model, split restrictions, run locks and attempt records remain in the underlying drivers. A new run is a reproduction attempt, not the original preregistered Test evaluation. Existing run locks are intended to prevent accidental overwrites; use a fresh release checkout for an independent reproduction.

A full reproduction is computationally substantial. The original primary Test run across seeds took approximately three hours, excluding data preparation, LSTM training and robustness reruns.

## Additional analyses

The repository also contains sensitivity/recalibration analyses (`src/ablation.py`), position-attack extensions (`src/a20.py` and `src/a20_lstm.py`), and figure-generation code under `src/fig/`. Some advanced orchestration paths retain historical Git-history checks and are provided as research source rather than as portable, one-command workflows. The release wrapper supports the primary dry-run, final-evaluation and multiseed workflows only. Consult [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before attempting additional analyses.

Regenerated mobility traces require the independent 5G handover simulator. The default configuration points to `external/5g-handover-sim/gate_traces`; obtain the simulator and traces separately and configure the path in a local experimental copy. Original regenerated traces and compiled simulator binaries are not included.

## Results and interpretation

The `results/` directory contains selected compact CSV summaries from the original research runs. It does not contain the full episode-level outputs, trained checkpoints or all inputs needed to regenerate every figure and uncertainty interval. The published/reported results were **not regenerated as part of preparing this repository release or updating this documentation**.

Interpret the findings within the evaluated scenarios and adversary menu. In particular:

- The reported worst-case metric is worst-case over the evaluated profile/attack menu, not over every possible adversary.
- Position-probe observations are synthesized. A deployed challenge-response protocol, real sensing latency, missing-response handling and joint transfer/probe scheduling were not validated.
- The PQC comparison models signature-related byte overhead; it does not implement signing or verification.
- VOI ranking and recalibration use evaluation batches and should not be described as validated causal, online streaming adaptation.
- The position-attack extension did not include its planned Validation evaluation.
- The hypothesis that larger signatures would degrade TRACE-MEC less than the Severity/LSTM alternatives in average attack loss was not supported.
- Early selection-only designs did not meet their development criteria; the final framework's reported worst-case improvement is attributed chiefly to the probe and robust-belief components.

Further scope and validation details are recorded in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Citation

The repository includes a `CITATION.cff` software-citation record. It currently provides the software title, version, release date and listed authors, but does not include an assigned repository DOI. When citing the work, also cite the original VeReMi NextGen and 5G HetNet data sources and the associated handover-simulator publication.

## Licence and permitted use

The original TRACE-MEC code is distributed under the **TRACE-MEC Research Reproduction License v1.0**, not an OSI-approved open-source licence. The permission is limited to noncommercial reproduction, verification and evaluation of the research results. Source-code modification, derivative development, commercial use and redistribution require prior written permission from the relevant rights holders. Dependencies, datasets and external simulator software remain subject to their own licences.

See [LICENSE](LICENSE) and [LICENSE-NOTICE.md](LICENSE-NOTICE.md) before using the code.
