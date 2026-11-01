# TRACE-MEC

Research code and manuscript for **TRACE-MEC: Injection-Robust Trust Transfer Across Deadline-Constrained Handovers in Vehicular Edge Computing**.

TRACE-MEC studies transfer of vehicle misbehavior evidence between MEC servers under deadlines, byte budgets, reassuring injections and evidence suppression. It combines a monotone belief, strongest-first transfer and a simulated independent position probe. The detector is exchangeable. The post-quantum experiment models ML-DSA-44 signature bytes; it does not implement signing or verification.

## Release contents

- `src/`: preprocessing, models, LSTM training, evaluation, bootstrap statistics, ablations, extension and figure source.
- `config/`: experimental settings and display names. Paths are relative to this folder.
- `results/`: compact CSV summaries from the original runs, not model checkpoints or episode-level outputs.
- `docs/`: empty output directory reserved for reports generated during reproduction. No internal notes are distributed.
- `release.py`, `SOURCE_SHA256.json`: portable primary-evaluation entry point and release source integrity checks.
- `REPRODUCIBILITY.md`: public scope, validation limits and supported workflows.

Raw data, generated evidence/episode parquet files, trained models, Python environments, local Git history, caches and logs are excluded. Published results have not been regenerated while preparing this release.

## Data availability

Obtain both independently simulated sources from their owners:

1. **VeReMi NextGen**: https://veremi-dataset.github.io/veremi-nextgen — vehicle messages and attack labels. Place original outer ZIP archives in `data/raw/veremi_nextgen/`. Primary scenarios: `InTAS_highway_2`, `InTAS_highway_7`, `InTAS_urban_2`, `InTAS_urban_7`. Each needs baseline, `timeDelayAttack`, `dataReplay`, and `dosAttack` archives. Preserve the original nested ZIP structure (`<stem>/<Train|Validation|Test>/<stem>.zip`) and ground-truth naming. See `config/config.yaml` and `src/preprocess.py`.
2. **Dataset for 5G HetNet Handover Evaluation**, v1: https://doi.org/10.5281/zenodo.22251564 — radio/mobility traces. Place `dataset.zip` in `data/raw/hetnet_2026/` as distributed. Underlying simulator: https://doi.org/10.5281/zenodo.18313308 and https://github.com/gti-uvigo/5g-handover-sim .

Preprocessing maps vehicle sessions to disjoint network path pools. These are not jointly recorded real-world observations. Dataset licenses remain with the owners; no data are redistributed here.

## Environment

Original main environment: Python 3.14.7; exact recorded dependencies in `requirements.txt`. Original LSTM environment: Python 3.12.14 and PyTorch 2.2.2; exact recorded dependencies in `requirements_torch.txt`. These are separate environments because the original Intel macOS system did not have a compatible PyTorch build for the main Python version. Other platforms may require compatible PyTorch wheels. The exact locks are archival environment records, not a claim that every wheel is available for every platform.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3.12 -m venv .venv_torch
.venv_torch/bin/python -m pip install -r requirements_torch.txt
.venv/bin/python release.py verify
```

Run commands from this release root. Use the equivalent environment paths on Windows. Ensure Git is installed for the retained reporting utilities.

## Primary reproduction sequence

Data preparation (writes derived data and extracts Test into a separate held-out directory):

```bash
.venv/bin/python src/preprocess.py --stage clean --scenarios all
.venv/bin/python src/preprocess.py --stage traces --scenarios all
```

For each of the four scenarios and each seed 0–4, train the LSTM and build the Train/Validation model outputs, for example:

```bash
.venv_torch/bin/python src/train.py --stage lstm --scenario InTAS_highway_2 --seed 0
.venv/bin/python src/model.py --smoke --scenario InTAS_highway_2 --seed 0
```

Repeat these two commands for every scenario/seed before the corresponding Test evaluation. A full reproduction is computationally substantial; the original main Test run across seeds took about three hours, excluding data preparation, LSTM training and robustness reruns.

Validate the evaluation path and then run primary Test:

```bash
.venv/bin/python release.py dryrun --seed 0
.venv/bin/python release.py final --seed 0
.venv/bin/python release.py seeds --seeds 1 2 3 4
```

The release wrapper substitutes SHA-256 source verification for private Git commit checks only; numerical model code, split restrictions, run locks and attempt records remain in the original drivers. A repeat run is a reproduction, not the original preregistered Test evaluation. Existing run locks deliberately prevent silent overwrites. Use a fresh release directory for an independent reproduction. Do not fabricate the dry-run marker.

## Additional experiments and figures

`src/ablation.py` contains sensitivity/recalibration analyses; `src/a20.py` and `src/a20_lstm.py` contain position-attack evaluation; `src/evaluate.py` includes regenerated mobility and path-fold evaluation. These advanced orchestration paths retain historical Git-history checks and are included as source, not advertised as portable commands. The release wrapper supports primary dry-run/final/multiseed evaluation only. See REPRODUCIBILITY.md.

The independent simulator must be obtained separately. `config/config.yaml` uses `external/5g-handover-sim/gate_traces` for regenerated traces; set this to the actual location in an experimental copy. Code modifications require written permission; use documented options for the supplied reproduction workflow. Original regenerated traces and compiled simulator binaries are not included.

Figure scripts in `src/fig/` read full regenerated episode outputs; the compact CSV summaries alone do not regenerate all uncertainty intervals or figures. The manuscript and internal editorial records are not bundled in this code-only release.

## Interpretation and limitations

See `REPRODUCIBILITY.md`. The worst-case metric is over the evaluated profile/attack menu, not every possible attacker. Probe observations are synthesized; actual network challenge-response, sensing latency, packet sizes and crypto verification are not implemented. Batch VOI selection and batch recalibration are not validated causal online schedulers. Report unsupported findings, including failed early robust selection designs and H13.

## Citation, rights and publication

`CITATION.cff` is a software citation scaffold; add the public repository URL, Zenodo DOI and confirmed software contributor identities when assigned. Dataset citations are given above. Cite both data sources and the associated simulator publication.

This folder is prepared for review and upload; nothing has been uploaded. Original TRACE-MEC code uses the restricted TRACE-MEC Research Reproduction License v1.0 (see `LICENSE` and `LICENSE-NOTICE.md`). Noncommercial reproduction/verification only; commercial use, redistribution and general derivative development require written permission. This is source-available, not open source. Source-code modifications are not permitted without written permission. A Zenodo software deposit should archive this same reviewed release; GitHub–Zenodo linking and DOI issuance occur after the repository/release exists.
