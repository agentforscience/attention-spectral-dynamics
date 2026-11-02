# Spectral Dynamics of Attention Matrices During Training

We tracked per-head attention spectra over training in 68 transformers (text, vision, BERT fine-tuning) and ran 120 more with controlled H(Y|X), to test five claims on rank bifurcation, entropy shape, entropy pruning, an MI bound, and depth-accelerated collapse. Four claims fail as stated, and the pruning rule passes only on SST-2, where entropy did no better than random selection.

## Key findings
- **No bifurcation.** 0/68 runs meet the criterion under any of five rank definitions; heads form one population.
- **Rank rises, it does not collapse, under default init.** Attention starts near-uniform (rank_eff/n ≈ .02–.09) and entropy climbs. Collapse appears only from peaked starts, and it deepens with depth (7% in layer 1 to 75% in layer 6 on AG News), at about 1.8× the speed rather than 3–5×.
- **S < 0.3·log n is not safe for pruning.** It removes 63–96% of heads on AG News and CIFAR-10 for 19–45 pp loss. Entropy is still a useful ranking on CIFAR-10 (3–13 pp better than random) but not on text.
- **I(A;X) ≤ H(Y|X) + O(1/√n) is false.** There is a simple counterexample, and at perfect accuracy measured information stays ≥ 1 nat (ε=2) with an excess that does not shrink with n. A valid variance bound is given in the report and held in all 19,200 checks.

## Reproduce
```bash
uv venv --python 3.12 && UV_CACHE_DIR=$PWD/.uv_cache uv sync
export NUMPY_MADVISE_HUGEPAGE=0 HF_HUB_OFFLINE=1
bash src/download_cifar.sh                         # AG News, SST-2 and BERT come from the HF cache
.venv/bin/python -I src/prepare_data.py ag sst cifar bert
.venv/bin/python -I src/test_units.py bert         # metric, MI-bound and BERT-port checks
# one run:
.venv/bin/python -I src/train.py --config ag_L6 --seed 0
.venv/bin/python -I src/mi_run.py --task order --eta 0.2 --n 2000 --seed 0
# whole grid (job list in logs/jobs/grid.txt; expects logs/jobs/ and results/{runs,mi}/):
.venv/bin/python -I src/queue_runner.py logs/jobs/grid.txt 12 20
.venv/bin/python -I src/analyze.py && .venv/bin/python -I src/figures.py && .venv/bin/python -I src/audit.py
```
The full grid took about 5 hours on four heavily shared GPUs.

## Files
- `REPORT.md`: full report. `planning.md`: pre-registered criteria and deviations. `literature_review.md`, `resources.md`.
- `src/`: `model.py` (encoder with attention access), `spectral.py` (metrics, MI bounds), `train.py`, `mi_run.py`, `prune_finetune.py`, `analyze.py`, `figures.py`, `audit.py`, `repro_check.py`, `queue_runner.py`.
- `results/`: tables and raw per-run logs. `figures/`: report figures.
