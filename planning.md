# Research plan: Spectral dynamics of attention matrices during training

Frozen before any model was trained (except a smoke test on throughput). Deviations are logged at the end.

## Motivation & Novelty Assessment

### Why This Research Matters
Attention matrices are used for interpretation, pruning and low-rank acceleration, yet most evidence about their
structure comes from *finished* models. If heads really split early into distinct spectral populations, a cheap
spectral statistic could guide early pruning and architecture choices.

### Gap in Existing Work
Head pruning work (Michel et al. 2019; Voita et al. 2019) ranks heads by loss sensitivity or learned gates, not by a
spectral criterion. Spectral work on attention concerns the *projection weights* or static/initial-state models
(Lin et al.'s Linformer spectrum of trained models; "Mind the Gap" at initialisation). Information-theoretic analyses
of deep nets warn that MI for deterministic networks is ill-posed (Saxe et al. 2018; Goldfeld et al. 2019), and we know
of no attention-specific bound of the stated form. No derivation of the hypothesised bound is in the workspace.

### Our Novel Contribution
Per-head, per-sample tracking of the input-conditional matrix A(x) over training, across text and vision, depth and
norm placement, with pre-registered numeric tests of five specific claims, and a controlled H(Y|X) test of the bound.

### Experiment Justification
- E1 Spectral trajectories (H1, H2, H5): the only way to observe bifurcation, entropy shapes and time to minimum.
- E2 Depth series L=3/6/12 (H5): "accelerates with depth" needs depth as a variable.
- E3 Controls: shuffled labels (does the dynamic need signal?), post-LN and pretrained BERT (is it a property of the setup?).
- E4 Entropy pruning against random, high-entropy and ablation-oracle baselines (H3).
- E5 MI grid over H(Y|X) (exact, via injected label noise) and n (H4).

## Research Question
Do attention matrices follow the five claimed dynamics, under stated, computable definitions?

## Definitions (primary)
A: head's n×n row-stochastic attention for one input, n = non-pad tokens incl. CLS. σ_i: singular values.
- ρ = AAᵀ/tr(AAᵀ); von Neumann entropy S = −Σ p_i log p_i, p_i = σ_i²/Σσ². Range [0, log n].
- **rank_eff = exp(S)** (primary). Secondary: stable rank ‖A‖²_F/σ₁², intrinsic (Σσ)²/Σσ², Roy–Vetterli exp H(σ/Σσ),
  numerical rank (σ>1%σ₁). Conclusions are checked under each. A head's value = mean over a fixed probe set of 256 dev inputs.
- Also: mean row Shannon entropy /log n, which separates uniform (rank 1, high row entropy) from pointer (rank 1, low).

## Hypothesis Decomposition and pre-registered criteria
Each is evaluated per configuration; "supported" requires the criterion in ≥80% of seeds.
- **H1 bifurcation.** At the 20% checkpoint: ≥10% of heads above 0.6n, ≥10% below 0.1n, ≥80% in the union, and dip test p<.05 on head values.
- **H2 entropy shape.** Layer-mean S(t). Early = first third of layers, late = last third.
  Monotone decrease: Spearman ρ(t,S) ≤ −0.8 and rebound (max above running min) ≤ 5% of range.
  U: minimum in 5–80% of training, drop and rebound each ≥ 0.02 log n, rebound ≥ 25% of drop.
- **H3 pruning.** Heads with mean S < 0.3 log n removed (output zeroed, no fine-tuning): test-accuracy loss, paired by seed, <2 pp with
  upper 95% CI <2 pp, *and* non-vacuous: at least 5% of heads pruned. Compared on identical counts to random, highest-entropy and ablation-oracle.
- **H4 bound.** I(A;X) estimated by Gaussian-noise channel pairwise bounds (Kolchinsky & Tracey), with lower bound Î_BD and upper bound Î_KL. Supported only if
  in regression Î = a + b H(Y|X) + c/√n, b's CI includes 1 and a ≤ small, and excess Î_BD − H(Y|X) shrinks like 1/√n. Refuted if the lower bound exceeds H(Y|X) by more than
  any plausible O(1/√n) term, or excess does not fall with n.
- **H5 depth.** t_min(l): first checkpoint with rank_eff(l) within 5% of the (start − min) drop of the minimum. t_min(1)/t_min(L) in [3,5] (CI overlaps) and a genuine fall (≥10% drop in both layers).

## Proposed Methodology
### Approach
Train real transformers from scratch (no simulation, no LLM API needed: internals are required, so local training is right).

### Experimental Steps
1. Prepare data: AG News (4-class, 120k, len≤64), SST-2 (phrases, len≤48), CIFAR-10 (ViT, 4×4 patches, n=65 fixed).
2. Custom encoder (d=192, 8 heads, d_k=24; CLS pooling; pre-LN) exposing A and a head mask.
3. Grid: {ag,sst,cifar} L=6, 8 seeds; {ag,cifar} L=3,12, 5 seeds; ag post-LN 5 seeds; {ag,cifar} shuffled labels 3 seeds; BERT fine-tune ag 3 seeds.
4. 41 checkpoints (dense over first 20%); log all metrics per head/sample in eval mode.
5. At end: pruning sweep (τ∈{.1…​.7}), baselines, short fine-tune as a secondary arm.
6. MI grid: synthetic order task (exact H(Y|X)=h₂(η)) and AG News, η∈{0,.1,.2,.3,.4}, n∈{500,2k,8k,32k}, 3 seeds.

### Baselines
Step-0 (untrained) spectra; shuffled labels; random, highest-entropy and oracle pruning.

### Evaluation Metrics
Accuracy; the metrics defined above; time-to-minimum.

### Statistical Analysis Plan
Seed is the unit of inference. Bootstrap (10k) CIs over seeds; Hartigan dip test; Holm correction within each hypothesis across configurations; α=.05.

## Expected Outcomes
From first principles: near-uniform init gives low S, so I expect rank to *rise* for some heads, which would invert "collapse". H4 should fail on the noise-free task.

## Timeline and Milestones
Setup 30 min; pilot 20 min; runs 2–3 h; analysis 45 min; docs 45 min.

## Potential Challenges
Shared GPUs (≤7 GB free): small models, one queue with memory check and retry. MI estimator ceiling log M: test by varying M. No hand-off of resources from the resource finder: literature gathered by hand.

## Success Criteria
Each of the five claims gets a clear verdict with effect sizes and CIs, including negative ones.

## Deviations from plan
1. (Pilot, no hypothesis data) MI noise grid moved from {.05–1} to {.2–4}: at small ε the estimate pins at the log M ceiling.
2. (After first 5 runs) Added `ag_L6_bigqk`, `cifar_L6_bigqk` (Q/K weights ×3 at init, 5 seeds). Not pre-registered. Reason: first runs showed near-uniform starts and rising rank, so
   "no collapse" could be an artefact of small default init. This arm tests that alternative explanation.
3. Literature gathered by hand because the resource finder was skipped.
4. Spearman of S with ablation importance added as a descriptive H3 statistic.
5. Fine-tuning moved from inside `train.py` to `src/prune_finetune.py` (uses saved weights; not available for BERT, whose weights are not saved).
6. Queue order changed mid-run so MI preceded the remaining depth repeats; GPU reservations lowered for MI (1.5→0.7 GB) and BERT (4→3 GB).
7. ε=2 chosen as the H4 reporting scale after seeing the data, as the smallest scale free of the log M ceiling. All scales are reported.
8. Rise-time ratios (time to peak) added for H5, because time-to-minimum is degenerate under default init.
