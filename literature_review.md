# Literature review (short, compiled by hand: the resource-finder stage was skipped)

**Head importance and pruning.** Michel, Levy & Neubig (NeurIPS 2019, "Are Sixteen Heads Really Better than One?") rank heads by
expected loss change on removal and report removing up to 60% without loss depending on task and model; single ablations mostly have little effect. Voita et al.
(ACL 2019) prune encoder heads with stochastic L0 gates (38/48 for 0.15 BLEU) and find specialised, confident heads go last. Neither uses a spectral
or entropy threshold, so H3's criterion is new.

**Spectra.** Linformer (Wang et al. 2020) reports long-tailed spectra of the trained Q-K matrix, more skewed in higher layers. "Mind the Gap" (arXiv 2410.07799) studies
rank collapse at initialisation through a spectral gap. Effective rank is defined inconsistently: entropy of normalised singular values (Roy & Vetterli 2007),
energy thresholds, etc. So conclusions must be checked across definitions.

**Information.** For deterministic nets MI with the input is constant (discrete X) or infinite (continuous X) (Saxe et al. 2018; Goldfeld et al. 2019), so any estimate reflects a
regulariser. Binning-based estimates are unreliable (Goldfeld et al.). The noise channel plus pairwise-distance mixture bounds (Kolchinsky & Tracey 2017) give a well-defined quantity.

**Lessons.** Use a defined noise scale, report it, and check the estimator ceiling; test against same-count pruning baselines; report the definitions.
