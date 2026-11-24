# Why recent KD candidates have not produced stable discovery gains

## Scope

This retrospective is limited to Team A's student classification
representation and KD mechanisms. It does not recommend changes to the
rejector, candidate-selection rule, score fusion, or clustering algorithm.
The evidence combines the formal three-seed CE/KD audit and recent fresh-split
pilots; one-seed pilots are treated as diagnostics, not estimates of stable
effects.

## What the evidence says

| Candidate/evidence | What changed mechanistically | What happened downstream |
|---|---|---|
| Standard KD, formal seeds 42/123/3407 | Real KD loss, gradient, and student updates; known-class probe only `+0.003333` over CE on average, improving on 2/3 seeds | Versus CE-only, mean AUROC `-0.007996`, AUPR `-0.008632`, unknown rejection `-1.158 pp`; clustering changes were mixed |
| Uncertainty KD, formal seeds 42/123/3407 | Detached teacher weights were applied in 12/12 epochs; weight SD `0.200781`, minimum ESS fraction `0.953333`, with nonzero KD gradients and student updates | Versus standard KD, AUROC/AUPR improved only `+0.004787/+0.006098`; unknown rejection `-0.242 pp`, and ACC/NMI decreased on average |
| Quality-aware KD, seeds 2155/2156 | Teacher true-class confidence weighting was active | Seed directions conflicted; mean unknown rejection `+0.775 pp` with SD `11.278 pp`, mean AUROC `+0.006310` with SD `0.020645`; no joint stable gain |
| Centered feature KD, seed 2135 | Nonzero within-class feature loss and encoder gradient in every epoch | AUROC `-0.008495`, unknown rejection `-2.700 pp`, candidate purity `-2.798 pp`; known feature geometry did not improve coherently |
| Class-center feature KD, seed 2137 | Nonzero class-center loss and encoder gradient in every epoch | Unknown rejection `+3.250 pp`, but AUROC `-0.005571`, candidate purity `-1.741 pp`, NMI `-3.580 pp`, and Auto-K error doubled |
| Selective KD, confirmation seeds 2163/2164 | Teacher-correct gate and student gradients were active in both seeds | Clustering improved, but unknown rejection and candidate purity fell in both seeds; ranking changes were seed-sensitive |
| SRD cross-network KD, seed 2170 | Positive KD loss/encoder gradient every epoch; teacher critic detached | Raw feature norm rose `33.7%`; normalized within-class distance fell and between/total scatter rose, but local center margin was unchanged; most detection/candidate/cluster metrics improved while NMI fell `1.531 pp` |

The formal CE-only evidence and detailed pilot records are in
[`team_a_formal_ce_kd_20261122.md`](../../analysis/team_a_formal_ce_kd_20261122.md),
[`selective_kd_confirmation_20261122.md`](selective_kd_confirmation_20261122.md),
and [`srd_kd_pilot_20261123.md`](srd_kd_pilot_20261123.md). The machine
results are local and gitignored by repository policy.

## Most likely explanations

1. **The training target does not contain the missing unknown structure.**
   CE and logits KD teach the student to preserve the teacher's decisions on
   the 60 known classes. Uncertainty and teacher-correctness weights only
   reallocate that same known-sample transfer. Center losses and SRD likewise
   reshape known-class features or teacher-classifier outputs. None directly
   specifies how the 40 unseen semantic-hard classes should arrange into
   clusters or separate from known support. A genuine training gradient is
   therefore necessary evidence, but not evidence that it optimizes the
   discovery objective.

2. **Teacher quality for known classification is not teacher quality for
   novelty geometry.** The corrected CIFAR-transform support audit found
   teacher prototype/centroid-distance AUROC around `0.61-0.63`, below the
   standard student's roughly `0.67-0.68` on the same confirmation splits.
   This weakens the assumption that copying teacher geometry will necessarily
   help unknown detection. The teacher can classify known samples well while
   having poor support geometry for unseen classes.

3. **Most interventions change a local property, not the whole representation
   structure.** SRD increased raw feature norm and improved known-validation
   between/total scatter, but normalized within-class and nearest-center
   distances both contracted while their margin stayed nearly fixed. Other
   feature transfers moved within-class compactness, known accuracy,
   rejection, candidate purity, and clustering in different directions. This
   pattern is consistent with a partial geometric change that does not
   guarantee a better global partition; it is not proof of a specific causal
   mechanism.

4. **The evaluated endpoint is a cascade with distinct objectives.** Ranking
   metrics (AUROC/AUPR), rejection at a fixed known-coverage threshold,
   candidate purity/recall, Auto-K, and clustering measure different failure
   modes. Across prior work, ranking often moved without unknown rejection or
   clustering moving with it. Reporting a single favorable metric would hide
   this disconnect. Thresholding at 95% known coverage also intentionally
   constrains known acceptance, so a small score shift need not increase
   unknown rejection.

5. **Uncertainty reweighting is active but only mildly nonuniform and may not
   rank the relevant errors.** The high ESS fractions (about `0.95` in the
   formal audit) show that mean-normalized weights preserve a near-uniform
   effective batch. This is not an implementation failure. It suggests the
   current uncertainty signal may be weakly aligned with which teacher targets
   help student representation or with which samples define the unknown
   boundary. The separate teacher uncertainty, student uncertainty, entropy,
   and final rejection scores must remain separate; conflating them would not
   solve this alignment question.

6. **The problem is seed-sensitive and the class splits are not fully
   independent evidence.** Quality-aware and selective pilots show opposite
   metric directions across fresh class combinations. Formal seed means are
   useful but class sets overlap, and single-seed pilots cannot distinguish a
   method effect from split-specific geometry. This explains why textbook
   methods can look plausible in one run yet fail confirmation without
   implying their original closed-set claims are invalid.

## Decision and next step

The repeated failure is not evidence that KD gradients are absent. For the
recent audited candidates, the losses did reach and update the student. The
larger gap is between **known-class imitation/compactness objectives** and the
actual **unseen-class support, candidate, and partition objectives**.

Do not continue scanning coefficients, temperatures, uncertainty sources, or
feature-distance variants for the frozen candidates. Keep standard logits KD
as the clean comparison baseline; retain uncertainty-KD and the experimental
variants as opt-in ablations, not established improvements.

The read-only standard/SRD audit is complete and saved as the local,
gitignored `analysis/team_a_srd_geometry_20261123_s2170.json`. It found a
known-validation representation signal, but the unchanged local center margin
and lower NMI leave uncertainty about whether that signal generalizes. The
predeclared next step is therefore a fixed two-seed SRD confirmation on fresh
seeds `2171/2172`, without any hyperparameter or score scan. If representation
and downstream gains do not reproduce together, freeze SRD. Do not use
unknown/test labels to design its weight, choose the checkpoint, or alter the
confirmation protocol.
