# Class-balanced uncertainty-KD weight audit

## Question and scope

The frozen-teacher audit found that uncertainty is associated with teacher
error both within and between known classes. This raised a narrow mechanism
question: would normalizing `exp(-u_teacher)` separately within each known
class preserve a useful within-class signal while reducing cross-class weight
differences? This was evaluated only as a batch-weight simulation. No training
loss, default configuration, rejection score, candidate-selection procedure,
Auto-K method, or clustering algorithm was changed.

## Procedure

For each frozen teacher, deterministic known train and known validation
inference produced teacher uncertainty and mapped known labels. Four shuffled
epochs were simulated with batch size 64, matching the student pilot batch
size. The baseline weight in each batch was
`exp(-u) / mean_batch(exp(-u))`. The diagnostic candidate first divided
`exp(-u)` by its mean among same-label examples in that batch, then
mean-normalized over the batch. We report mean batch weight standard deviation,
mean batch ESS fraction, and the fraction of sample exposures belonging to a
class represented only once in its batch. This simulation is a mechanism
screen, not a training result.

Known labels were used only to group known training/validation examples for
this post-hoc diagnostic. Unknown/test labels were not loaded or used.

## Results

| Seed / split | Class difficulty vs mean uncertainty Spearman | Within-class error AUROC | Singleton exposure | Baseline weight SD / ESS | Class-balanced SD / ESS |
|---|---:|---:|---:|---:|---:|
| 2173 train | 0.723 | 0.701 | 34.8% | 0.226 / 0.951 | 0.111 / 0.987 |
| 2173 validation | 0.690 | 0.711 | 34.2% | 0.225 / 0.952 | 0.112 / 0.987 |
| 2176 train | 0.833 | 0.673 | 34.5% | 0.188 / 0.966 | 0.092 / 0.991 |
| 2176 validation | 0.844 | 0.688 | 35.0% | 0.189 / 0.965 | 0.095 / 0.991 |

The within-class association is nontrivial but weaker than global error ranking
(global uncertainty-error AUROC is `0.747/0.751` on seed 2173 train/validation
and `0.752/0.756` on seed 2176). Class difficulty rankings are stable across
train/validation: error-rate Spearman is `0.947` and `0.961`; mean-uncertainty
Spearman is `0.962` and `0.975` for seeds 2173 and 2176, respectively.

## Decision

Do not implement or train the class-balanced candidate. It roughly halves
weight dispersion and moves ESS close to 1, while around one third of sample
exposures are singleton-class cases that cannot receive a within-class
relative weight before batch renormalization. CIFAR-100 known classes are
balanced in this protocol, so there is no class-frequency imbalance to
correct; class-level teacher difficulty is itself predictive of error and
would be deliberately removed. These results do not establish that preserving
class-level difficulty is optimal, but they give no positive reason to erase
it with this rule.

The current detached `exp(-u)` uncertainty-KD stays opt-in and unchanged. No
performance claim is made because no student was trained. The two machine
reports are local gitignored diagnostics:
`analysis/team_a_teacher_uncertainty_batch_balance_20261125_s2173.json` and
`analysis/team_a_teacher_uncertainty_batch_balance_20261125_s2176.json`.
