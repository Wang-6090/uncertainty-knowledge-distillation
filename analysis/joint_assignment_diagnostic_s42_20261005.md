# Joint Assignment Diagnostic (2026-10-05)

## Question

The joint novel loss stayed close to `log(K)` in previous runs. This can mean that the
novel head is not learning meaningful assignments, even when the scalar loss decreases.
The code now reports assignment entropy, mean maximum probability, and the number of
prototype indices used in each batch. These are diagnostics only and do not change the
gradient.

## Batch-size check

The fixed joint-random protocol was rerun with `discovery_batch_size=512`, while the
training batch remained 64. This tests whether K=40 is simply too large for a 64-sample
batch.

| Setting | AUROC | FPR95 | OSCR | Known acceptance | Unknown rejection | Candidate purity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint random, discovery batch 64 | 0.5051 | 0.9359 | 0.0648 | 0.9704 | 0.0561 | 0.5500 |
| Joint random, discovery batch 512 | 0.4973 | 0.9161 | 0.0631 | 0.9655 | 0.0459 | 0.4615 |

The larger batch lowers FPR95 in this seed but lowers AUROC, OSCR slightly, unknown
rejection, and candidate purity. The joint loss remains about `3.685`, very close to
`log(40)=3.6889`; merely increasing the batch is not sufficient.

## Diagnostic smoke

On a toy K=4 run, the new fields were:

- assignment entropy: `1.3772` versus uniform `log(4)=1.3863`;
- mean maximum probability: `0.2896` versus uniform `0.25`;
- active prototype indices: `2.75` out of 4.

This confirms near-uniform, partially collapsed assignments rather than a well-formed
novel partition. The next implementation should use a global cross-batch assignment or
memory-bank target, and must report occupancy and assignment stability before claiming
that novel discovery has improved.
