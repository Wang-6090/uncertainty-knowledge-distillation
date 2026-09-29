# Mixed discovery-pool protocol audit (2026-09-29)

## Problem found

The original `discovery_pool_mode=mixed` implementation concatenated
`known_train` and the novel pool. Therefore the same known samples could
receive supervised classification gradients and also be treated as unlabeled
discovery images in the same run. This was not test leakage, but it made the
mixed-pool comparison less clean and changed the amount of supervised data.

## Code change

Mixed mode now reserves a disjoint subset of the known training partition for
the known part of the unlabeled discovery pool. The supervised train subset and
the reserved known discovery subset are selected with the same random or
stratified policy and never share sample IDs. Validation and official test
samples are untouched.

New option:

```powershell
--mixed-known-pool-ratio 0.2
```

The option is only meaningful for `--discovery-pool-mode mixed` and must be in
`(0, 1)`. It defaults to 0.2. The value is recorded in each run's config.

## Validation

- Added a unit test that checks supervised-known and unlabeled-known IDs are
  disjoint and that both known and novel samples occur in the mixed pool.
- Added a unit test that rejects the old unsafe API call without an explicit
  disjoint known discovery subset.
- `python -m pytest -q tests --disable-warnings --maxfail=1`: **86 passed**.
- Python compilation passed.
- Toy `inspect_data` with mixed mode and a 0.25 reserve ratio completed.

## Experimental consequence

Existing mixed-pool results remain useful as historical diagnostics, but they
should not be compared as final fair results with the corrected protocol. The
next paired experiment must retrain the mixed model with this corrected data
construction. Keep pool size, seed, teacher checkpoint, student settings,
epochs, loss weights, detector, and threshold policy fixed; only compare the
corrected disjoint mixed pool against the corrected no-discovery baseline and,
separately, the oracle novel-only upper-bound setting.
