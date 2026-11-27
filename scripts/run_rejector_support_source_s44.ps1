$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Protocol-only comparison on identical frozen student checkpoints.
# The only changed factor is the known support used to fit the nnPU rejector:
# supervised training-known features (baseline) vs disjoint open-validation
# known features (--rejector-strict-mixed). Novel PU pool and threshold protocol
# remain unchanged. No model retraining and no clustering.

$Split = ".\splits_cifar100_protocols\cifar100_60_40_semantic_isolated.json"
$Common = @(
    "--dataset", "cifar100", "--data-root", ".\data", "--num-known", "60",
    "--seed", "44", "--split-path", $Split, "--known-split-mode", "random",
    "--image-size", "64", "--batch-size", "64", "--num-workers", "0",
    "--device", "auto", "--open-val-ratio", "0.1", "--calibration-ratio", "0.25",
    "--discovery-pool-mode", "mixed", "--matched-discovery-pool",
    "--matched-pool-size", "5400", "--matched-known-prior", "0.2",
    "--backbone", "resnet18", "--student-backbone", "resnet18",
    "--rejection-feature-dim", "128", "--num-novel", "40", "--mc-samples", "4",
    "--score-mode", "feature_rejector", "--rejector-training", "nnpu",
    "--rejector-nnpu-risk", "nu_corrected", "--rejector-nnpu-model", "linear",
    "--rejector-known-prior", "0.2", "--rejector-max-samples", "300",
    "--rejector-feature-mode", "support_augmented",
    "--threshold-policy", "known_coverage", "--target-known-coverage", "0.95",
    "--skip-clustering"
)

foreach ($Arm in @(
    @{ Tag = "margin_only" },
    @{ Tag = "margin_plus_uniform" }
)) {
    $Student = ".\runs\rej_uniform_lr01_openval_s44_$($Arm.Tag)_student\student.pt"
    if (-not (Test-Path $Student)) { throw "Missing checkpoint: $Student" }

    foreach ($Protocol in @(
        @{ Tag = "train_known_support"; Strict = $false },
        @{ Tag = "train_known_eval_transform"; Strict = $false; EvalTransform = $true },
        @{ Tag = "openval_known_support"; Strict = $true }
    )) {
        $Output = ".\runs\rejector_support_source_s44_matched300_$($Arm.Tag)_$($Protocol.Tag)"
        $Args = @("train.py", "discover") + $Common
        if ($Protocol.Strict) { $Args += "--rejector-strict-mixed" }
        if ($Protocol.EvalTransform) {
            $Args += @("--rejector-known-support-transform", "eval")
        }
        $Args += @("--work-dir", $Output, "--student-ckpt", $Student)
        & python @Args
        if ($LASTEXITCODE -ne 0) {
            throw "Discovery failed for $($Arm.Tag), $($Protocol.Tag)"
        }
    }
}

Write-Host "Completed frozen-checkpoint rejector support-source comparison."
