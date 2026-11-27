$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Controlled follow-up to the three-seed rejection-branch OE pilot.
# The only treatment factor is alpha_outlier_rejection_uniform: 0 vs 0.05.
# Both arms use encoder_lr_scale=0.1 and a disjoint open-validation subset.

$Teacher = ".\runs\semantic_isolated_full_s44_disjointcal_teacher\teacher.pt"
$Split = ".\splits_cifar100_protocols\cifar100_60_40_semantic_isolated.json"
$Cifar10Marker = ".\data\cifar-10-batches-py\data_batch_1"
if (-not (Test-Path $Teacher)) { throw "Missing teacher checkpoint: $Teacher" }
if (-not (Test-Path $Cifar10Marker)) { throw "CIFAR-10 OE data unavailable: $Cifar10Marker" }

$TrainCommon = @(
    "--dataset", "cifar100", "--data-root", ".\data", "--num-known", "60",
    "--seed", "44", "--split-path", $Split, "--known-split-mode", "random",
    "--image-size", "64", "--batch-size", "64", "--num-workers", "0",
    "--device", "auto", "--open-val-ratio", "0.1",
    "--backbone", "resnet18", "--teacher-backbone", "resnet34",
    "--student-backbone", "resnet18", "--pretrained", "--epochs", "5",
    "--alpha-unc", "0.1", "--alpha-kd", "1.0", "--kd-mode", "uncertainty",
    "--temperature", "2.0", "--uncertainty-weight-mode", "raw",
    "--alpha-feat-kd", "0.1", "--alpha-proto", "0.1", "--alpha-supcon", "0.1",
    "--alpha-discovery-uncertainty-pu", "0.1",
    "--discovery-uncertainty-pu-risk", "nu_corrected",
    "--discovery-uncertainty-known-prior", "0.2",
    "--discovery-pool", "--discovery-pool-mode", "mixed",
    "--matched-discovery-pool", "--matched-pool-size", "5400",
    "--matched-known-prior", "0.2", "--rejection-feature-dim", "128",
    "--alpha-rejection-known-ce", "1.0", "--outlier-dataset", "cifar10",
    "--outlier-data-root", ".\data", "--alpha-outlier-rejection-feature-margin", "0.05",
    "--outlier-rejection-feature-margin", "0.2"
)
$DiscoverCommon = @(
    "--dataset", "cifar100", "--data-root", ".\data", "--num-known", "60",
    "--seed", "44", "--split-path", $Split, "--known-split-mode", "random",
    "--image-size", "64", "--batch-size", "64", "--num-workers", "0",
    "--device", "auto", "--open-val-ratio", "0.1", "--calibration-ratio", "0.25",
    "--discovery-pool-mode", "mixed",
    "--matched-discovery-pool", "--matched-pool-size", "5400",
    "--matched-known-prior", "0.2",
    "--backbone", "resnet18", "--student-backbone", "resnet18",
    "--rejection-feature-dim", "128", "--num-novel", "40", "--mc-samples", "4",
    "--score-mode", "feature_rejector", "--rejector-training", "nnpu",
    "--rejector-nnpu-risk", "nu_corrected", "--rejector-nnpu-model", "linear",
    "--rejector-known-prior", "0.2", "--rejector-feature-mode", "support_augmented",
    "--threshold-policy", "known_coverage", "--target-known-coverage", "0.95",
    "--skip-clustering"
)

foreach ($Arm in @(
    @{ Tag = "margin_only"; UniformWeight = "0.0" },
    @{ Tag = "margin_plus_uniform"; UniformWeight = "0.05" }
)) {
    $StudentDir = ".\runs\rej_uniform_lr01_openval_s44_$($Arm.Tag)_student"
    $Student = Join-Path $StudentDir "student.pt"
    $DiscoverDir = ".\runs\rej_uniform_lr01_openval_s44_$($Arm.Tag)_discover"

    if (-not (Test-Path $Student)) {
        & python train.py train_student @TrainCommon `
            --encoder-lr-scale 0.1 `
            --alpha-outlier-rejection-uniform $Arm.UniformWeight `
            --work-dir $StudentDir --teacher-ckpt $Teacher --student-ckpt $Student
        if ($LASTEXITCODE -ne 0) { throw "Student training failed: $($Arm.Tag)" }
    } else {
        Write-Host "Reusing completed student checkpoint: $Student"
    }

    & python train.py discover @DiscoverCommon `
        --work-dir $DiscoverDir --student-ckpt $Student
    if ($LASTEXITCODE -ne 0) { throw "Discovery failed: $($Arm.Tag)" }
}

Write-Host "Completed paired seed-44 rejection-OE follow-up. Only the uniform-OE weight differs."
