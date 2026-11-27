$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Exact seed-123 replication of run_nnpu_encoder_gradient_s44.ps1.
# The teacher is retrained with the current CIFAR transform/protocol because
# the older seed-123 teacher was created before that configuration was saved.

$TeacherDir = ".\runs\semantic_isolated_full_s123_disjointcal_teacher"
$Teacher = Join-Path $TeacherDir "teacher.pt"
$TeacherArgs = @(
    "--dataset", "cifar100", "--data-root", ".\data", "--num-known", "60",
    "--seed", "123", "--split-path", ".\splits_cifar100_protocols\cifar100_60_40_semantic_isolated.json",
    "--known-split-mode", "random", "--transform-profile", "cifar",
    "--image-size", "64", "--batch-size", "64", "--num-workers", "0",
    "--device", "auto", "--calibration-ratio", "0.25", "--backbone", "resnet34",
    "--pretrained", "--epochs", "5"
)
if (-not (Test-Path $Teacher)) {
    & python train.py train_teacher @TeacherArgs --work-dir $TeacherDir --teacher-ckpt $Teacher
    if ($LASTEXITCODE -ne 0) { throw "teacher training failed" }
}

$Common = @(
    "--dataset", "cifar100", "--data-root", ".\data", "--num-known", "60",
    "--seed", "123", "--split-path", ".\splits_cifar100_protocols\cifar100_60_40_semantic_isolated.json",
    "--known-split-mode", "random", "--transform-profile", "cifar",
    "--image-size", "64", "--batch-size", "64", "--num-workers", "0", "--device", "auto",
    "--calibration-ratio", "0.25", "--backbone", "resnet18", "--student-backbone", "resnet18",
    "--teacher-backbone", "resnet34", "--pretrained", "--epochs", "5",
    "--encoder-lr-scale", "0.1", "--alpha-unc", "0.1", "--alpha-kd", "1.0",
    "--kd-mode", "uncertainty", "--temperature", "2.0", "--uncertainty-weight-mode", "raw",
    "--alpha-feat-kd", "0.1", "--alpha-proto", "0.1", "--alpha-supcon", "0.1",
    "--alpha-discovery-uncertainty-pu", "0.1", "--discovery-uncertainty-pu-risk", "nu_corrected",
    "--discovery-uncertainty-known-prior", "0.2", "--discovery-pool", "--discovery-pool-mode", "mixed",
    "--matched-discovery-pool", "--matched-pool-size", "5400", "--matched-known-prior", "0.2"
)
$Discover = @(
    "--dataset", "cifar100", "--data-root", ".\data", "--num-known", "60",
    "--seed", "123", "--split-path", ".\splits_cifar100_protocols\cifar100_60_40_semantic_isolated.json",
    "--known-split-mode", "random", "--transform-profile", "cifar",
    "--image-size", "64", "--batch-size", "64", "--num-workers", "0", "--device", "auto",
    "--calibration-ratio", "0.25", "--backbone", "resnet18", "--student-backbone", "resnet18",
    "--pretrained", "--num-novel", "40", "--mc-samples", "4",
    "--score-mode", "normalized_entropy_min_class_knn", "--knn-ood",
    "--threshold-policy", "known_coverage", "--target-known-coverage", "0.95", "--skip-clustering"
)

foreach ($Arm in @("encoder_grad", "head_only")) {
    $StudentDir = ".\runs\nnpu_encoder_gradient_s123_${Arm}_student"
    $Student = Join-Path $StudentDir "student.pt"
    $ArmArgs = @()
    if ($Arm -eq "head_only") { $ArmArgs += "--discovery-uncertainty-pu-detach-features" }
    if (-not (Test-Path $Student)) {
        & python train.py train_student @Common @ArmArgs --work-dir $StudentDir --teacher-ckpt $Teacher --student-ckpt $Student
        if ($LASTEXITCODE -ne 0) { throw "student training failed for $Arm" }
    }
    $DiscoverDir = ".\runs\nnpu_encoder_gradient_s123_${Arm}_discover"
    & python train.py discover @Discover --work-dir $DiscoverDir --student-ckpt $Student
    if ($LASTEXITCODE -ne 0) { throw "discovery failed for $Arm" }
}

Write-Host "Completed seed-123 nnPU encoder-gradient isolation comparison."
