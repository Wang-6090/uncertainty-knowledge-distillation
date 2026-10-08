param(
    [int]$Seed = 42,
    [int]$Epochs = 3,
    [string]$Tag = "uncertainty_2x2_pilot"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Component screening only: keep every arm paired and use one shared teacher.
$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_protocols\cifar100_60_40_semantic_isolated.json"
$Prefix = "${Tag}_s${Seed}"
$TeacherRun = ".\runs\${Prefix}_teacher"
$TeacherCkpt = Join-Path $TeacherRun "teacher.pt"

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: python $($Arguments -join ' ')"
    }
}

$TrainCommon = @(
    "--dataset", "cifar100",
    "--data-root", $DataRoot,
    "--num-known", "60",
    "--seed", "$Seed",
    "--split-path", $SplitPath,
    "--known-split-mode", "random",
    "--image-size", "64",
    "--batch-size", "64",
    "--num-workers", "0",
    "--epochs", "$Epochs",
    "--limit-train", "1200",
    "--limit-val", "300",
    "--limit-test", "1000",
    "--device", "auto"
)

$DiscoverCommon = @(
    "--dataset", "cifar100",
    "--data-root", $DataRoot,
    "--num-known", "60",
    "--seed", "$Seed",
    "--split-path", $SplitPath,
    "--known-split-mode", "random",
    "--image-size", "64",
    "--batch-size", "64",
    "--num-workers", "0",
    "--limit-train", "1200",
    "--limit-val", "300",
    "--limit-test", "1000",
    "--device", "auto"
)

Write-Host "=== Shared teacher: $Prefix ==="
Invoke-CheckedPython (@("train.py", "train_teacher") + $TrainCommon + @(
    "--backbone", "resnet34",
    "--teacher-backbone", "resnet34",
    "--pretrained",
    "--alpha-unc", "0.1",
    "--work-dir", $TeacherRun,
    "--teacher-ckpt", $TeacherCkpt
))

$Arms = @(
    @{ Name = "baseline"; Extra = @() },
    @{ Name = "nnpu"; Extra = @(
        "--alpha-discovery-uncertainty-pu", "0.1",
        "--discovery-uncertainty-pu-risk", "nu_corrected",
        "--discovery-uncertainty-known-prior", "0.2"
    ) },
    @{ Name = "feature_margin"; Extra = @(
        "--alpha-discovery-uncertainty-feature-margin", "0.05",
        "--discovery-uncertainty-feature-margin", "0.2"
    ) },
    @{ Name = "combined"; Extra = @(
        "--alpha-discovery-uncertainty-pu", "0.1",
        "--discovery-uncertainty-pu-risk", "nu_corrected",
        "--discovery-uncertainty-known-prior", "0.2",
        "--alpha-discovery-uncertainty-feature-margin", "0.05",
        "--discovery-uncertainty-feature-margin", "0.2"
    ) }
)

foreach ($Arm in $Arms) {
    $StudentRun = ".\runs\${Prefix}_$($Arm.Name)"
    $StudentCkpt = Join-Path $StudentRun "student.pt"
    Write-Host "=== Student: $($Arm.Name) ==="
    Invoke-CheckedPython (@("train.py", "train_student") + $TrainCommon + @(
        "--backbone", "resnet18",
        "--student-backbone", "resnet18",
        "--teacher-backbone", "resnet34",
        "--pretrained",
        "--alpha-unc", "0.1",
        "--alpha-kd", "1.0",
        "--kd-mode", "uncertainty",
        "--temperature", "2.0",
        "--uncertainty-weight-mode", "raw",
        "--alpha-feat-kd", "0.1",
        "--alpha-proto", "0.1",
        "--alpha-supcon", "0.1",
        "--discovery-pool",
        "--discovery-pool-mode", "mixed",
        "--matched-discovery-pool",
        "--matched-pool-size", "5400",
        "--matched-known-prior", "0.2",
        "--work-dir", $StudentRun,
        "--teacher-ckpt", $TeacherCkpt,
        "--student-ckpt", $StudentCkpt
    ) + $Arm.Extra)

    $DetectRun = ".\runs\${Prefix}_$($Arm.Name)_detect"
    Write-Host "=== Detector: $($Arm.Name) ==="
    Invoke-CheckedPython (@("train.py", "discover") + $DiscoverCommon + @(
        "--backbone", "resnet18",
        "--student-backbone", "resnet18",
        "--num-novel", "40",
        "--mc-samples", "4",
        "--score-mode", "feature_rejector",
        "--rejector-training", "nnpu",
        "--rejector-nnpu-risk", "nu_corrected",
        "--rejector-known-prior", "0.2",
        "--rejector-feature-mode", "support_augmented",
        "--rejector-mc-samples", "1",
        "--rejector-max-samples", "5000",
        "--threshold-policy", "known_coverage",
        "--target-known-coverage", "0.95",
        "--discovery-pool-mode", "mixed",
        "--matched-discovery-pool",
        "--matched-pool-size", "5400",
        "--matched-known-prior", "0.2",
        "--skip-clustering",
        "--work-dir", $DetectRun,
        "--student-ckpt", $StudentCkpt
    ))
}

Write-Host "2x2 uncertainty pilot completed. Compare discovery_report.json under runs/${Prefix}_*_detect/."
