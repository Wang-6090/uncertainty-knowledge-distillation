param(
    [int]$Seed = 42,
    [switch]$FullData,
    [switch]$RejectorStratifiedKnown,
    [int]$RejectorMaxSamples = 5000,
    [int]$EpochsOverride = 0,
    [string]$RunTag = "",
    [switch]$KnownCenterMarginTreatment
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# A short, matched pilot for independent semantic class-split validation.
# Test labels are used only by train.py for final metrics.
$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_protocols\cifar100_60_40_semantic_isolated.json"
$Prefix = if ($FullData) { "semantic_isolated_full" } else { "semantic_isolated_pilot" }
if (-not [string]::IsNullOrWhiteSpace($RunTag)) {
    $Prefix = "${Prefix}_${RunTag}"
}
$Epochs = if ($FullData) { 5 } else { 3 }
if ($EpochsOverride -gt 0) {
    $Epochs = $EpochsOverride
}
$LimitTrain = if ($FullData) { 0 } else { 1200 }
$LimitVal = if ($FullData) { 0 } else { 300 }
$LimitTest = if ($FullData) { 0 } else { 1000 }
$TeacherRun = ".\runs\${Prefix}_s${Seed}_teacher"
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
    "--limit-train", "$LimitTrain",
    "--limit-val", "$LimitVal",
    "--limit-test", "$LimitTest",
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
    "--limit-train", "$LimitTrain",
    "--limit-val", "$LimitVal",
    "--limit-test", "$LimitTest",
    "--device", "auto"
)
$RejectorSamplingArgs = @(
    "--rejector-max-samples", "$RejectorMaxSamples"
)
if ($RejectorStratifiedKnown) {
    $RejectorSamplingArgs += "--rejector-stratified-known"
}

Write-Host "=== Shared teacher ==="
Invoke-CheckedPython (@("train.py", "train_teacher") + $TrainCommon + @(
    "--backbone", "resnet34",
    "--pretrained",
    "--alpha-unc", "0.1",
    "--work-dir", $TeacherRun,
    "--teacher-ckpt", $TeacherCkpt
))

$StudentArms = @(
    @{ Name = "baseline"; Extra = @() },
    @{ Name = "treatment"; Extra = @(
        "--alpha-discovery-uncertainty-pu", "0.1",
        "--discovery-uncertainty-pu-risk", "nu_corrected",
        "--discovery-uncertainty-known-prior", "0.2",
        "--alpha-discovery-uncertainty-feature-margin", "0.05",
        "--discovery-uncertainty-feature-margin", "0.2"
    ) }
)
if ($KnownCenterMarginTreatment) {
    $StudentArms += @{ Name = "center_margin"; Extra = @(
        "--alpha-center-margin", "0.02",
        "--center-margin", "0.1"
    ) }
}

foreach ($Arm in $StudentArms) {
    $StudentRun = ".\runs\${Prefix}_s${Seed}_$($Arm.Name)"
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

    $DetectRun = ".\runs\${Prefix}_s${Seed}_$($Arm.Name)_rejector_support"
    Write-Host "=== Support-only nnPU rejector: $($Arm.Name) ==="
    Invoke-CheckedPython ((@("train.py", "discover") + $DiscoverCommon + @(
        "--backbone", "resnet18",
        "--student-backbone", "resnet18",
        "--num-novel", "40",
        "--mc-samples", "4",
        "--score-mode", "feature_rejector",
        "--rejector-training", "nnpu",
        "--rejector-nnpu-risk", "nu_corrected",
        "--rejector-known-prior", "0.2",
        "--rejector-feature-mode", "support_augmented",
        "--rejector-mc-samples", "1"
    ) + $RejectorSamplingArgs + @(
        "--threshold-policy", "known_coverage",
        "--target-known-coverage", "0.95",
        "--discovery-pool-mode", "mixed",
        "--matched-discovery-pool",
        "--matched-pool-size", "5400",
        "--matched-known-prior", "0.2",
        "--skip-clustering",
        "--work-dir", $DetectRun,
        "--student-ckpt", $StudentCkpt
    )))
}

Write-Host "Semantic-isolated pilot completed. Compare discovery_report.json files under runs/."
