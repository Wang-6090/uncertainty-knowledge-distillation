param(
    [int]$Seed = 42,
    [int]$Epochs = 10,
    [string]$Tag = "uncertainty_nnpu_weight_sweep",
    [double[]]$Weights = @(0.3, 0.5)
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_protocols\cifar100_60_40_semantic_isolated.json"
$Prefix = "${Tag}_s${Seed}"
$TeacherCkpt = ".\runs\uncertainty_2x2_e10_s42_teacher\teacher.pt"

if (-not (Test-Path $TeacherCkpt)) {
    throw "Shared teacher checkpoint not found: $TeacherCkpt"
}

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

foreach ($Weight in $Weights) {
    $Label = ("w{0:0.###}" -f $Weight).Replace('.', 'p')
    $StudentRun = ".\runs\${Prefix}_${Label}"
    $StudentCkpt = Join-Path $StudentRun "student.pt"
    $DetectRun = ".\runs\${Prefix}_${Label}_detect"

    Write-Host "=== Student: nnPU weight $Weight ==="
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
        "--alpha-discovery-uncertainty-pu", "$Weight",
        "--discovery-uncertainty-pu-risk", "nu_corrected",
        "--discovery-uncertainty-known-prior", "0.2",
        "--work-dir", $StudentRun,
        "--teacher-ckpt", $TeacherCkpt,
        "--student-ckpt", $StudentCkpt
    ))

    Write-Host "=== Detector: nnPU weight $Weight ==="
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

Write-Host "nnPU weight sweep completed. Compare discovery_report.json under runs/${Prefix}_*_detect/."
