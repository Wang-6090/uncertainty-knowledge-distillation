$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Controlled pilot: compare the existing full-data random-split student with
# the same teacher/training protocol plus a small CIFAR-10 OE uniform loss.
# CIFAR-100 novel labels are not used during training or calibration.
$Tag = "protocol_fullpilot_random_s42_oe_cifar10_u005"
$Teacher = ".\runs\protocol_fullpilot_random_s42_teacher\teacher.pt"
$StudentDir = ".\runs\${Tag}_student"
$Student = Join-Path $StudentDir "student.pt"
$DetectDir = ".\runs\${Tag}_detect"

if (-not (Test-Path $Teacher)) {
    throw "Required baseline teacher checkpoint not found: $Teacher"
}
if (-not (Test-Path ".\data\cifar-10-batches-py\data_batch_1")) {
    throw "CIFAR-10 is not available under .\data; this pilot intentionally does not download data."
}

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: python $($Arguments -join ' ')"
    }
}

$Common = @(
    "--dataset", "cifar100",
    "--data-root", ".\data",
    "--num-known", "60",
    "--seed", "42",
    "--split-path", ".\splits_cifar100_protocols\cifar100_60_40_random.json",
    "--known-split-mode", "random",
    "--image-size", "64",
    "--batch-size", "64",
    "--num-workers", "0",
    "--device", "auto"
)

Write-Host "=== Train student with CIFAR-10 OE uniform loss (weight=0.05) ==="
Invoke-CheckedPython (@("train.py", "train_student") + $Common + @(
    "--backbone", "resnet18",
    "--teacher-backbone", "resnet34",
    "--pretrained",
    "--epochs", "5",
    "--alpha-unc", "0.1",
    "--alpha-kd", "1.0",
    "--kd-mode", "uncertainty",
    "--temperature", "2.0",
    "--uncertainty-weight-mode", "raw",
    "--alpha-feat-kd", "0.1",
    "--alpha-proto", "0.1",
    "--alpha-supcon", "0.1",
    "--outlier-dataset", "cifar10",
    "--outlier-data-root", ".\data",
    "--alpha-outlier-uniform", "0.05",
    "--work-dir", $StudentDir,
    "--teacher-ckpt", $Teacher,
    "--student-ckpt", $Student
))

Write-Host "=== Detect with the exact full-pilot detector/calibration ==="
Invoke-CheckedPython (@("train.py", "discover") + $Common + @(
    "--backbone", "resnet18",
    "--student-backbone", "resnet18",
    "--num-novel", "40",
    "--mc-samples", "4",
    "--score-mode", "normalized_entropy_mahalanobis",
    "--threshold-policy", "known_coverage",
    "--target-known-coverage", "0.95",
    "--skip-clustering",
    "--work-dir", $DetectDir,
    "--student-ckpt", $Student
))

Write-Host "OE pilot completed: $Tag"
