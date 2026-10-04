$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Representation-preservation pilot. Reuse the existing seed-42 teacher and
# compare the historical student optimizer (encoder lr=1.0x) with a student
# whose pretrained encoder uses 0.1x while all heads keep the base lr.
$StudentDir = ".\runs\protocol_fullpilot_random_s42_encoder_lr01_student"
$Student = Join-Path $StudentDir "student.pt"
$DetectDir = ".\runs\protocol_fullpilot_random_s42_encoder_lr01_detect"
$Teacher = ".\runs\protocol_fullpilot_random_s42_teacher\teacher.pt"

if (-not (Test-Path $Teacher)) {
    throw "Required teacher checkpoint not found: $Teacher"
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

Write-Host "=== Train student with encoder lr scale 0.1 ==="
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
    "--encoder-lr-scale", "0.1",
    "--work-dir", $StudentDir,
    "--teacher-ckpt", $Teacher,
    "--student-ckpt", $Student
))

Write-Host "=== Detect with the fixed full-pilot detector/calibration ==="
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

Write-Host "Encoder lr pilot completed."
