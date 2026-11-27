$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# One-seed follow-up to the seed-42 OE pilot. Both students share one teacher,
# split, seed, training budget, base losses, detector, and known-only calibration.
# Only the OE student receives an unlabeled CIFAR-10 uniform-logit loss.
$Seed = 123
$Epochs = 5
$Tag = "protocol_oe_pair_s${Seed}"
$Split = ".\splits_cifar100_protocols\cifar100_60_40_random.json"
$TeacherDir = ".\runs\${Tag}_teacher"
$Teacher = Join-Path $TeacherDir "teacher.pt"

if (-not (Test-Path ".\data\cifar-100-python\train")) {
    throw "CIFAR-100 is not available under .\data."
}
if (-not (Test-Path ".\data\cifar-10-batches-py\data_batch_1")) {
    throw "CIFAR-10 is not available under .\data."
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
    "--seed", "$Seed",
    "--split-path", $Split,
    "--known-split-mode", "random",
    "--image-size", "64",
    "--batch-size", "64",
    "--num-workers", "0",
    "--device", "auto"
)

Write-Host "=== Train paired teacher: $Tag ==="
Invoke-CheckedPython (@("train.py", "train_teacher") + $Common + @(
    "--backbone", "resnet34",
    "--pretrained",
    "--epochs", "$Epochs",
    "--alpha-unc", "0.1",
    "--work-dir", $TeacherDir,
    "--teacher-ckpt", $Teacher
))

foreach ($Variant in @(
    @{ Name = "baseline"; OeArgs = @() },
    @{ Name = "oe_u005"; OeArgs = @(
        "--outlier-dataset", "cifar10",
        "--outlier-data-root", ".\data",
        "--alpha-outlier-uniform", "0.05"
    ) }
)) {
    $StudentDir = ".\runs\${Tag}_$($Variant.Name)_student"
    $Student = Join-Path $StudentDir "student.pt"
    $DetectDir = ".\runs\${Tag}_$($Variant.Name)_detect"

    Write-Host "=== Train student: $Tag $($Variant.Name) ==="
    Invoke-CheckedPython (@("train.py", "train_student") + $Common + @(
        "--backbone", "resnet18",
        "--teacher-backbone", "resnet34",
        "--pretrained",
        "--epochs", "$Epochs",
        "--alpha-unc", "0.1",
        "--alpha-kd", "1.0",
        "--kd-mode", "uncertainty",
        "--temperature", "2.0",
        "--uncertainty-weight-mode", "raw",
        "--alpha-feat-kd", "0.1",
        "--alpha-proto", "0.1",
        "--alpha-supcon", "0.1",
        "--work-dir", $StudentDir,
        "--teacher-ckpt", $Teacher,
        "--student-ckpt", $Student
    ) + $Variant.OeArgs)

    Write-Host "=== Detect: $Tag $($Variant.Name) ==="
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
}

Write-Host "Paired OE seed pilot completed: $Tag"
