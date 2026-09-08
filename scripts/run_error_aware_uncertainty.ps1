$ErrorActionPreference = "Continue"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$BaseRun = ".\runs\cifar_error_aware_uncertainty"
$TeacherCkpt = Join-Path $BaseRun "teacher.pt"
$StudentCkpt = Join-Path $BaseRun "student.pt"

New-Item -ItemType Directory -Force -Path $BaseRun | Out-Null

if (-not (Test-Path $TeacherCkpt)) {
    python train.py train_teacher --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --pretrained --epochs 15 --alpha-proto 0.1 --alpha-pseudo 0.0 --uncertainty-target-mode classification_error --work-dir $BaseRun --teacher-ckpt $TeacherCkpt --device auto
    if ($LASTEXITCODE -ne 0) { throw "error-aware teacher training failed" }
}

if (-not (Test-Path $StudentCkpt)) {
    python train.py train_student --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --pretrained --epochs 15 --alpha-proto 0.1 --alpha-pseudo 0.0 --uncertainty-target-mode classification_error --work-dir $BaseRun --teacher-ckpt $TeacherCkpt --student-ckpt $StudentCkpt --device auto
    if ($LASTEXITCODE -ne 0) { throw "error-aware student training failed" }
}

$modes = @("full", "entropy_aleatoric", "entropy_proto")
foreach ($mode in $modes) {
    $runDir = ".\runs\cifar_error_aware_uncertainty_$mode"
    $report = Join-Path $runDir "discovery_report.json"
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    if (Test-Path $report) {
        Write-Host "discovery already exists for $mode; skipping"
        continue
    }
    python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --pretrained --num-novel 40 --mc-samples 8 --open-val-ratio 0.2 --score-mode $mode --work-dir $runDir --student-ckpt $StudentCkpt --device auto
    if ($LASTEXITCODE -ne 0) { throw "error-aware discovery failed for mode=$mode" }
}

Write-Host "error-aware uncertainty experiment finished."
