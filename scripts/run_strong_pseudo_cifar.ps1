$ErrorActionPreference = "Continue"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$BaseRun = ".\runs\cifar_strong_pseudo_alpha_0_1"
$DiscoverRun = ".\runs\cifar_strong_pseudo_alpha_0_1_fixed_entropy_proto"
$TeacherCkpt = Join-Path $BaseRun "teacher.pt"
$StudentCkpt = Join-Path $BaseRun "student.pt"

New-Item -ItemType Directory -Force -Path $BaseRun, $DiscoverRun | Out-Null

if (-not (Test-Path $TeacherCkpt)) {
    python train.py train_teacher --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --pretrained --epochs 15 --alpha-proto 0.1 --alpha-pseudo 0.1 --pseudo-mode strong --pseudo-feature-noise 0.05 --work-dir $BaseRun --teacher-ckpt $TeacherCkpt --device auto
    if ($LASTEXITCODE -ne 0) { throw "strong pseudo teacher training failed" }
}

if (-not (Test-Path $StudentCkpt)) {
    python train.py train_student --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --pretrained --epochs 15 --alpha-proto 0.1 --alpha-pseudo 0.1 --pseudo-mode strong --pseudo-feature-noise 0.05 --work-dir $BaseRun --teacher-ckpt $TeacherCkpt --student-ckpt $StudentCkpt --device auto
    if ($LASTEXITCODE -ne 0) { throw "strong pseudo student training failed" }
}

if (-not (Test-Path (Join-Path $DiscoverRun "discovery_report.json"))) {
    python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --pretrained --num-novel 40 --mc-samples 8 --open-val-ratio 0.2 --score-mode entropy_proto --work-dir $DiscoverRun --student-ckpt $StudentCkpt --device auto
    if ($LASTEXITCODE -ne 0) { throw "strong pseudo discovery failed" }
}

Write-Host "strong pseudo experiment finished: $DiscoverRun"
