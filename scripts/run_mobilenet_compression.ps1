$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:TORCH_HOME = Join-Path $Root ".torch_cache"
New-Item -ItemType Directory -Force -Path $env:TORCH_HOME | Out-Null

$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$TeacherRun = ".\runs\revised_teacher_resnet34"
$TeacherCkpt = Join-Path $TeacherRun "teacher.pt"
$Run = ".\runs\revised_mobilenet_uncertainty_kd"
$StudentCkpt = Join-Path $Run "student.pt"
$DetectRun = ".\runs\revised_mobilenet_uncertainty_kd_detect"

python train.py train_student --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet18 --teacher-backbone resnet34 --student-backbone mobilenet_v3_small --pretrained --epochs 15 --alpha-unc 0.0 --alpha-kd 1.0 --kd-mode uncertainty --alpha-feat-kd 0.1 --alpha-supcon 0.0 --alpha-proto 0.0 --alpha-pseudo 0.0 --work-dir $Run --teacher-ckpt $TeacherCkpt --student-ckpt $StudentCkpt --device auto

python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone mobilenet_v3_small --student-backbone mobilenet_v3_small --pretrained --num-novel 40 --mc-samples 8 --cluster-k oracle --score-mode normalized_entropy_mahalanobis --work-dir $DetectRun --student-ckpt $StudentCkpt --device auto
