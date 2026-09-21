$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$BaseRun = ".\runs\cifar_pseudo_train"
$DiscoverRun = ".\runs\cifar_pseudo_discover"
$TeacherCkpt = Join-Path $BaseRun "teacher.pt"
$StudentCkpt = Join-Path $BaseRun "student.pt"

# Formal CIFAR-100 run: pretrained ResNet18, full available data, prototype and pseudo-unknown losses.
python train.py train_teacher --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --pretrained --epochs 15 --alpha-proto 0.1 --alpha-pseudo 0.1 --work-dir $BaseRun --teacher-ckpt $TeacherCkpt --device auto

python train.py train_student --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --pretrained --epochs 15 --alpha-proto 0.1 --alpha-pseudo 0.1 --work-dir $BaseRun --teacher-ckpt $TeacherCkpt --student-ckpt $StudentCkpt --device auto

python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --pretrained --num-novel 40 --mc-samples 8 --open-val-ratio 0.2 --score-mode auto --auto-calibrate-score --work-dir $DiscoverRun --student-ckpt $StudentCkpt --device auto
