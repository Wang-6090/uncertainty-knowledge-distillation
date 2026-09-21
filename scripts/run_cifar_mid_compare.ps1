$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$BaseRun = ".\runs\cifar_mid_train"
$FullRun = ".\runs\cifar_mid_full"
$EntropyRun = ".\runs\cifar_mid_entropy_proto"
$TeacherCkpt = Join-Path $BaseRun "teacher.pt"
$StudentCkpt = Join-Path $BaseRun "student.pt"

python train.py train_teacher --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --epochs 15 --limit-train 5000 --limit-val 1000 --limit-test 2000 --work-dir $BaseRun --teacher-ckpt $TeacherCkpt --device auto

python train.py train_student --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --epochs 15 --limit-train 5000 --limit-val 1000 --limit-test 2000 --work-dir $BaseRun --teacher-ckpt $TeacherCkpt --student-ckpt $StudentCkpt --device auto

python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --limit-train 5000 --limit-val 1000 --limit-test 2000 --num-novel 40 --mc-samples 4 --score-mode full --work-dir $FullRun --student-ckpt $StudentCkpt --device auto

python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --limit-train 5000 --limit-val 1000 --limit-test 2000 --num-novel 40 --mc-samples 4 --score-mode entropy_proto --work-dir $EntropyRun --student-ckpt $StudentCkpt --device auto

python analyze_results.py --runs cifar_mid_full cifar_mid_entropy_proto --root .\runs --out-dir .\analysis
