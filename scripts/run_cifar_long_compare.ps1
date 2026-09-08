$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$BaseRun = ".\runs\cifar_long_train"
$FullRun = ".\runs\cifar_long_full"
$EntropyRun = ".\runs\cifar_long_entropy_proto"
$TeacherCkpt = Join-Path $BaseRun "teacher.pt"
$StudentCkpt = Join-Path $BaseRun "student.pt"

python train.py train_teacher --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --epochs 20 --limit-train 1200 --limit-val 300 --limit-test 1000 --work-dir $BaseRun --teacher-ckpt $TeacherCkpt --device auto

python train.py train_student --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --epochs 20 --limit-train 1200 --limit-val 300 --limit-test 1000 --work-dir $BaseRun --teacher-ckpt $TeacherCkpt --student-ckpt $StudentCkpt --device auto

python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --limit-train 1200 --limit-val 300 --limit-test 1000 --num-novel 40 --mc-samples 4 --score-mode full --work-dir $FullRun --student-ckpt $StudentCkpt --device auto

python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --limit-train 1200 --limit-val 300 --limit-test 1000 --num-novel 40 --mc-samples 4 --score-mode entropy_proto --work-dir $EntropyRun --student-ckpt $StudentCkpt --device auto

python analyze_results.py --runs cifar_long_full cifar_long_entropy_proto --root .\runs --out-dir .\analysis
