$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$BaseRun = ".\runs\toy_pseudo_train"
$DiscoverRun = ".\runs\toy_pseudo_discover"
$TeacherCkpt = Join-Path $BaseRun "teacher.pt"
$StudentCkpt = Join-Path $BaseRun "student.pt"
$SplitPath = ".\splits_toy_10_90.json"

python train.py train_teacher --dataset toy --data-root .\data --num-known 10 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 8 --num-workers 0 --epochs 2 --limit-train 128 --limit-val 64 --limit-test 256 --alpha-proto 0.1 --alpha-pseudo 0.1 --work-dir $BaseRun --teacher-ckpt $TeacherCkpt --device auto

python train.py train_student --dataset toy --data-root .\data --num-known 10 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 8 --num-workers 0 --epochs 2 --limit-train 128 --limit-val 64 --limit-test 256 --alpha-proto 0.1 --alpha-pseudo 0.1 --work-dir $BaseRun --teacher-ckpt $TeacherCkpt --student-ckpt $StudentCkpt --device auto

python train.py discover --dataset toy --data-root .\data --num-known 10 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 8 --num-workers 0 --limit-train 128 --limit-val 64 --limit-test 256 --num-novel 90 --mc-samples 4 --score-mode entropy_proto --work-dir $DiscoverRun --student-ckpt $StudentCkpt --device auto
