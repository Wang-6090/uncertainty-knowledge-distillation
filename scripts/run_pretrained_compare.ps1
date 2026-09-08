$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$BaseRun = ".\runs\cifar_pretrained_train"
$FullRun = ".\runs\cifar_pretrained_full"
$EntropyRun = ".\runs\cifar_pretrained_entropy_proto"
$TeacherCkpt = Join-Path $BaseRun "teacher.pt"
$StudentCkpt = Join-Path $BaseRun "student.pt"

python train.py train_teacher --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --pretrained --epochs 15 --limit-train 5000 --limit-val 1000 --limit-test 2000 --alpha-proto 0.1 --work-dir $BaseRun --teacher-ckpt $TeacherCkpt --device auto

python train.py train_student --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --pretrained --epochs 15 --limit-train 5000 --limit-val 1000 --limit-test 2000 --alpha-proto 0.1 --work-dir $BaseRun --teacher-ckpt $TeacherCkpt --student-ckpt $StudentCkpt --device auto

python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --pretrained --limit-train 5000 --limit-val 1000 --limit-test 2000 --num-novel 40 --mc-samples 4 --score-mode entropy_proto --work-dir $EntropyRun --student-ckpt $StudentCkpt --device auto

python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --pretrained --limit-train 5000 --limit-val 1000 --limit-test 2000 --num-novel 40 --mc-samples 4 --score-mode auto --auto-calibrate-score --open-val-ratio 0.2 --work-dir $FullRun --student-ckpt $StudentCkpt --device auto

python analyze_results.py --runs cifar_pretrained_entropy_proto cifar_pretrained_full --root .\runs --out-dir .\analysis\pretrained_compare
