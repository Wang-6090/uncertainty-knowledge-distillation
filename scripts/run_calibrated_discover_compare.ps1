$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$StudentCkpt = ".\runs\cifar_proto_train\student.pt"

python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --limit-train 5000 --limit-val 1000 --limit-test 2000 --num-novel 40 --mc-samples 4 --open-val-ratio 0.2 --score-mode entropy_proto --work-dir .\runs\cifar_calib_entropy_proto --student-ckpt $StudentCkpt --device auto

python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --limit-train 5000 --limit-val 1000 --limit-test 2000 --num-novel 40 --mc-samples 4 --open-val-ratio 0.2 --score-mode auto --auto-calibrate-score --work-dir .\runs\cifar_calib_auto --student-ckpt $StudentCkpt --device auto

python analyze_results.py --runs cifar_calib_entropy_proto cifar_calib_auto --root .\runs --out-dir .\analysis\calibrated_discover
