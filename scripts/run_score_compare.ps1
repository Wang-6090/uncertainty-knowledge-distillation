$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Common = "--dataset cifar100 --data-root .\data --num-known 60 --seed 42 --split-path .\splits_cifar100_60_40.json --image-size 64 --batch-size 32 --num-workers 0 --limit-train 5000 --limit-val 1000 --limit-test 2000 --num-novel 40 --mc-samples 4 --backbone resnet18 --student-ckpt .\runs\cifar_mid_train\student.pt --device auto"

Invoke-Expression "python train.py discover $Common --score-mode full --work-dir .\runs\cifar_score_full"
Invoke-Expression "python train.py discover $Common --score-mode entropy_proto --work-dir .\runs\cifar_score_entropy_proto_mid"
Invoke-Expression "python train.py discover $Common --score-mode max_softmax --work-dir .\runs\cifar_score_max_softmax"
Invoke-Expression "python train.py discover $Common --score-mode energy --work-dir .\runs\cifar_score_energy"

python analyze_results.py --runs cifar_score_full cifar_score_entropy_proto_mid cifar_score_max_softmax cifar_score_energy --root .\runs --out-dir .\analysis
