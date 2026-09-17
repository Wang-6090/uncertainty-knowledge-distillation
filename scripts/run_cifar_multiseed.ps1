$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:TORCH_HOME = Join-Path $Root ".torch_cache"
New-Item -ItemType Directory -Force -Path $env:TORCH_HOME | Out-Null

$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$Seeds = @(42, 43, 44)

foreach ($seed in $Seeds) {
    $teacherRun = ".\runs\cifar100_seed${seed}_teacher"
    $teacher = Join-Path $teacherRun "teacher.pt"
    python train.py train_teacher --dataset cifar100 --data-root $DataRoot --num-known 60 --seed $seed --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet34 --teacher-backbone resnet34 --pretrained --epochs 15 --alpha-unc 0.1 --work-dir $teacherRun --teacher-ckpt $teacher --device auto

    $studentRun = ".\runs\cifar100_seed${seed}_student"
    $student = Join-Path $studentRun "student.pt"
    python train.py train_student --dataset cifar100 --data-root $DataRoot --num-known 60 --seed $seed --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet18 --teacher-backbone resnet34 --student-backbone resnet18 --pretrained --epochs 15 --alpha-kd 1.0 --kd-mode uncertainty --uncertainty-weight-mode mean_normalized --work-dir $studentRun --teacher-ckpt $teacher --student-ckpt $student --device auto

    $detectRun = ".\runs\cifar100_seed${seed}_detect"
    python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed $seed --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet18 --student-backbone resnet18 --pretrained --num-novel 40 --mc-samples 8 --score-mode normalized_entropy_mahalanobis --temperature-calibration --cluster-k auto --cluster-method kmeans --cluster-feature projection_pca --cluster-selection composite --cluster-normalize --work-dir $detectRun --student-ckpt $student --device auto
}

python aggregate_multiseed.py --root .\runs --runs cifar100_seed42_detect cifar100_seed43_detect cifar100_seed44_detect --out .\analysis\cifar100_multiseed.json
