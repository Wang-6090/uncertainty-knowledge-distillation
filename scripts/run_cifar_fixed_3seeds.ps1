$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$DataRoot = ".\CIFAR-100-dataset-main"
$SplitPath = ".\splits_cifar100_60_40.json"
$Seeds = @(42, 43, 44)
$Epochs = 15

# Fixed ImageFolder CIFAR-100 60/40 protocol. The local dataset lives under
# CIFAR-100-dataset-main/{train,test}. Class-name splits are saved as
# splits_cifar100_60_40_imagefolder.json.
foreach ($seed in $Seeds) {
    $teacherRun = ".\runs\fixed_seed_${seed}_teacher"
    $teacherCkpt = Join-Path $teacherRun "teacher.pt"
    python train.py train_teacher --dataset imagefolder --data-root $DataRoot --num-known 60 --seed $seed --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet34 --teacher-backbone resnet34 --pretrained --epochs $Epochs --alpha-unc 0.1 --work-dir $teacherRun --teacher-ckpt $teacherCkpt --device auto

    foreach ($variant in @(@("ce", 0.0, "standard", "raw", "confidence"), @("standard_kd", 1.0, "standard", "raw", "confidence"), @("uncertainty_kd", 1.0, "uncertainty", "mean_normalized", "classification_error"))) {
        $name = $variant[0]
        $run = ".\runs\fixed_seed_${seed}_${name}"
        $student = Join-Path $run "student.pt"
        python train.py train_student --dataset imagefolder --data-root $DataRoot --num-known 60 --seed $seed --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet18 --teacher-backbone resnet34 --student-backbone resnet18 --pretrained --epochs $Epochs --alpha-kd $variant[1] --kd-mode $variant[2] --uncertainty-weight-mode $variant[3] --uncertainty-target-mode $variant[4] --alpha-unc 0.0 --alpha-feat-kd 0.0 --alpha-supcon 0.0 --alpha-proto 0.0 --alpha-pseudo 0.0 --work-dir $run --teacher-ckpt $teacherCkpt --student-ckpt $student --device auto
        $detectRun = ".\runs\fixed_seed_${seed}_${name}_detect"
        python train.py discover --dataset imagefolder --data-root $DataRoot --num-known 60 --seed $seed --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet18 --student-backbone resnet18 --pretrained --num-novel 40 --mc-samples 8 --score-mode normalized_entropy_mahalanobis --temperature-calibration --cluster-k auto --cluster-method kmeans --cluster-feature projection_pca --cluster-selection composite --cluster-normalize --work-dir $detectRun --student-ckpt $student --device auto
    }
}

$runNames = @()
foreach ($seed in $Seeds) {
    foreach ($name in @("ce", "standard_kd", "uncertainty_kd")) {
        $runNames += "fixed_seed_${seed}_${name}_detect"
    }
}
python analyze_results.py --runs $runNames --root .\runs --out-dir .\analysis\fixed_3seeds
