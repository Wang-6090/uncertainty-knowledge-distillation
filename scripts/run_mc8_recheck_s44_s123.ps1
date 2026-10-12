$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Detector-only stability audit. Student checkpoints and all protocol settings
# are fixed; compare the existing MC=4 reports with MC=8. No model is trained.
foreach ($Seed in @(44, 123)) {
    foreach ($Arm in @("encoder_grad", "head_only")) {
        $Student = ".\runs\nnpu_encoder_gradient_s${Seed}_${Arm}_student\student.pt"
        if (-not (Test-Path $Student)) { throw "missing student checkpoint: $Student" }
        $Work = ".\runs\nnpu_mc8_s${Seed}_${Arm}_discover"
        & python train.py discover `
            --dataset cifar100 --data-root .\data --num-known 60 --seed $Seed `
            --split-path .\splits_cifar100_protocols\cifar100_60_40_semantic_isolated.json `
            --known-split-mode random --transform-profile cifar --image-size 64 `
            --batch-size 64 --num-workers 0 --device auto --calibration-ratio 0.25 `
            --backbone resnet18 --student-backbone resnet18 --pretrained `
            --num-novel 40 --mc-samples 8 --knn-ood `
            --score-mode normalized_entropy_uncertainty_min_class_knn `
            --uncertainty-score-weight 0.5 `
            --threshold-policy known_coverage --target-known-coverage 0.95 `
            --skip-clustering --work-dir $Work --student-ckpt $Student
        if ($LASTEXITCODE -ne 0) { throw "MC=8 discovery failed for seed $Seed arm $Arm" }
    }
}

Write-Host "Completed MC=8 detector stability recheck."
