$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Detector-only paired test. No student is retrained. Compare uncertainty
# correction weights 0.25 and 0.75 on each fixed checkpoint. The already-run
# weight=0.5 arm is in runs/nnpu_uncertainty_score_s*_..._discover.

foreach ($Seed in @(44, 123)) {
    foreach ($Arm in @("encoder_grad", "head_only")) {
        $Student = ".\runs\nnpu_encoder_gradient_s${Seed}_${Arm}_student\student.pt"
        if (-not (Test-Path $Student)) { throw "missing student checkpoint: $Student" }
        foreach ($Weight in @(0.25, 0.75)) {
            $WeightTag = $Weight.ToString("0.00", [Globalization.CultureInfo]::InvariantCulture).Replace(".", "")
            $Work = ".\runs\nnpu_uncertainty_score_w${WeightTag}_s${Seed}_${Arm}_discover"
            & python train.py discover `
                --dataset cifar100 --data-root .\data --num-known 60 --seed $Seed `
                --split-path .\splits_cifar100_protocols\cifar100_60_40_semantic_isolated.json `
                --known-split-mode random --transform-profile cifar --image-size 64 `
                --batch-size 64 --num-workers 0 --device auto --calibration-ratio 0.25 `
                --backbone resnet18 --student-backbone resnet18 --pretrained `
                --num-novel 40 --mc-samples 4 --knn-ood `
                --score-mode normalized_entropy_uncertainty_min_class_knn `
                --uncertainty-score-weight $Weight `
                --threshold-policy known_coverage --target-known-coverage 0.95 `
                --skip-clustering --work-dir $Work --student-ckpt $Student
            if ($LASTEXITCODE -ne 0) { throw "uncertainty-score discovery failed for seed $Seed arm $Arm weight $Weight" }
        }
    }
}

Write-Host "Completed detector-only uncertainty-score weight comparison."
