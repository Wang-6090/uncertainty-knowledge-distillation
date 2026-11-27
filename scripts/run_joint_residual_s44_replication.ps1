$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Second-seed replication of joint_mixed_residual against the seed-44 equal
# assignment arm from run_joint_prior_aware_pilot_s44.ps1. Only the residual
# switch differs; residual temperature=1 and floor=0 are held fixed.
$Teacher = ".\runs\semantic_isolated_full_s44_disjointcal_teacher\teacher.pt"
$Split = ".\splits_cifar100_protocols\cifar100_60_40_semantic_isolated.json"
if (-not (Test-Path $Teacher)) { throw "Missing seed-44 teacher checkpoint: $Teacher" }

$RunDir = ".\runs\joint_residual_s44_repro"
$Student = Join-Path $RunDir "student.pt"
& python train.py train_student `
    --dataset cifar100 --data-root .\data --num-known 60 --seed 44 `
    --split-path $Split --known-split-mode random --image-size 64 `
    --batch-size 64 --num-workers 0 --device auto --calibration-ratio 0.25 `
    --backbone resnet18 --student-backbone resnet18 --teacher-backbone resnet34 `
    --pretrained --epochs 3 --limit-train 1200 --limit-val 300 `
    --limit-test 1000 --limit-discovery 1200 --alpha-unc 0.1 --alpha-kd 1.0 `
    --kd-mode uncertainty --temperature 2.0 --uncertainty-weight-mode raw `
    --alpha-feat-kd 0.1 --alpha-proto 0.1 --alpha-supcon 0.1 `
    --alpha-discovery-uncertainty-pu 0.1 `
    --discovery-uncertainty-pu-risk nu_corrected `
    --discovery-uncertainty-known-prior 0.2 --discovery-pool `
    --discovery-pool-mode mixed --matched-discovery-pool `
    --matched-pool-size 1200 --matched-known-prior 0.2 `
    --encoder-lr-scale 0.1 --joint-discovery --joint-num-novel 40 `
    --joint-prototype-init kmeans --joint-prototype-warmup-epochs 1 `
    --joint-prototype-refresh-epochs 1 --joint-space unified `
    --alpha-joint-discovery 0.2 --alpha-joint-known-ce 0.1 `
    --joint-assignment-temperature 1.0 --joint-weighted-sinkhorn `
    --joint-mixed-residual --work-dir $RunDir `
    --teacher-ckpt $Teacher --student-ckpt $Student
if ($LASTEXITCODE -ne 0) { throw "student training failed" }

$DetectDir = ".\runs\joint_residual_s44_repro_detect"
& python train.py discover --dataset cifar100 --data-root .\data `
    --num-known 60 --num-novel 40 --seed 44 --split-path $Split `
    --known-split-mode random --image-size 64 --batch-size 64 --num-workers 0 `
    --device auto --limit-train 1200 --limit-val 300 --limit-test 1000 `
    --limit-discovery 1200 --calibration-ratio 0.25 `
    --score-mode normalized_entropy_min_class_knn --knn-ood --knn-k 10 `
    --knn-feature features --threshold-policy known_coverage `
    --target-known-coverage 0.95 --cluster-k oracle --cluster-method kmeans `
    --cluster-feature projection_pca --cluster-pca-dim 32 --cluster-normalize `
    --mc-samples 4 --student-ckpt $Student `
    --novel-head-ckpt (Join-Path $RunDir "novel_head.pt") --work-dir $DetectDir
if ($LASTEXITCODE -ne 0) { throw "detection failed" }

Write-Host "Completed seed-44 joint mixed-residual replication."
