$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Matched seed-123 replication of the seed-44 one-factor PU-energy pilot.
$Teacher = ".\runs\semantic_isolated_full_s123_cifar_profile_teacher\teacher.pt"
$Split = ".\splits_cifar100_protocols\cifar100_60_40_semantic_isolated.json"
$StudentDir = ".\runs\semantic_isolated_full_s123_disjointcal_lr01_energypu_student"
$Student = Join-Path $StudentDir "student.pt"

$Common = @(
    "--dataset", "cifar100", "--data-root", ".\data", "--num-known", "60",
    "--seed", "123", "--split-path", $Split, "--known-split-mode", "random",
    "--image-size", "64", "--batch-size", "64", "--num-workers", "0",
    "--device", "auto", "--calibration-ratio", "0.25",
    "--backbone", "resnet18", "--student-backbone", "resnet18",
    "--teacher-backbone", "resnet34", "--pretrained", "--epochs", "5",
    "--encoder-lr-scale", "0.1", "--alpha-unc", "0.1", "--alpha-kd", "1.0",
    "--kd-mode", "uncertainty", "--temperature", "2.0",
    "--uncertainty-weight-mode", "raw", "--alpha-feat-kd", "0.1",
    "--alpha-proto", "0.1", "--alpha-supcon", "0.1",
    "--alpha-discovery-uncertainty-pu", "0.1",
    "--discovery-uncertainty-pu-risk", "nu_corrected",
    "--discovery-uncertainty-known-prior", "0.2", "--discovery-pool",
    "--discovery-pool-mode", "mixed", "--matched-discovery-pool",
    "--matched-pool-size", "5400", "--matched-known-prior", "0.2",
    "--alpha-discovery-energy-pu", "0.1",
    "--discovery-energy-pu-known-prior", "0.2",
    "--discovery-energy-pu-risk", "hard_nnpu", "--energy-margin", "1.0"
)

& python train.py train_student @Common --work-dir $StudentDir `
    --teacher-ckpt $Teacher --student-ckpt $Student
if ($LASTEXITCODE -ne 0) { throw "seed-123 energy-PU student training failed" }

$DiscoverDir = ".\runs\semantic_isolated_full_s123_disjointcal_lr01_energypu_discover"
& python train.py discover `
    --dataset cifar100 --data-root .\data --num-known 60 --seed 123 `
    --split-path $Split --known-split-mode random --image-size 64 `
    --batch-size 64 --num-workers 0 --device auto --calibration-ratio 0.25 `
    --backbone resnet18 --student-backbone resnet18 --pretrained `
    --num-novel 40 --mc-samples 4 --knn-ood `
    --score-mode normalized_entropy_min_class_knn `
    --threshold-policy known_coverage --target-known-coverage 0.95 `
    --skip-clustering --work-dir $DiscoverDir --student-ckpt $Student
if ($LASTEXITCODE -ne 0) { throw "seed-123 energy-PU discovery failed" }

Write-Host "Completed seed-123 one-factor PU-energy replication."
