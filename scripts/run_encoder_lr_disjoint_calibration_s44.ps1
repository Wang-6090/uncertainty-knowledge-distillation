$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Reproduce the strict seed-44 encoder learning-rate comparison. The teacher
# must already exist under runs/semantic_isolated_full_s44_disjointcal_teacher.
$Common = @(
    "--dataset", "cifar100", "--data-root", ".\data", "--num-known", "60",
    "--seed", "44", "--split-path", ".\splits_cifar100_protocols\cifar100_60_40_semantic_isolated.json",
    "--known-split-mode", "random", "--image-size", "64", "--batch-size", "64",
    "--num-workers", "0", "--device", "auto", "--calibration-ratio", "0.25",
    "--backbone", "resnet18", "--student-backbone", "resnet18",
    "--teacher-backbone", "resnet34", "--pretrained", "--epochs", "5",
    "--alpha-unc", "0.1", "--alpha-kd", "1.0", "--kd-mode", "uncertainty",
    "--temperature", "2.0", "--uncertainty-weight-mode", "raw",
    "--alpha-feat-kd", "0.1", "--alpha-proto", "0.1", "--alpha-supcon", "0.1",
    "--alpha-discovery-uncertainty-pu", "0.1",
    "--discovery-uncertainty-pu-risk", "nu_corrected",
    "--discovery-uncertainty-known-prior", "0.2", "--discovery-pool",
    "--discovery-pool-mode", "mixed", "--matched-discovery-pool",
    "--matched-pool-size", "5400", "--matched-known-prior", "0.2"
)
$Teacher = ".\runs\semantic_isolated_full_s44_disjointcal_teacher\teacher.pt"

foreach ($Scale in @("1.0", "0.1")) {
    $Tag = $Scale.Replace('.', '')
    $StudentDir = ".\runs\semantic_isolated_full_s44_disjointcal_lr${Tag}_student"
    $Student = Join-Path $StudentDir "student.pt"
    & python train.py train_student @Common --encoder-lr-scale $Scale `
        --work-dir $StudentDir --teacher-ckpt $Teacher --student-ckpt $Student
    if ($LASTEXITCODE -ne 0) { throw "student training failed for scale $Scale" }

    $DiscoverDir = ".\runs\semantic_isolated_full_s44_disjointcal_lr${Tag}_discover"
    & python train.py discover @Common --num-novel 40 --mc-samples 4 `
        --score-mode normalized_entropy_min_class_knn --knn-ood `
        --threshold-policy known_coverage --target-known-coverage 0.95 `
        --skip-clustering --work-dir $DiscoverDir --student-ckpt $Student
    if ($LASTEXITCODE -ne 0) { throw "discovery failed for scale $Scale" }
}

Write-Host "Completed seed-44 disjoint-calibration encoder-LR comparison."
