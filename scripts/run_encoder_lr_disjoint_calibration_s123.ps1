$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Strict seed-123 replication of the encoder-LR scale comparison. Unlike the
# older seed-123 runs, both arms share identical mixed-pool/nnPU settings and
# use a disjoint known-validation calibration subset.
$Teacher = ".\runs\semantic_isolated_full_s123_cifar_profile_teacher\teacher.pt"
$Split = ".\splits_cifar100_protocols\cifar100_60_40_semantic_isolated.json"
if (-not (Test-Path $Teacher)) { throw "Missing seed-123 teacher checkpoint: $Teacher" }

$Common = @(
    "--dataset", "cifar100", "--data-root", ".\data", "--num-known", "60",
    "--seed", "123", "--split-path", $Split, "--known-split-mode", "random",
    "--image-size", "64", "--batch-size", "64", "--num-workers", "0",
    "--device", "auto", "--calibration-ratio", "0.25",
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
$DiscoverCommon = @(
    "--dataset", "cifar100", "--data-root", ".\data", "--num-known", "60",
    "--seed", "123", "--split-path", $Split, "--known-split-mode", "random",
    "--image-size", "64", "--batch-size", "64", "--num-workers", "0",
    "--device", "auto", "--calibration-ratio", "0.25",
    "--backbone", "resnet18", "--student-backbone", "resnet18",
    "--teacher-backbone", "resnet34"
)

foreach ($Scale in @("1.0", "0.1")) {
    $Tag = if ($Scale -eq "1.0") { "lr10" } else { "lr01" }
    $StudentDir = ".\runs\semantic_isolated_full_s123_disjointcal_${Tag}_student"
    $Student = Join-Path $StudentDir "student.pt"
    if (-not (Test-Path $Student)) {
        & python train.py train_student @Common --encoder-lr-scale $Scale `
            --work-dir $StudentDir --teacher-ckpt $Teacher --student-ckpt $Student
        if ($LASTEXITCODE -ne 0) { throw "Student training failed for encoder LR scale $Scale" }
    } else {
        Write-Host "Reusing existing checkpoint: $Student"
    }

    $DiscoverDir = ".\runs\semantic_isolated_full_s123_disjointcal_${Tag}_discover"
    & python train.py discover @DiscoverCommon --num-novel 40 --mc-samples 4 `
        --score-mode normalized_entropy_min_class_knn --knn-ood `
        --threshold-policy known_coverage --target-known-coverage 0.95 `
        --skip-clustering --work-dir $DiscoverDir --student-ckpt $Student
    if ($LASTEXITCODE -ne 0) { throw "Discovery failed for encoder LR scale $Scale" }
}

Write-Host "Completed strict seed-123 disjoint-calibration encoder-LR comparison."
