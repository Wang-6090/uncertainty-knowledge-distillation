$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Question: does the mixed-pool uncertainty nnPU objective help the detector
# representation by sending gradients into the shared encoder, or is its
# benefit limited to fitting the uncertainty head?
#
# Paired protocol: seed 44, CIFAR-100 semantic-isolated 60/40, full data,
# same teacher, five epochs, batch size 64, matched 5,400-sample mixed pool
# with known prior 0.2, same optimizer/losses/checkpoint selection/detector,
# and the same disjoint validation selection/calibration protocol. The only
# treatment factor is --discovery-uncertainty-pu-detach-features.
# The detached arm is the head-only control; the default arm lets nnPU update
# the encoder through the uncertainty head. Test labels are evaluation only.

$Common = @(
    "--dataset", "cifar100", "--data-root", ".\data", "--num-known", "60",
    "--seed", "44", "--split-path", ".\splits_cifar100_protocols\cifar100_60_40_semantic_isolated.json",
    "--known-split-mode", "random", "--image-size", "64", "--batch-size", "64",
    "--num-workers", "0", "--device", "auto", "--calibration-ratio", "0.25",
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
    "--matched-pool-size", "5400", "--matched-known-prior", "0.2"
)
$Teacher = ".\runs\semantic_isolated_full_s44_disjointcal_teacher\teacher.pt"
$Discover = @(
    "--dataset", "cifar100", "--data-root", ".\data", "--num-known", "60",
    "--seed", "44", "--split-path", ".\splits_cifar100_protocols\cifar100_60_40_semantic_isolated.json",
    "--known-split-mode", "random", "--image-size", "64", "--batch-size", "64",
    "--num-workers", "0", "--device", "auto", "--calibration-ratio", "0.25",
    "--backbone", "resnet18", "--student-backbone", "resnet18",
    "--pretrained", "--num-novel", "40", "--mc-samples", "4",
    "--score-mode", "normalized_entropy_min_class_knn", "--knn-ood",
    "--threshold-policy", "known_coverage", "--target-known-coverage", "0.95",
    "--skip-clustering"
)

foreach ($Arm in @("encoder_grad", "head_only")) {
    $StudentDir = ".\runs\nnpu_encoder_gradient_s44_${Arm}_student"
    $Student = Join-Path $StudentDir "student.pt"
    $ArmArgs = @()
    if ($Arm -eq "head_only") {
        $ArmArgs += "--discovery-uncertainty-pu-detach-features"
    }
    if (-not (Test-Path $Student)) {
        & python train.py train_student @Common @ArmArgs --work-dir $StudentDir `
            --teacher-ckpt $Teacher --student-ckpt $Student
        if ($LASTEXITCODE -ne 0) { throw "student training failed for $Arm" }
    }

    $DiscoverDir = ".\runs\nnpu_encoder_gradient_s44_${Arm}_discover"
    & python train.py discover @Discover --work-dir $DiscoverDir --student-ckpt $Student
    if ($LASTEXITCODE -ne 0) { throw "discovery failed for $Arm" }
}

Write-Host "Completed seed-44 nnPU encoder-gradient isolation comparison."
