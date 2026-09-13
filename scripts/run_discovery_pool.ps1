$ErrorActionPreference = "Stop"
$env:TQDM_DISABLE = "1"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Stage F of the README protocol: unlabeled discovery-pool consistency.
# Start with a frozen K and two-view cosine consistency. Pseudo labels are
# optional and can be enabled by raising --alpha-cluster.
$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$TeacherRun = ".\runs\protocol_teacher"
$TeacherCkpt = Join-Path $TeacherRun "teacher.pt"
$StudentRun = ".\runs\protocol_C_uncertainty_kd_s42"
$StudentCkpt = Join-Path $StudentRun "student.pt"
$DiscoveryRun = ".\runs\protocol_F_discovery"
$DiscoveryCkpt = Join-Path $DiscoveryRun "student_discovery.pt"

if (-not (Test-Path $StudentCkpt)) {
    throw "Expected student checkpoint $StudentCkpt. Run scripts/run_protocol_ablation.ps1 first or train a student."
}

python train.py train_discovery --dataset cifar100 --data-root $DataRoot --download --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --epochs 8 --lr 1e-4 --alpha-unc 0.0 --alpha-kd 1.0 --kd-mode uncertainty --alpha-feat-kd 0.0 --alpha-supcon 0.1 --alpha-proto 0.0 --alpha-consistency 0.5 --alpha-contrast 0.1 --alpha-cluster 0.0 --discovery-cluster-k fixed8 --include-discovery-pool --limit-discovery 4000 --work-dir $DiscoveryRun --teacher-ckpt $TeacherCkpt --student-ckpt $StudentCkpt --output-ckpt $DiscoveryCkpt --device auto

python train.py discover --dataset cifar100 --data-root $DataRoot --download --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --num-novel 40 --mc-samples 4 --score-mode entropy_mahalanobis --cluster-k both --temperature-scaling --compare-scores --cluster-confidence-percentile 50 --work-dir .\runs\protocol_F_discovery_detect --student-ckpt $DiscoveryCkpt --device auto

python analyze_results.py --runs protocol_F_discovery_detect --root .\runs --out-dir .\analysis\protocol_discovery
