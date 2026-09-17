$ErrorActionPreference = "Stop"
$env:TQDM_DISABLE = "1"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Stage F of the README protocol: unlabeled discovery-pool consistency.
# Use the current two-view discovery objective together with the
# unknown-promoting loss on an unknown-only discovery pool.
$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$TeacherRun = ".\runs\protocol_teacher"
$TeacherCkpt = Join-Path $TeacherRun "teacher.pt"
$DiscoveryRun = ".\runs\protocol_F_discovery"
$DiscoveryCkpt = Join-Path $DiscoveryRun "student_discovery.pt"

python train.py train_student --dataset cifar100 --data-root $DataRoot --download --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --epochs 8 --lr 1e-4 --alpha-unc 0.0 --alpha-kd 1.0 --kd-mode uncertainty --alpha-feat-kd 0.0 --alpha-supcon 0.1 --alpha-proto 0.0 --discovery-pool --alpha-discovery 0.5 --alpha-discovery-unknown 0.1 --discovery-loss nt_xent --limit-discovery 4000 --work-dir $DiscoveryRun --teacher-ckpt $TeacherCkpt --student-ckpt $DiscoveryCkpt --device auto

python train.py discover --dataset cifar100 --data-root $DataRoot --download --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --num-novel 40 --mc-samples 4 --score-mode entropy_mahalanobis --cluster-k auto --temperature-calibration --work-dir .\runs\protocol_F_discovery_detect --student-ckpt $DiscoveryCkpt --device auto

python analyze_results.py --runs protocol_F_discovery_detect --root .\runs --out-dir .\analysis\protocol_discovery
