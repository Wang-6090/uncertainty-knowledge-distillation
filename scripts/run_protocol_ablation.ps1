$ErrorActionPreference = "Stop"
$env:TQDM_DISABLE = "1"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Fixed protocol for README section 4, problem F.
# Default is 3 seeds and the A/B/C comparison. Later rows can be enabled
# after C is shown to beat B.
$Seeds = @(42, 43, 44)
$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$TeacherRun = ".\runs\protocol_teacher"
$TeacherCkpt = Join-Path $TeacherRun "teacher.pt"

python train.py train_teacher --dataset cifar100 --data-root $DataRoot --download --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet18 --epochs 20 --alpha-unc 0.1 --alpha-proto 0.0 --alpha-pseudo 0.0 --work-dir $TeacherRun --teacher-ckpt $TeacherCkpt --device auto

$experiments = @(
    @{ Name = "A_ce"; KD = 0.0; Mode = "standard"; Feat = 0.0; Unc = 0.0; SupCon = 0.0; Proto = 0.0 },
    @{ Name = "B_standard_kd"; KD = 1.0; Mode = "standard"; Feat = 0.0; Unc = 0.0; SupCon = 0.0; Proto = 0.0 },
    @{ Name = "C_uncertainty_kd"; KD = 1.0; Mode = "uncertainty"; Feat = 0.0; Unc = 0.0; SupCon = 0.0; Proto = 0.0 }
)

$detectRuns = @()
foreach ($seed in $Seeds) {
    foreach ($exp in $experiments) {
        $run = ".\runs\protocol_$($exp.Name)_s$seed"
        $student = Join-Path $run "student.pt"
        python train.py train_student --dataset cifar100 --data-root $DataRoot --download --num-known 60 --seed $seed --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet18 --epochs 20 --alpha-unc $exp.Unc --alpha-kd $exp.KD --kd-mode $exp.Mode --alpha-feat-kd $exp.Feat --alpha-supcon $exp.SupCon --alpha-proto $exp.Proto --alpha-pseudo 0.0 --work-dir $run --teacher-ckpt $TeacherCkpt --student-ckpt $student --device auto --uncertainty-target-mode confidence

        $detectRunName = "protocol_$($exp.Name)_s$seed`_detect"
        $detectRun = ".\runs\$detectRunName"
        python train.py discover --dataset cifar100 --data-root $DataRoot --download --num-known 60 --seed $seed --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet18 --num-novel 40 --mc-samples 4 --score-mode entropy_mahalanobis --cluster-k auto --temperature-calibration --work-dir $detectRun --student-ckpt $student --device auto
        $detectRuns += $detectRunName
    }
}

python analyze_results.py --runs @detectRuns --root .\runs --out-dir .\analysis\protocol_ablation
