$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Keep downloaded pretrained weights inside the project workspace. This avoids
# permission problems with the user's global torch cache on Windows.
$env:TORCH_HOME = Join-Path $Root ".torch_cache"
New-Item -ItemType Directory -Force -Path $env:TORCH_HOME | Out-Null

$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$TeacherRun = ".\runs\revised_teacher_resnet34"
$TeacherCkpt = Join-Path $TeacherRun "teacher.pt"

# One fixed teacher is shared by all student ablations.
python train.py train_teacher --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet34 --teacher-backbone resnet34 --pretrained --epochs 15 --alpha-unc 0.1 --alpha-proto 0.0 --alpha-pseudo 0.0 --work-dir $TeacherRun --teacher-ckpt $TeacherCkpt --device auto

$experiments = @(
    @{ Name = "A_ce"; KD = 0.0; UKD = "standard"; Feat = 0.0; Unc = 0.0; SupCon = 0.0; Proto = 0.0 },
    @{ Name = "B_standard_kd"; KD = 1.0; UKD = "standard"; Feat = 0.0; Unc = 0.0; SupCon = 0.0; Proto = 0.0 },
    @{ Name = "C_uncertainty_kd"; KD = 1.0; UKD = "uncertainty"; Feat = 0.0; Unc = 0.0; SupCon = 0.0; Proto = 0.0 },
    @{ Name = "D_uncertainty_feature_kd"; KD = 1.0; UKD = "uncertainty"; Feat = 0.1; Unc = 0.0; SupCon = 0.0; Proto = 0.0 },
    @{ Name = "E_full_representation"; KD = 1.0; UKD = "uncertainty"; Feat = 0.1; Unc = 0.1; SupCon = 0.1; Proto = 0.1 }
)

foreach ($exp in $experiments) {
    $run = ".\runs\revised_$($exp.Name)"
    $student = Join-Path $run "student.pt"
    python train.py train_student --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet18 --teacher-backbone resnet34 --student-backbone resnet18 --pretrained --epochs 15 --alpha-unc $exp.Unc --alpha-kd $exp.KD --kd-mode $exp.UKD --alpha-feat-kd $exp.Feat --alpha-supcon $exp.SupCon --alpha-proto $exp.Proto --alpha-pseudo 0.0 --work-dir $run --teacher-ckpt $TeacherCkpt --student-ckpt $student --device auto

    $detectRun = ".\runs\revised_$($exp.Name)_detect"
    python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet18 --student-backbone resnet18 --pretrained --limit-train 0 --limit-val 0 --limit-test 0 --num-novel 40 --mc-samples 8 --score-mode normalized_entropy_mahalanobis --work-dir $detectRun --student-ckpt $student --device auto
}

python analyze_results.py --runs revised_A_ce_detect revised_B_standard_kd_detect revised_C_uncertainty_kd_detect revised_D_uncertainty_feature_kd_detect revised_E_full_representation_detect --root .\runs --out-dir .\analysis\revised_ablation
