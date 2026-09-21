$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$TeacherRun = ".\runs\revised_teacher_resnet34"
$TeacherCkpt = Join-Path $TeacherRun "teacher.pt"

# Only alpha-energy changes between the two student runs.
$experiments = @(
    @{ Name = "base"; AlphaEnergy = 0.0 },
    @{ Name = "energy"; AlphaEnergy = 0.1 }
)

foreach ($exp in $experiments) {
    $run = ".\runs\energy_compare_$($exp.Name)"
    $student = Join-Path $run "student.pt"
    $detectRun = ".\runs\energy_compare_$($exp.Name)_detect"

    python train.py train_student --dataset cifar100 --data-root $DataRoot `
      --num-known 60 --seed 42 --split-path $SplitPath `
      --backbone resnet18 --teacher-backbone resnet34 --student-backbone resnet18 `
      --pretrained --image-size 64 --batch-size 64 --num-workers 0 --epochs 15 `
      --alpha-energy $exp.AlphaEnergy --energy-margin 1.0 --energy-temperature 1.0 `
      --work-dir $run --teacher-ckpt $TeacherCkpt --student-ckpt $student --device auto

    python train.py discover --dataset cifar100 --data-root $DataRoot `
      --num-known 60 --seed 42 --split-path $SplitPath `
      --backbone resnet18 --student-backbone resnet18 --pretrained `
      --image-size 64 --batch-size 64 --num-workers 0 --num-novel 40 --mc-samples 8 `
      --score-mode normalized_entropy_mahalanobis --cluster-k auto `
      --cluster-method kmeans --cluster-feature projection `
      --work-dir $detectRun --student-ckpt $student --device auto
}

python analyze_results.py --runs energy_compare_base_detect energy_compare_energy_detect `
  --root .\runs --out-dir .\analysis\energy_compare
