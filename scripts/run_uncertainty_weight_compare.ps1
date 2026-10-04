$ErrorActionPreference = "Stop"
$env:TQDM_DISABLE = "1"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Keep pretrained weights inside the project so the script works on Windows
# even when the global Torch cache is not writable.
$env:TORCH_HOME = Join-Path $Root ".torch_cache"
New-Item -ItemType Directory -Force -Path $env:TORCH_HOME | Out-Null

function Invoke-Python {
    & python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

# Compare the original uncertainty weighting with batch-mean-normalized
# weighting. All other settings and the teacher checkpoint are shared.
$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$TeacherRun = ".\runs\uncertainty_weight_teacher"
$TeacherCkpt = Join-Path $TeacherRun "teacher.pt"
$Epochs = 15

Invoke-Python train.py train_teacher --dataset cifar100 --data-root $DataRoot --download --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet34 --teacher-backbone resnet34 --pretrained --epochs $Epochs --alpha-unc 0.1 --alpha-proto 0.0 --alpha-pseudo 0.0 --work-dir $TeacherRun --teacher-ckpt $TeacherCkpt --device auto

foreach ($weightMode in @("raw", "mean_normalized")) {
    $studentRun = ".\runs\uncertainty_weight_${weightMode}"
    $studentCkpt = Join-Path $studentRun "student.pt"
    Invoke-Python train.py train_student --dataset cifar100 --data-root $DataRoot --download --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet18 --teacher-backbone resnet34 --student-backbone resnet18 --pretrained --epochs $Epochs --alpha-unc 0.0 --alpha-kd 1.0 --kd-mode uncertainty --uncertainty-weight-mode $weightMode --alpha-feat-kd 0.0 --alpha-supcon 0.0 --alpha-proto 0.0 --alpha-pseudo 0.0 --work-dir $studentRun --teacher-ckpt $TeacherCkpt --student-ckpt $studentCkpt --device auto

    $detectRun = ".\runs\uncertainty_weight_${weightMode}_detect"
    Invoke-Python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet18 --student-backbone resnet18 --pretrained --num-novel 40 --mc-samples 8 --score-mode normalized_entropy_mahalanobis --cluster-k oracle --work-dir $detectRun --student-ckpt $studentCkpt --device auto
}

Invoke-Python analyze_results.py --runs uncertainty_weight_raw_detect uncertainty_weight_mean_normalized_detect --root .\runs --out-dir .\analysis\uncertainty_weight_compare
Write-Host "uncertainty weight comparison finished."
