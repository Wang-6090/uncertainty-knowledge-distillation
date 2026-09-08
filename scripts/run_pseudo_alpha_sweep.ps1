# Python/tqdm may write progress output to stderr. Keep it visible without
# letting PowerShell treat that normal output as a terminating script error.
$ErrorActionPreference = "Continue"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$Alphas = @("0.0", "0.05", "0.1", "0.2")

foreach ($alpha in $Alphas) {
    $tag = $alpha.Replace('.', '_')
    $baseRun = ".\runs\cifar_pseudo_alpha_$tag"
    $discoverRun = ".\runs\cifar_pseudo_alpha_${tag}_discover"
    $teacherCkpt = Join-Path $baseRun "teacher.pt"
    $studentCkpt = Join-Path $baseRun "student.pt"
    $logPath = Join-Path $discoverRun "sweep.log"

    New-Item -ItemType Directory -Force -Path $baseRun, $discoverRun | Out-Null
    "===== alpha_pseudo=$alpha =====" | Tee-Object -FilePath $logPath

    if (Test-Path $teacherCkpt) {
        Write-Host "teacher already exists for alpha_pseudo=$alpha; skipping"
    } else {
        & python train.py train_teacher --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --pretrained --epochs 15 --alpha-proto 0.1 --alpha-pseudo $alpha --work-dir $baseRun --teacher-ckpt $teacherCkpt --device auto 2>&1 | Tee-Object -FilePath $logPath -Append
        if ($LASTEXITCODE -ne 0) { throw "teacher failed for alpha_pseudo=$alpha" }
    }

    if (Test-Path $studentCkpt) {
        Write-Host "student already exists for alpha_pseudo=$alpha; skipping"
    } else {
        & python train.py train_student --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --pretrained --epochs 15 --alpha-proto 0.1 --alpha-pseudo $alpha --work-dir $baseRun --teacher-ckpt $teacherCkpt --student-ckpt $studentCkpt --device auto 2>&1 | Tee-Object -FilePath $logPath -Append
        if ($LASTEXITCODE -ne 0) { throw "student failed for alpha_pseudo=$alpha" }
    }

    if (Test-Path (Join-Path $discoverRun "discovery_report.json")) {
        Write-Host "discovery already exists for alpha_pseudo=$alpha; skipping"
    } else {
        & python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --pretrained --num-novel 40 --mc-samples 8 --open-val-ratio 0.2 --score-mode auto --auto-calibrate-score --work-dir $discoverRun --student-ckpt $studentCkpt --device auto 2>&1 | Tee-Object -FilePath $logPath -Append
        if ($LASTEXITCODE -ne 0) { throw "discover failed for alpha_pseudo=$alpha" }
    }
}

Write-Host "alpha_pseudo sweep finished. Results are in runs\\cifar_pseudo_alpha_*_discover."
