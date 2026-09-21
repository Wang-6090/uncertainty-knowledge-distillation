$ErrorActionPreference = "Stop"
$env:TQDM_DISABLE = "1"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:TORCH_HOME = Join-Path $Root ".torch_cache"
New-Item -ItemType Directory -Force -Path $env:TORCH_HOME | Out-Null

function Invoke-Python {
    & python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

$Seeds = @(42, 123)
$Epochs = 15
$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"

foreach ($seed in $Seeds) {
    $teacherRun = ".\runs\revised_ms_s$($seed)_teacher"
    $teacherCkpt = Join-Path $teacherRun "teacher.pt"
    if (-not (Test-Path $teacherCkpt)) {
        Invoke-Python train.py train_teacher --dataset cifar100 --data-root $DataRoot --download --num-known 60 --seed $seed --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet34 --teacher-backbone resnet34 --pretrained --epochs $Epochs --alpha-unc 0.1 --alpha-proto 0.0 --alpha-pseudo 0.0 --work-dir $teacherRun --teacher-ckpt $teacherCkpt --device auto
    } else {
        Write-Host "skip existing teacher: $teacherCkpt"
    }

    $studentRun = ".\runs\revised_ms_s$($seed)_F_discovery_pool"
    $studentCkpt = Join-Path $studentRun "student.pt"
    if (-not (Test-Path $studentCkpt)) {
        Invoke-Python train.py train_student --dataset cifar100 --data-root $DataRoot --download --num-known 60 --seed $seed --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet18 --teacher-backbone resnet34 --student-backbone resnet18 --pretrained --epochs $Epochs --alpha-unc 0.1 --alpha-kd 1.0 --kd-mode uncertainty --alpha-feat-kd 0.1 --alpha-supcon 0.1 --alpha-proto 0.1 --alpha-pseudo 0.0 --limit-discovery 0 --discovery-pool --discovery-pool-mode unknown --alpha-discovery 0.05 --discovery-loss nt_xent --work-dir $studentRun --teacher-ckpt $teacherCkpt --student-ckpt $studentCkpt --device auto
    } else {
        Write-Host "skip existing discovery student: $studentCkpt"
    }

    foreach ($clusterMode in @("oracle", "auto")) {
        $detectRun = ".\runs\revised_ms_s$($seed)_F_discovery_pool_detect_$clusterMode"
        $reportPath = Join-Path $detectRun "discovery_report.json"
        if (-not (Test-Path $reportPath)) {
            Invoke-Python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed $seed --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet18 --student-backbone resnet18 --pretrained --num-novel 40 --mc-samples 8 --score-mode normalized_entropy_mahalanobis --cluster-k $clusterMode --work-dir $detectRun --student-ckpt $studentCkpt --device auto
        } else {
            Write-Host "skip existing report: $reportPath"
        }
    }
}

Invoke-Python analyze_multiseed.py --root ".\runs" --glob "revised_ms_s*_detect_*" --out-dir ".\analysis\revised_multiseed"
Write-Host "discovery-pool multi-seed comparison finished."
