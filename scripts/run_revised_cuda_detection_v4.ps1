$ErrorActionPreference = "Stop"
$env:TQDM_DISABLE = "1"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Invoke-Python {
    & python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

Invoke-Python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

$Seeds = @(42, 123, 3407)
$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$OutputRoot = ".\analysis\revised_cuda_multiseed_v4"

foreach ($seed in $Seeds) {
    foreach ($method in @(
        "A_ce",
        "B_standard_kd",
        "C_uncertainty_kd",
        "D_uncertainty_feature_kd",
        "E_full_representation"
    )) {
        $studentCkpt = ".\runs\revised_cuda_ms_s${seed}_${method}\student.pt"
        if (-not (Test-Path $studentCkpt)) {
            throw "Missing student checkpoint: $studentCkpt"
        }

        foreach ($clusterMode in @("oracle", "auto")) {
            $detectRun = ".\runs\revised_cuda_ms_s${seed}_${method}_v4_detect_${clusterMode}"
            $reportPath = Join-Path $detectRun "discovery_report.json"
            if (Test-Path $reportPath) {
                Write-Host "skip existing report: $reportPath"
                continue
            }

            Invoke-Python train.py discover --dataset cifar100 --data-root $DataRoot --download `
                --num-known 60 --seed $seed --split-path $SplitPath `
                --image-size 64 --batch-size 64 --num-workers 0 `
                --backbone resnet18 --student-backbone resnet18 --pretrained `
                --num-novel 40 --mc-samples 8 --score-mode normalized_entropy_mahalanobis `
                --cluster-k $clusterMode --cluster-feature projection_pca `
                --cluster-selection composite --cluster-normalize `
                --threshold-policy class_conditional --threshold-min-class-samples 5 `
                --candidate-purify open_score --candidate-keep-ratio 0.75 `
                --work-dir $detectRun --student-ckpt $studentCkpt --device cuda
        }
    }
}

Invoke-Python analyze_cuda_multiseed.py `
    --root .\runs `
    --glob "revised_cuda_ms_s*_v4_detect_*" `
    --out-dir $OutputRoot

Write-Host "CUDA detection v4 finished."
