# Fixed-score re-evaluation for the completed pseudo-unknown ablation.
$ErrorActionPreference = "Continue"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$Alphas = @("0.0", "0.05", "0.1", "0.2")

foreach ($alpha in $Alphas) {
    $tag = $alpha.Replace('.', '_')
    $studentCkpt = ".\runs\cifar_pseudo_alpha_$tag\student.pt"
    $runDir = ".\runs\cifar_pseudo_alpha_${tag}_fixed_entropy_proto"
    $report = Join-Path $runDir "discovery_report.json"
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    if (Test-Path $report) {
        Write-Host "fixed-score result already exists for alpha_pseudo=$alpha; skipping"
        continue
    }

    Write-Host "===== fixed entropy_proto, alpha_pseudo=$alpha ====="
    & python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --pretrained --num-novel 40 --mc-samples 8 --open-val-ratio 0.2 --score-mode entropy_proto --work-dir $runDir --student-ckpt $studentCkpt --device auto
    if ($LASTEXITCODE -ne 0) { throw "fixed-score discovery failed for alpha_pseudo=$alpha" }
}

Write-Host "fixed entropy_proto re-evaluation finished."
