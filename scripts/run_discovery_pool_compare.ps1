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

$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$TeacherCkpt = ".\runs\revised_ms_s42_teacher\teacher.pt"
$CommonTrain = @(
    "train.py", "train_student",
    "--dataset", "cifar100",
    "--data-root", $DataRoot,
    "--num-known", "60",
    "--seed", "42",
    "--split-path", $SplitPath,
    "--image-size", "64",
    "--batch-size", "64",
    "--num-workers", "0",
    "--backbone", "resnet18",
    "--teacher-backbone", "resnet34",
    "--student-backbone", "resnet18",
    "--pretrained",
    "--epochs", "2",
    "--limit-train", "1200",
    "--limit-val", "300",
    "--limit-test", "1000",
    "--alpha-unc", "0.1",
    "--alpha-kd", "1.0",
    "--kd-mode", "uncertainty",
    "--alpha-feat-kd", "0.1",
    "--alpha-supcon", "0.1",
    "--alpha-proto", "0.1",
    "--alpha-pseudo", "0.0",
    "--teacher-ckpt", $TeacherCkpt,
    "--device", "auto"
)
$CommonDetect = @(
    "train.py", "discover",
    "--dataset", "cifar100",
    "--data-root", $DataRoot,
    "--num-known", "60",
    "--seed", "42",
    "--split-path", $SplitPath,
    "--image-size", "64",
    "--batch-size", "64",
    "--num-workers", "0",
    "--backbone", "resnet18",
    "--student-backbone", "resnet18",
    "--pretrained",
    "--limit-train", "1200",
    "--limit-val", "300",
    "--limit-test", "1000",
    "--num-novel", "40",
    "--mc-samples", "4",
    "--score-mode", "normalized_entropy_mahalanobis",
    "--cluster-k", "oracle",
    "--device", "auto"
)

$Runs = @(
    @{ Name = "discovery_compare_base"; Extra = @() },
    @{ Name = "discovery_compare_ntxent"; Extra = @("--limit-discovery", "1200", "--discovery-pool", "--discovery-pool-mode", "unknown", "--alpha-discovery", "0.05", "--discovery-loss", "nt_xent") },
    @{ Name = "discovery_compare_ntxent_unknown002"; Extra = @("--limit-discovery", "1200", "--discovery-pool", "--discovery-pool-mode", "unknown", "--alpha-discovery", "0.05", "--alpha-discovery-unknown", "0.02", "--discovery-loss", "nt_xent") },
    @{ Name = "discovery_compare_ntxent_unknown"; Extra = @("--limit-discovery", "1200", "--discovery-pool", "--discovery-pool-mode", "unknown", "--alpha-discovery", "0.05", "--alpha-discovery-unknown", "0.05", "--discovery-loss", "nt_xent") }
)

foreach ($run in $Runs) {
    $trainRun = ".\runs\$($run.Name)"
    $studentCkpt = Join-Path $trainRun "student.pt"
    $trainArgs = $CommonTrain + @("--work-dir", $trainRun, "--student-ckpt", $studentCkpt) + $run.Extra
    Invoke-Python @trainArgs

    $detectRun = ".\runs\$($run.Name)_detect"
    $detectArgs = $CommonDetect + @("--work-dir", $detectRun, "--student-ckpt", $studentCkpt)
    Invoke-Python @detectArgs
}

Invoke-Python analyze_results.py --runs discovery_compare_base_detect discovery_compare_ntxent_detect discovery_compare_ntxent_unknown002_detect discovery_compare_ntxent_unknown_detect --root ".\runs" --out-dir ".\analysis\discovery_pool_compare"
Write-Host "discovery pool comparison finished."
