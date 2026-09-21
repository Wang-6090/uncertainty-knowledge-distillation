param(
    [string]$DataRoot = ".\data",
    [string]$SplitPath = ".\splits_cifar100_60_40.json",
    [int[]]$Seeds = @(42, 43, 44),
    [int]$Epochs = 15,
    [int]$BatchSize = 64,
    [int]$ImageSize = 64,
    [int]$LimitTrain = 0,
    [int]$LimitVal = 0,
    [int]$LimitTest = 0
)

$ErrorActionPreference = "Stop"
$methods = @(
    @{ Name = "ce"; Unc = 0.0; KD = 0.0; Feat = 0.0; SupCon = 0.0; Proto = 0.0 },
    @{ Name = "standard_kd"; Unc = 0.0; KD = 1.0; Feat = 0.0; SupCon = 0.0; Proto = 0.0 },
    @{ Name = "uncertainty_kd"; Unc = 0.0; KD = 1.0; Feat = 0.0; SupCon = 0.0; Proto = 0.0; KDMode = "uncertainty" },
    @{ Name = "feature_kd"; Unc = 0.0; KD = 1.0; Feat = 0.1; SupCon = 0.0; Proto = 0.0 },
    @{ Name = "full"; Unc = 0.1; KD = 1.0; Feat = 0.1; SupCon = 0.1; Proto = 0.1 }
)

function Invoke-Python([string[]]$Arguments) {
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "python failed with exit code $LASTEXITCODE" }
}

foreach ($seed in $Seeds) {
    $teacherRun = ".\runs\protocol_s${seed}_teacher"
    $teacherCkpt = Join-Path $teacherRun "teacher.pt"
    if (-not (Test-Path $teacherCkpt)) {
        Invoke-Python @("train.py", "train_teacher", "--dataset", "cifar100", "--data-root", $DataRoot, "--num-known", "60", "--num-novel", "40", "--seed", $seed, "--split-path", $SplitPath, "--image-size", $ImageSize, "--batch-size", $BatchSize, "--backbone", "resnet34", "--teacher-backbone", "resnet34", "--pretrained", "--epochs", $Epochs, "--limit-train", $LimitTrain, "--limit-val", $LimitVal, "--limit-test", $LimitTest, "--work-dir", $teacherRun, "--teacher-ckpt", $teacherCkpt, "--device", "auto")
    }
    foreach ($method in $methods) {
        $run = ".\runs\protocol_s${seed}_$($method.Name)"
        $student = Join-Path $run "student.pt"
        $detect = "${run}_detect"
        if (-not (Test-Path $student)) {
            $kdMode = if ($method.ContainsKey("KDMode")) { $method.KDMode } else { "standard" }
            Invoke-Python @("train.py", "train_student", "--dataset", "cifar100", "--data-root", $DataRoot, "--num-known", "60", "--num-novel", "40", "--seed", $seed, "--split-path", $SplitPath, "--image-size", $ImageSize, "--batch-size", $BatchSize, "--teacher-backbone", "resnet34", "--student-backbone", "resnet18", "--pretrained", "--epochs", $Epochs, "--alpha-unc", $method.Unc, "--alpha-kd", $method.KD, "--kd-mode", $kdMode, "--alpha-feat-kd", $method.Feat, "--alpha-supcon", $method.SupCon, "--alpha-proto", $method.Proto, "--limit-train", $LimitTrain, "--limit-val", $LimitVal, "--limit-test", $LimitTest, "--teacher-ckpt", $teacherCkpt, "--student-ckpt", $student, "--work-dir", $run, "--device", "auto")
        }
        if (-not (Test-Path (Join-Path $detect "discovery_report.json"))) {
            Invoke-Python @("train.py", "discover", "--dataset", "cifar100", "--data-root", $DataRoot, "--num-known", "60", "--num-novel", "40", "--seed", $seed, "--split-path", $SplitPath, "--image-size", $ImageSize, "--batch-size", $BatchSize, "--student-backbone", "resnet18", "--pretrained", "--score-mode", "normalized_entropy_mahalanobis", "--cluster-k", "auto", "--limit-train", $LimitTrain, "--limit-val", $LimitVal, "--limit-test", $LimitTest, "--student-ckpt", $student, "--work-dir", $detect, "--device", "auto")
        }
    }
}

Invoke-Python @("analyze_multiseed.py", "--root", ".\runs", "--glob", "protocol_s*_*_detect", "--out-dir", ".\analysis\recommended_protocol")
Write-Host "Recommended protocol completed. See .\analysis\recommended_protocol\multiseed_summary.md"
