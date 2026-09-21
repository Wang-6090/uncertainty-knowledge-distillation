$ErrorActionPreference = "Stop"
$env:TQDM_DISABLE = "1"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Keep pretrained weights inside the project so Windows does not depend on
# permissions of the user's global Torch cache.
$env:TORCH_HOME = Join-Path $Root ".torch_cache"
New-Item -ItemType Directory -Force -Path $env:TORCH_HOME | Out-Null

function Invoke-Python {
    & python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

# Interim robustness experiment: repeat the A-E ablation for the completed
# seeds. Add 3407 back for the final three-seed result when resources permit.
# Each student is evaluated with oracle K and automatically estimated K.
$Seeds = @(42, 123)
$Epochs = 15
$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"

$experiments = @(
    @{ Name = "A_ce"; KD = 0.0; UKD = "standard"; Feat = 0.0; Unc = 0.0; SupCon = 0.0; Proto = 0.0 },
    @{ Name = "B_standard_kd"; KD = 1.0; UKD = "standard"; Feat = 0.0; Unc = 0.0; SupCon = 0.0; Proto = 0.0 },
    @{ Name = "C_uncertainty_kd"; KD = 1.0; UKD = "uncertainty"; Feat = 0.0; Unc = 0.0; SupCon = 0.0; Proto = 0.0 },
    @{ Name = "D_uncertainty_feature_kd"; KD = 1.0; UKD = "uncertainty"; Feat = 0.1; Unc = 0.0; SupCon = 0.0; Proto = 0.0 },
    @{ Name = "E_full_representation"; KD = 1.0; UKD = "uncertainty"; Feat = 0.1; Unc = 0.1; SupCon = 0.1; Proto = 0.1 }
)

foreach ($seed in $Seeds) {
    $teacherRun = ".\runs\revised_ms_s${seed}_teacher"
    $teacherCkpt = Join-Path $teacherRun "teacher.pt"
    if (-not (Test-Path $teacherCkpt)) {
        Invoke-Python train.py train_teacher --dataset cifar100 --data-root $DataRoot --download --num-known 60 --seed $seed --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet34 --teacher-backbone resnet34 --pretrained --epochs $Epochs --alpha-unc 0.1 --alpha-proto 0.0 --alpha-pseudo 0.0 --work-dir $teacherRun --teacher-ckpt $teacherCkpt --device auto
    } else {
        Write-Host "skip existing teacher: $teacherCkpt"
    }

    foreach ($exp in $experiments) {
        $studentRun = ".\runs\revised_ms_s${seed}_$($exp.Name)"
        $studentCkpt = Join-Path $studentRun "student.pt"
        if (-not (Test-Path $studentCkpt)) {
            Invoke-Python train.py train_student --dataset cifar100 --data-root $DataRoot --download --num-known 60 --seed $seed --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet18 --teacher-backbone resnet34 --student-backbone resnet18 --pretrained --epochs $Epochs --alpha-unc $exp.Unc --alpha-kd $exp.KD --kd-mode $exp.UKD --alpha-feat-kd $exp.Feat --alpha-supcon $exp.SupCon --alpha-proto $exp.Proto --alpha-pseudo 0.0 --work-dir $studentRun --teacher-ckpt $teacherCkpt --student-ckpt $studentCkpt --device auto
        } else {
            Write-Host "skip existing student: $studentCkpt"
        }

        foreach ($clusterMode in @("oracle", "auto")) {
            $detectRun = ".\runs\revised_ms_s${seed}_$($exp.Name)_detect_$clusterMode"
            $reportPath = Join-Path $detectRun "discovery_report.json"
            if (-not (Test-Path $reportPath)) {
                Invoke-Python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed $seed --split-path $SplitPath --image-size 64 --batch-size 64 --num-workers 0 --backbone resnet18 --student-backbone resnet18 --pretrained --num-novel 40 --mc-samples 8 --score-mode normalized_entropy_mahalanobis --cluster-k $clusterMode --work-dir $detectRun --student-ckpt $studentCkpt --device auto
            } else {
                Write-Host "skip existing report: $reportPath"
            }
        }
    }
}

Invoke-Python analyze_multiseed.py --root .\runs --glob "revised_ms_s*_detect_*" --out-dir .\analysis\revised_multiseed
Write-Host "multi-seed revised ablation finished."
