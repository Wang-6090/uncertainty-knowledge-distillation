$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Seeds = @(42, 123, 3407)
$Protocols = @(
    @{ Name = "random"; Split = ".\splits_cifar100_protocols\cifar100_60_40_random.json" },
    @{ Name = "hard"; Split = ".\splits_cifar100_protocols\cifar100_60_40_semantic_hard.json" },
    @{ Name = "isolated"; Split = ".\splits_cifar100_protocols\cifar100_60_40_semantic_isolated.json" }
)

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: python $($Arguments -join ' ')"
    }
}

foreach ($Seed in $Seeds) {
    foreach ($Protocol in $Protocols) {
        $Tag = "protocol_controlled_$($Protocol.Name)_s$Seed"
        $TeacherDir = ".\runs\${Tag}_teacher"
        $StudentDir = ".\runs\${Tag}_student"
        $DetectDir = ".\runs\${Tag}_detect"
        $TeacherCkpt = Join-Path $TeacherDir "teacher.pt"
        $StudentCkpt = Join-Path $StudentDir "student.pt"

        $Common = @(
            "--dataset", "cifar100",
            "--data-root", ".\data",
            "--num-known", "60",
            "--seed", "$Seed",
            "--split-path", $Protocol.Split,
            "--known-split-mode", "random",
            "--image-size", "64",
            "--batch-size", "32",
            "--num-workers", "0",
            "--limit-train", "1200",
            "--limit-val", "300",
            "--limit-test", "1000",
            "--device", "auto"
        )

        Write-Host "=== Teacher: $Tag ==="
        Invoke-CheckedPython (@("train.py", "train_teacher") + $Common + @(
            "--backbone", "resnet34",
            "--pretrained",
            "--epochs", "3",
            "--alpha-unc", "0.1",
            "--work-dir", $TeacherDir,
            "--teacher-ckpt", $TeacherCkpt
        ))

        Write-Host "=== Student: $Tag ==="
        Invoke-CheckedPython (@("train.py", "train_student") + $Common + @(
            "--backbone", "resnet18",
            "--teacher-backbone", "resnet34",
            "--pretrained",
            "--epochs", "3",
            "--alpha-unc", "0.1",
            "--alpha-kd", "1.0",
            "--kd-mode", "uncertainty",
            "--temperature", "2.0",
            "--uncertainty-weight-mode", "raw",
            "--alpha-feat-kd", "0.1",
            "--alpha-proto", "0.1",
            "--alpha-supcon", "0.1",
            "--work-dir", $StudentDir,
            "--teacher-ckpt", $TeacherCkpt,
            "--student-ckpt", $StudentCkpt
        ))

        Write-Host "=== Detection: $Tag ==="
        Invoke-CheckedPython (@("train.py", "discover") + $Common + @(
            "--backbone", "resnet18",
            "--student-backbone", "resnet18",
            "--num-novel", "40",
            "--mc-samples", "4",
            "--score-mode", "normalized_entropy_mahalanobis",
            "--threshold-policy", "known_coverage",
            "--target-known-coverage", "0.95",
            "--skip-clustering",
            "--work-dir", $DetectDir,
            "--student-ckpt", $StudentCkpt
        ))
    }
}

Write-Host "Controlled protocol audit completed."
