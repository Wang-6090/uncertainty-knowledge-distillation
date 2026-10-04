$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$TeacherRun = ".\runs\revised_teacher_resnet34"
$TeacherCkpt = Join-Path $TeacherRun "teacher.pt"

# Compare the current revised student against an Outlier-Exposure-style
# variant that uses the unlabeled novel discovery pool for energy separation.
$experiments = @(
    @{ Name = "base"; DiscoveryEnergy = 0.0; UseDiscovery = $false },
    @{ Name = "discovery_energy"; DiscoveryEnergy = 0.1; UseDiscovery = $true }
)

foreach ($exp in $experiments) {
    $run = ".\runs\discovery_energy_$($exp.Name)"
    $student = Join-Path $run "student.pt"
    $detectRun = ".\runs\discovery_energy_$($exp.Name)_detect"
    $extra = @()
    if ($exp.UseDiscovery) {
        $extra += @(
            "--discovery-pool",
            "--discovery-pool-mode", "unknown",
            "--limit-discovery", "4000",
            "--alpha-discovery", "0.05",
            "--alpha-discovery-energy", "$($exp.DiscoveryEnergy)",
            "--discovery-loss", "nt_xent"
        )
    }

    python train.py train_student --dataset cifar100 --data-root $DataRoot `
      --num-known 60 --seed 42 --split-path $SplitPath `
      --backbone resnet18 --teacher-backbone resnet34 --student-backbone resnet18 `
      --pretrained --image-size 64 --batch-size 64 --num-workers 0 --epochs 15 `
      --work-dir $run --teacher-ckpt $TeacherCkpt --student-ckpt $student --device auto `
      @extra

    python train.py discover --dataset cifar100 --data-root $DataRoot `
      --num-known 60 --seed 42 --split-path $SplitPath `
      --backbone resnet18 --student-backbone resnet18 --pretrained `
      --image-size 64 --batch-size 64 --num-workers 0 --num-novel 40 --mc-samples 8 `
      --score-mode normalized_entropy_mahalanobis --cluster-k auto `
      --cluster-method kmeans --cluster-feature projection `
      --work-dir $detectRun --student-ckpt $student --device auto
}

python analyze_results.py --runs discovery_energy_base_detect discovery_energy_discovery_energy_detect `
  --root .\runs --out-dir .\analysis\discovery_energy_compare
