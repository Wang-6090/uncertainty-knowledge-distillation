$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$DataRoot = ".\data"
$SplitPath = ".\splits_cifar100_60_40.json"
$AlphaList = @(0.0, 0.05, 0.1, 0.2)

foreach ($alpha in $AlphaList) {
  $tag = ($alpha.ToString().Replace(".", "_"))
  $RunDir = ".\runs\proto_alpha_$tag"
  $TeacherCkpt = Join-Path $RunDir "teacher.pt"
  $StudentCkpt = Join-Path $RunDir "student.pt"

  python train.py train_teacher --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --epochs 10 --limit-train 5000 --limit-val 1000 --limit-test 2000 --device auto --alpha-proto $alpha --work-dir $RunDir --teacher-ckpt $TeacherCkpt
  python train.py train_student --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --epochs 10 --limit-train 5000 --limit-val 1000 --limit-test 2000 --device auto --alpha-proto $alpha --work-dir $RunDir --teacher-ckpt $TeacherCkpt --student-ckpt $StudentCkpt
  python train.py discover --dataset cifar100 --data-root $DataRoot --num-known 60 --seed 42 --split-path $SplitPath --image-size 64 --batch-size 32 --num-workers 0 --backbone resnet18 --limit-train 5000 --limit-val 1000 --limit-test 2000 --num-novel 40 --mc-samples 4 --score-mode entropy_proto --device auto --work-dir $RunDir --student-ckpt $StudentCkpt
}

python analyze_results.py --runs proto_alpha_0_0 proto_alpha_0_05 proto_alpha_0_1 proto_alpha_0_2 --root .\runs --out-dir .\analysis\proto_alpha_sweep
