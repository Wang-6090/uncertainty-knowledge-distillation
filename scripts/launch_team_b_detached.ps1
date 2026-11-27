<#
.SYNOPSIS
    以 Windows 计划任务方式启动 / 停止李柯颖线（team B）的分离式训练。

.DESCRIPTION
    直接用 DSH 后台任务或 WMI 启动训练时，进程会挂在一个控制台上：
    只要界面/会话结束，进程就会收到 Ctrl+C（退出码 0xC000013A），
    已经训练到一半的 student 因为 train.py 只在训练结束才写 student.pt 而全部丢失。
    计划任务运行在任务计划程序自己的会话里，不受控制台中断影响。

    本脚本会为每个 (seed, mode) 组合生成一个 .cmd 包装脚本，并注册一个
    dsh_tb_<tag>_s<seed>_<mode> 计划任务，然后（可选）立即启动。

.PARAMETER Tag
    输出子目录名，例如 formal_20261124（对应 runs/team_b_rejection/<tag>）。

.PARAMETER Epochs
    教师/学生训练轮数，默认 30。

.PARAMETER Launch
    真正启动计划任务；不加时只生成脚本并打印，不启动。

.PARAMETER Unregister
    停止并删除本 tag 的所有计划任务。

.EXAMPLE
    pwsh -File .\scripts\launch_team_b_detached.ps1 -Tag formal_20261124
    pwsh -File .\scripts\launch_team_b_detached.ps1 -Tag formal_20261124 -Launch
    pwsh -File .\scripts\launch_team_b_detached.ps1 -Tag formal_20261124 -Unregister
#>
[CmdletBinding()]
param(
    [string]$Tag = 'formal_20261124',
    [string]$Seeds = '42,123,3407',
    [string]$Modes = 'scratch_cnn,shared_backbone',
    [int]$Epochs = 30,
    [int]$McSamples = 8,
    [string]$ImageSize = '64',
    [switch]$Launch,
    [switch]$Unregister
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv_cuda\Scripts\python.exe'
$runRoot = Join-Path $root "runs\team_b_rejection\$Tag"
$jobDir = Join-Path $runRoot '_jobs'

if (-not (Test-Path $python)) {
    throw "找不到 CUDA venv 的 python：$python。先用 .venv_cuda 装好 torch cu130。"
}

$seedList = $Seeds.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$modeList = $Modes.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ }

function Get-TaskName([string]$seed, [string]$mode) {
    "dsh_tb_${Tag}_s${seed}_${mode}"
}

if ($Unregister) {
    foreach ($seed in $seedList) {
        foreach ($mode in $modeList) {
            $name = Get-TaskName $seed $mode
            $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
            if ($task) {
                Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
                Unregister-ScheduledTask -TaskName $name -Confirm:$false
                Write-Host "removed task: $name"
            }
        }
    }
    exit 0
}

New-Item -ItemType Directory -Force -Path $jobDir | Out-Null
$enc = [System.Text.Encoding]::GetEncoding(936)   # cmd.exe 使用系统 ANSI(GBK) 读取批处理

foreach ($seed in $seedList) {
    foreach ($mode in $modeList) {
        $log = Join-Path $runRoot "_job_s${seed}_${mode}.out"
        $argLine = "scripts\run_team_b_rejection.py --mode formal --tag $Tag " +
                   "--seeds $seed --init-modes $mode " +
                   "--teacher-epochs $Epochs --student-epochs $Epochs " +
                   "--device cuda --threads 2 --num-workers 0 " +
                   "--image-size $ImageSize --mc-samples $McSamples " +
                   "--skip-existing --save-epoch-checkpoints --skip-analysis"
        $text = "@echo off`r`n" +
                "cd /d `"$root`"`r`n" +
                "set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`r`n" +
                "`"$python`" $argLine > `"$log`" 2>&1`r`n" +
                "echo EXITCODE=%ERRORLEVEL% >> `"$log`"`r`n"
        $cmdFile = Join-Path $jobDir "run_s${seed}_${mode}.cmd"
        [System.IO.File]::WriteAllText($cmdFile, $text, $enc)

        $name = Get-TaskName $seed $mode
        $action = New-ScheduledTaskAction -Execute 'cmd.exe' `
            -Argument "/c `"$cmdFile`"" -WorkingDirectory $root
        $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddYears(1)
        $principal = New-ScheduledTaskPrincipal `
            -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
        Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
            -Principal $principal -Force | Out-Null
        Write-Host "registered task: $name -> $cmdFile"

        if ($Launch) {
            Start-ScheduledTask -TaskName $name
            Write-Host "started task: $name"
            # 错峰启动：6 个进程同时初始化会在 8 GB 显卡上触发 CUDA OOM
            Start-Sleep -Seconds 40
        }
    }
}

if (-not $Launch) {
    Write-Host ''
    Write-Host '只生成未启动。确认后加 -Launch 再执行一次。'
} else {
    Write-Host ''
    Write-Host "6 个任务已启动。查看进度："
    Write-Host "  Get-ScheduledTask -TaskName 'dsh_tb_${Tag}_*' | Select TaskName,State"
    Write-Host "  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv"
    Write-Host "  Get-ChildItem '$runRoot' -Directory -Filter 's*_student' | ForEach-Object { (Select-String (Join-Path `$_.FullName 'train.log') -Pattern '\[student\]\[' | Measure-Object).Count }"
}
