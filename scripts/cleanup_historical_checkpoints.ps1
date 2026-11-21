param(
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$cutoff = [datetime]'2026-11-10'
$protectedTokens = @(
    'team_a_quality_kd_confirm_s2155',
    'team_a_quality_kd_confirm_s2156',
    'team_a_quality_kd_s2154',
    'rejection_gradient_conflict_formal_s2152',
    'rejection_gradient_componentwise_formal_s2153',
    'rejection_relational_kd_s2140',
    'formal_conjunction_audit_20261121',
    '_archive_confirmed_invalid_20261024'
)

$candidateFiles = Get-ChildItem 'runs' -File -Recurse | Where-Object {
    $_.Extension -in '.pt', '.pth', '.ckpt' -and
    $_.Directory.LastWriteTime -lt $cutoff -and
    -not (($protectedTokens | Where-Object {
        $_.DirectoryName -like "*$_*"
    }).Count -gt 0)
}

$summary = $candidateFiles | Measure-Object Length -Sum
$candidateFiles |
    Group-Object DirectoryName |
    Sort-Object Count -Descending |
    Select-Object -First 20 Name, Count, @{Name='GB'; Expression={
        [math]::Round((($_.Group | Measure-Object Length -Sum).Sum) / 1GB, 3)
    }} |
    Format-Table -AutoSize

Write-Output "Candidate checkpoint files: $($summary.Count)"
Write-Output "Candidate size GB: $([math]::Round($summary.Sum / 1GB, 3))"
Write-Output "Mode: $(if ($Execute) { 'EXECUTE' } else { 'PREVIEW; no files changed' })"

if (-not $Execute) {
    Write-Output 'Run again with -Execute only after reviewing this scope.'
    exit 0
}

$manifest = Join-Path (Get-Location) 'analysis/cleanup_checkpoint_manifest_20261121.csv'
$candidateFiles |
    Select-Object FullName, Length, LastWriteTime |
    Export-Csv -LiteralPath $manifest -NoTypeInformation -Encoding UTF8

foreach ($file in $candidateFiles) {
    Remove-Item -LiteralPath $file.FullName -Force
}

Write-Output "Removed $($summary.Count) checkpoint files."
Write-Output "Manifest: $manifest"
