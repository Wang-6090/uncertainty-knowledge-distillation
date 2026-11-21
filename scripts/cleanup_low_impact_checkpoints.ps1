param(
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$protectedTokens = @(
    'team_a_quality_kd_confirm_s2155',
    'team_a_quality_kd_confirm_s2156',
    'team_a_quality_kd_s2154',
    'rejection_gradient_conflict_formal_s2152',
    'rejection_gradient_componentwise_formal_s2153',
    'rejection_relational_kd_s2140',
    'formal_conjunction_audit_20261121'
)

$candidateFiles = Get-ChildItem 'runs' -File -Recurse | Where-Object {
    $_.Extension -in '.pt', '.pth', '.ckpt' -and
    ($_.FullName -match '(?i)\\archive_old\\' -or
     $_.FullName -match '(?i)\\_archive_confirmed_invalid_20261024\\' -or
     $_.DirectoryName -match '(?i)smoke') -and
    -not (($protectedTokens | Where-Object {
        $_.DirectoryName -like "*$_*"
    }).Count -gt 0)
}

$summary = $candidateFiles | Measure-Object Length -Sum
$candidateFiles |
    Group-Object DirectoryName |
    Sort-Object Count -Descending |
    Select-Object -First 30 Name, Count, @{Name='MB'; Expression={
        [math]::Round((($_.Group | Measure-Object Length -Sum).Sum) / 1MB, 1)
    }} |
    Format-Table -AutoSize

Write-Output "Candidate checkpoint files: $($summary.Count)"
Write-Output "Candidate size GB: $([math]::Round($summary.Sum / 1GB, 3))"
Write-Output "Mode: $(if ($Execute) { 'EXECUTE' } else { 'PREVIEW; no files changed' })"

if (-not $Execute) {
    Write-Output 'Run again with -Execute only after reviewing this scope.'
    exit 0
}

$manifest = Join-Path (Get-Location) 'analysis/cleanup_low_impact_manifest_20261121.csv'
$candidateFiles |
    Select-Object FullName, Length, LastWriteTime |
    Export-Csv -LiteralPath $manifest -NoTypeInformation -Encoding UTF8

foreach ($file in $candidateFiles) {
    Remove-Item -LiteralPath $file.FullName -Force
}

Write-Output "Removed $($summary.Count) checkpoint files."
Write-Output "Manifest: $manifest"
