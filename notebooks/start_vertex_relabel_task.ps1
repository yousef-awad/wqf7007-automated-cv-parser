param(
    [int]$Workers = 3,
    [double]$RequestsPerMinute = 20,
    [double]$SampleFrac = 0.5,
    [string]$Model = "gemini-3.1-flash-lite",
    [string]$Project = "project-12ae9020-458c-4247-8dd",
    [string]$Location = "global",
    [string]$DatasetName = "dataset4_gemini31_lite_relaxed_skills",
    [ValidateSet("strict", "relaxed")]
    [string]$SkillPolicy = "relaxed",
    [string]$Output = "data\processed\resume_bio_annotated_dataset4.csv",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$StdoutLog = Join-Path $LogDir "vertex-relabel-$Stamp.out.log"
$StderrLog = Join-Path $LogDir "vertex-relabel-$Stamp.err.log"

$Args = @(
    "notebooks/relabel_vertex_parallel.py",
    "--model", $Model,
    "--project", $Project,
    "--location", $Location,
    "--dataset-name", $DatasetName,
    "--skill-policy", $SkillPolicy,
    "--workers", "$Workers",
    "--requests-per-minute", "$RequestsPerMinute",
    "--sample-frac", "$SampleFrac",
    "--output", $Output
)

if ($Overwrite) {
    $Args += "--overwrite"
}

$Process = Start-Process `
    -FilePath "python" `
    -ArgumentList $Args `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Started Vertex relabel task."
Write-Host "PID       : $($Process.Id)"
Write-Host "Model     : $Model"
Write-Host "Dataset   : $DatasetName"
Write-Host "Skill rule: $SkillPolicy"
Write-Host "Workers   : $Workers"
Write-Host "Rate cap  : $RequestsPerMinute RPM"
Write-Host "Sample    : $SampleFrac per split"
Write-Host "stdout log: $StdoutLog"
Write-Host "stderr log: $StderrLog"
Write-Host "Output    : $(Join-Path $RepoRoot $Output)"
