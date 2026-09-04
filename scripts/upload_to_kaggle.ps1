$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$datasetDir = Join-Path $repoRoot "data/generated/kaggle-20260903-01"
$kernelDir = Join-Path $repoRoot "notebooks/kaggle"

Write-Host "=== 1. Uploading dataset to Kaggle ===" -ForegroundColor Cyan
Set-Location $datasetDir
try {
    kaggle datasets create -p . -r zip
} catch {
    Write-Host "Dataset may already exist, uploading new version..." -ForegroundColor Yellow
    kaggle datasets version -p . -m "Updated dataset and source" -r zip
}

Write-Host "=== 2. Pushing training kernel to Kaggle GPU ===" -ForegroundColor Cyan
Set-Location $repoRoot
kaggle kernels push -p $kernelDir

Write-Host "=== 3. Checking kernel status ===" -ForegroundColor Cyan
kaggle kernels status trietp1253201581/anomaly-representation-training-gpu
