param(
    [string]$PythonBin = "",
    [string]$TrainFile = "C:\data\diving48_embeddings\clip_vit_b16\train.pt",
    [string]$ValFile = "C:\data\diving48_embeddings\clip_vit_b16\test.pt",
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

if (-not $PythonBin) {
    if (Test-Path "D:\venvs\PGM_project\Scripts\python.exe") {
        $PythonBin = "D:\venvs\PGM_project\Scripts\python.exe"
    } else {
        $PythonBin = "python"
    }
}

New-Item -ItemType Directory -Force -Path "logs\p_series" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\p_series" | Out-Null

$Runs = @(
    @{
        Id = "P14"
        Config = "configs\p_series\P14_trajectory_matrix_linear_pgm_lam050.yaml"
        RunDir = "outputs\p_series\P14_trajectory_matrix_linear_pgm_lam050"
        Log = "logs\p_series\P14_trajectory_matrix_linear_pgm_lam050.log"
    },
    @{
        Id = "P15"
        Config = "configs\p_series\P15_trajectory_matrix_linear_pgm_lam060.yaml"
        RunDir = "outputs\p_series\P15_trajectory_matrix_linear_pgm_lam060"
        Log = "logs\p_series\P15_trajectory_matrix_linear_pgm_lam060.log"
    },
    @{
        Id = "P16"
        Config = "configs\p_series\P16_trajectory_matrix_linear_pgm_lam070.yaml"
        RunDir = "outputs\p_series\P16_trajectory_matrix_linear_pgm_lam070"
        Log = "logs\p_series\P16_trajectory_matrix_linear_pgm_lam070.log"
    },
    @{
        Id = "P17"
        Config = "configs\p_series\P17_trajectory_matrix_linear_pgm_lam080.yaml"
        RunDir = "outputs\p_series\P17_trajectory_matrix_linear_pgm_lam080"
        Log = "logs\p_series\P17_trajectory_matrix_linear_pgm_lam080.log"
    },
    @{
        Id = "P18"
        Config = "configs\p_series\P18_trajectory_matrix_linear_pgm_lam090.yaml"
        RunDir = "outputs\p_series\P18_trajectory_matrix_linear_pgm_lam090"
        Log = "logs\p_series\P18_trajectory_matrix_linear_pgm_lam090.log"
    },
    @{
        Id = "P19"
        Config = "configs\p_series\P19_trajectory_matrix_linear_pgm_lam100.yaml"
        RunDir = "outputs\p_series\P19_trajectory_matrix_linear_pgm_lam100"
        Log = "logs\p_series\P19_trajectory_matrix_linear_pgm_lam100.log"
    },
    @{
        Id = "P20"
        Config = "configs\p_series\P20_prePGM_lam050_trajectory_matrix_linear.yaml"
        RunDir = "outputs\p_series\P20_prePGM_lam050_trajectory_matrix_linear"
        Log = "logs\p_series\P20_prePGM_lam050_trajectory_matrix_linear.log"
    },
    @{
        Id = "P21"
        Config = "configs\p_series\P21_prePGM_lam060_trajectory_matrix_linear.yaml"
        RunDir = "outputs\p_series\P21_prePGM_lam060_trajectory_matrix_linear"
        Log = "logs\p_series\P21_prePGM_lam060_trajectory_matrix_linear.log"
    },
    @{
        Id = "P22"
        Config = "configs\p_series\P22_prePGM_lam070_trajectory_matrix_linear.yaml"
        RunDir = "outputs\p_series\P22_prePGM_lam070_trajectory_matrix_linear"
        Log = "logs\p_series\P22_prePGM_lam070_trajectory_matrix_linear.log"
    },
    @{
        Id = "P23"
        Config = "configs\p_series\P23_prePGM_lam080_trajectory_matrix_linear.yaml"
        RunDir = "outputs\p_series\P23_prePGM_lam080_trajectory_matrix_linear"
        Log = "logs\p_series\P23_prePGM_lam080_trajectory_matrix_linear.log"
    },
    @{
        Id = "P24"
        Config = "configs\p_series\P24_prePGM_lam090_trajectory_matrix_linear.yaml"
        RunDir = "outputs\p_series\P24_prePGM_lam090_trajectory_matrix_linear"
        Log = "logs\p_series\P24_prePGM_lam090_trajectory_matrix_linear.log"
    },
    @{
        Id = "P25"
        Config = "configs\p_series\P25_prePGM_lam100_trajectory_matrix_linear.yaml"
        RunDir = "outputs\p_series\P25_prePGM_lam100_trajectory_matrix_linear"
        Log = "logs\p_series\P25_prePGM_lam100_trajectory_matrix_linear.log"
    }
)

Write-Host "P-series lambda 0.50-1.00 trainer"
Write-Host "Started: $(Get-Date -Format o)"
Write-Host "Python: $PythonBin"
Write-Host "Train file: $TrainFile"
Write-Host "Val file: $ValFile"
Write-Host "Device: $Device"

$Trained = New-Object System.Collections.Generic.List[string]
$Skipped = New-Object System.Collections.Generic.List[string]

foreach ($Run in $Runs) {
    $Best = Join-Path $Run.RunDir "checkpoints\best.pt"
    $Last = Join-Path $Run.RunDir "checkpoints\last.pt"
    $Metrics = Join-Path $Run.RunDir "metrics.json"

    if ((Test-Path $Best) -and (Test-Path $Last) -and (Test-Path $Metrics)) {
        Write-Host "[$(Get-Date -Format o)] SKIP $($Run.Id): completed at $($Run.RunDir)"
        $Skipped.Add("$($Run.Id):$($Run.RunDir)") | Out-Null
        continue
    }

    Write-Host "[$(Get-Date -Format o)] START $($Run.Id)"
    $CommandText = "$PythonBin -u scripts\train_embeddings.py --config $($Run.Config) --train_file $TrainFile --val_file $ValFile --device $Device --run_dir $($Run.RunDir)"
    Write-Host "Command: $CommandText"

    & $PythonBin -u scripts\train_embeddings.py `
        --config $Run.Config `
        --train_file $TrainFile `
        --val_file $ValFile `
        --device $Device `
        --run_dir $Run.RunDir 2>&1 | Tee-Object -FilePath $Run.Log

    if ($LASTEXITCODE -ne 0) {
        throw "Run $($Run.Id) failed with exit code $LASTEXITCODE"
    }

    $Trained.Add("$($Run.Id):$($Run.RunDir)") | Out-Null
    Write-Host "[$(Get-Date -Format o)] FINISH $($Run.Id)"
}

Write-Host ""
Write-Host "Finished: $(Get-Date -Format o)"
Write-Host "Skipped completed runs:"
if ($Skipped.Count -eq 0) {
    Write-Host "  none"
} else {
    foreach ($Item in $Skipped) {
        Write-Host "  $Item"
    }
}
Write-Host "Newly trained runs:"
if ($Trained.Count -eq 0) {
    Write-Host "  none"
} else {
    foreach ($Item in $Trained) {
        Write-Host "  $Item"
    }
}
