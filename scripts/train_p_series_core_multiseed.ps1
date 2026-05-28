param(
    [string]$PythonBin = "D:\venvs\PGM_project\Scripts\python.exe",
    [string]$TrainFile = "C:\data\diving48_embeddings\clip_vit_b16\train.pt",
    [string]$ValFile = "C:\data\diving48_embeddings\clip_vit_b16\test.pt",
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path "logs\p_series_multiseed" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\p_series_multiseed" | Out-Null

$Runs = @(
    @{
        Id = "P3"
        Name = "P3_noPGM"
        Config = "configs\p_series\P3_trajectory_matrix_linear_noPGM.yaml"
    },
    @{
        Id = "P12"
        Name = "P12_prePGM_lam030"
        Config = "configs\p_series\P12_prePGM_lam030_trajectory_matrix_linear.yaml"
    },
    @{
        Id = "P17"
        Name = "P17_postPGM_lam080"
        Config = "configs\p_series\P17_trajectory_matrix_linear_pgm_lam080.yaml"
    }
)

$Seeds = @(1, 2)

Write-Host "P-series core multiseed trainer"
Write-Host "Started: $(Get-Date -Format o)"
Write-Host "Python: $PythonBin"
Write-Host "Train file: $TrainFile"
Write-Host "Val file: $ValFile"
Write-Host "Device: $Device"

foreach ($Run in $Runs) {
    foreach ($Seed in $Seeds) {
        $RunDir = "outputs\p_series_multiseed\$($Run.Name)_seed$Seed"
        $LogPath = "logs\p_series_multiseed\$($Run.Name)_seed$Seed.log"
        $Best = Join-Path $RunDir "checkpoints\best.pt"
        $Last = Join-Path $RunDir "checkpoints\last.pt"
        $Metrics = Join-Path $RunDir "metrics.json"

        if ((Test-Path $Best) -and (Test-Path $Last) -and (Test-Path $Metrics)) {
            Write-Host "[$(Get-Date -Format o)] SKIP $($Run.Id) seed=${Seed}: completed at $RunDir"
            continue
        }

        Write-Host "[$(Get-Date -Format o)] START $($Run.Id) seed=$Seed"
        Write-Host "Command: $PythonBin -u scripts\train_embeddings.py --config $($Run.Config) --train_file $TrainFile --val_file $ValFile --device $Device --seed $Seed --run_dir $RunDir"

        & $PythonBin -u scripts\train_embeddings.py `
            --config $Run.Config `
            --train_file $TrainFile `
            --val_file $ValFile `
            --device $Device `
            --seed $Seed `
            --run_dir $RunDir 2>&1 | Tee-Object -FilePath $LogPath

        if ($LASTEXITCODE -ne 0) {
            throw "Run $($Run.Id) seed=$Seed failed with exit code $LASTEXITCODE"
        }

        Write-Host "[$(Get-Date -Format o)] FINISH $($Run.Id) seed=$Seed"
    }
}

Write-Host "Finished: $(Get-Date -Format o)"
