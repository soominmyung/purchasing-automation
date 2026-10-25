# Serve each remaining SFT candidate in turn and screen it (scripts/screen_base_models.py).
$Root = Split-Path $PSScriptRoot -Parent
$M = "--tokenizer-mode","mistral","--config-format","mistral","--load-format","mistral"
$Candidates = @(
    @{Model="google/gemma-4-12B-it"; Name="gemma-4-12b-it"; Extra=@()},
    @{Model="google/gemma-4-E4B-it"; Name="gemma-4-e4b-it"; Extra=@()},
    @{Model="mistralai/Ministral-3-8B-Instruct-2512-BF16"; Name="ministral-3-8b"; Extra=$M},
    @{Model="mistralai/Ministral-3-14B-Instruct-2512-BF16"; Name="ministral-3-14b"; Extra=$M}
)
foreach ($c in $Candidates) {
    & "$PSScriptRoot\serve_local_model.ps1" -Model $c.Model -Name $c.Name -Extra $c.Extra | Out-Null
    $ok = $false
    for ($i = 0; $i -lt 90; $i++) {
        try { Invoke-RestMethod http://localhost:8000/v1/models -TimeoutSec 3 | Out-Null; $ok = $true; break } catch { Start-Sleep 10 }
        if (-not (docker ps -q -f name=sft-candidate)) { break }
    }
    if (-not $ok) {
        Write-Output "$($c.Name): server did not start"
        docker logs --tail 40 sft-candidate 2>&1 | Out-File -Encoding utf8 "$Root\eval_results\sft_screen_$($c.Name)_server_error.log"
        continue
    }
    Write-Output "$($c.Name): serving"
    python "$Root\scripts\screen_base_models.py" --model "local:$($c.Name)" 2>&1 | Select-String -NotMatch "Deprecat|_stores\[" | ForEach-Object { $_.Line }
}
& "$PSScriptRoot\serve_local_model.ps1" -Stop
python "$Root\scripts\screen_base_models.py" --report
