# Gemma 4 12B (untuned) served bf16, then fp8, on the RTX 5090: quality on the 25 evaluation cases
# with the production checks, and throughput on the production request (scripts/bench_vllm_v5.py).
$Root = Split-Path $PSScriptRoot -Parent
foreach ($q in @(@{Label="bf16"; Extra=@()}, @{Label="fp8"; Extra=@("--quantization","fp8")})) {
    & "$PSScriptRoot\serve_local_model.ps1" -Model google/gemma-4-12B-it -Name gemma-4-12b-it -Extra $q.Extra | Out-Null
    $ok = $false
    for ($i = 0; $i -lt 90; $i++) { try { Invoke-RestMethod http://localhost:8000/v1/models -TimeoutSec 3 | Out-Null; $ok = $true; break } catch { Start-Sleep 10 } }
    if (-not $ok) { "$($q.Label): server did not start"; continue }
    docker logs sft-candidate 2>&1 | Select-String "KV cache|Maximum concurrency|model weights" | Select-Object -Last 3 | ForEach-Object { "$($q.Label) server: " + $_.Line }
    "=== $($q.Label) quality $(Get-Date -Format HH:mm)"
    $env:LOCAL_LLM_MAX_TOKENS = "6000"
    python "$Root\scripts\screen_base_models.py" --model local:gemma-4-12b-it --tag "serve-$($q.Label)" --workers 6 2>&1 | Select-String "^->|Error|Traceback" | ForEach-Object { $_.Line }
    "=== $($q.Label) throughput $(Get-Date -Format HH:mm)"
    python "$Root\scripts\bench_vllm_v5.py" --model gemma-4-12b-it --label "gemma4-12b-$($q.Label)" 2>&1 | Select-String "concurrency|->|Error" | ForEach-Object { $_.Line }
}
"=== ALL DONE $(Get-Date -Format HH:mm)"
