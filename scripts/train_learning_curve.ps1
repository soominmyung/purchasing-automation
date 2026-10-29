# The learning curve: all passing runs first (seed 1), then 25 and 50 runs, then the
# second seed of each. Each run is scripts/train_sft_v5.py in the Unsloth container; adapters and logs in C:\sft\runs.
$Root = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force C:\sft\work\scripts, C:\sft\work\training_data\sft_v5, C:\sft\runs | Out-Null
Copy-Item "$Root\scripts\train_sft_v5.py" C:\sft\work\scripts\ -Force
Copy-Item "$Root\training_data\sft_v5\teacher_scenarios.jsonl" C:\sft\work\training_data\sft_v5\ -Force
$runs = @(@(0,1), @(25,1), @(50,1), @(0,2), @(25,2), @(50,2))
foreach ($r in $runs) {
    "=== suppliers $($r[0]) seed $($r[1]) started $(Get-Date -Format HH:mm)"
    docker run --rm --gpus all --ipc=host -v C:\sft\work:/work -v C:\sft\runs:/runs `
        -v C:\hf_models\hub:/hfhub -e HF_HUB_CACHE=/hfhub -e HF_HUB_OFFLINE=1 -w /work `
        --entrypoint python unsloth/unsloth:core-v0.1.903-beta `
        scripts/train_sft_v5.py --suppliers $r[0] --seed $r[1] --epochs 3 2>&1 |
        Select-String -Pattern "^gemma4|eval_loss|^done|Error|Traceback|assert" | ForEach-Object { $_.Line }
}
"=== ALL DONE $(Get-Date -Format HH:mm)"
