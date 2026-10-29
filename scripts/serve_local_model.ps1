# Serve one SFT candidate with vLLM on this PC's RTX 5090.
# The pipeline calls it as "local:<name>" (config.py local_llm_url, services/agents._llm).
#
#   .\scripts\serve_local_model.ps1 -Model Qwen/Qwen3.5-9B -Name qwen3.5-9b
#   .\scripts\serve_local_model.ps1 -Stop
#
# Weights come from C:\hf_models\hub (downloaded by C:\hf_models\download.py), read offline.
# Ministral reads its own format: -Extra --tokenizer-mode,mistral,--config-format,mistral,--load-format,mistral
# Every candidate is served in bf16 with the same flags, so the screening compares the models only.
param(
    [string]$Model,
    [string]$Name,
    [int]$MaxModelLen = 16384,
    [double]$GpuMemory = 0.92,
    [string[]]$Extra = @(),
    [switch]$Stop
)
$Image = "vllm/vllm-openai:v0.31.0"
docker rm -f sft-candidate 2>$null | Out-Null
if ($Stop) { return }
docker run -d --name sft-candidate --gpus all --ipc=host -p 8000:8000 `
    -v C:\hf_models\hub:/root/.cache/huggingface/hub -v C:\sft\runs:/adapters -e HF_HUB_OFFLINE=1 `
    $Image --model $Model --served-model-name $Name --dtype bfloat16 `
    --max-model-len $MaxModelLen --gpu-memory-utilization $GpuMemory --seed 0 @Extra
