# Self-Hosted Inference Serving: vLLM + FP8 on a Single L4

A follow-on to the [SFT + DPO fine-tuning study](../README.md#fine-tuning-study-sft--dpo-on-llama-3-8b),
which distilled the GPT-4o analysis agent into a self-hostable Llama-3-8B. That study answered
*"can a small model match the quality?"* This one answers the next question a self-hosting decision
actually depends on: **"how efficiently can we serve it, and what does quantization cost?"**

> **TL;DR** — Re-serving the fine-tuned Llama-3-8B with **vLLM** instead of a naive HuggingFace
> `generate()` loop raised throughput from **11 → 242 tok/s** on the same GPU via continuous
> batching. Adding **FP8** quantization pushed it to **393 tok/s (~1.6×)** and **~3.9× KV-cache
> capacity**, with **no statistically significant quality regression** (pairwise GPT-4o judge, n=30).
> End to end: **~36× the naive baseline on one L4.**

---

## Setup

| | |
|---|---|
| **Model** | Fine-tuned Llama-3-8B — `unsloth/llama-3-8b` base + the SFT LoRA adapter, served via vLLM `--enable-lora` (no merge needed) |
| **GPU** | 1× NVIDIA L4 (24 GB), GCP `g2-standard-8`, `us-central1` |
| **Serving** | Official `vllm/vllm-openai` Docker image, FP16, CUDA graphs **on**, `--max-model-len 4096`, `--gpu-memory-utilization 0.90` |
| **Baseline** | The original `eval_sft.py` path — HuggingFace `transformers` + PEFT, sequential `generate()` (no batching engine) |
| **Workload** | A representative purchasing-analysis prompt; concurrency sweep, 16 requests/level, 128 output tokens |

Everything below is reproducible from `scripts/`: `bench_vllm.py` (concurrency sweep),
`bench_hf_baseline.py` (the naive baseline), and `gen_serving_outputs.py` → `eval_quantization_quality.py`
(the FP16-vs-FP8 quality check).

Serving is containerized (the standard production path), so the deployment is one reproducible command:

```bash
docker run -d --gpus all -p 8000:8000 --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/lora_adapter:/lora_adapter \
  vllm/vllm-openai:latest \
  --model unsloth/llama-3-8b --enable-lora --lora-modules ft=/lora_adapter \
  --max-lora-rank 16 --max-model-len 4096 --gpu-memory-utilization 0.90
```

---

## 1. Continuous batching — throughput scales, latency stays flat

Sweeping concurrent requests against the FP16 server:

| Concurrency | Throughput (tok/s) | p50 latency (s) |
|---:|---:|---:|
| 1  | 16.1  | 7.94 |
| 2  | 31.3  | 8.18 |
| 4  | 62.2  | 8.23 |
| 8  | 122.8 | 8.31 |
| **16** | **242.2** | 8.44 |
| 32 | 240.5 | 8.45 |

Throughput rises **~15× from concurrency 1 → 16 while per-request latency stays ~8 s** — the
defining signature of continuous batching: requests are packed into the GPU together, so aggregate
tokens/sec climbs without slowing any individual request. It saturates at ~16 (compute-bound here),
so concurrency 32 adds nothing.

## 2. vLLM vs the naive baseline

| Serving method | Throughput (tok/s) |
|---|---:|
| HuggingFace `generate()`, sequential (original path) | 11.0 |
| vLLM FP16, single stream (conc 1) | 16.1 |
| vLLM FP16, batched (conc 16) | 242.2 |

Even single-stream, vLLM beats the naive loop (paged attention + CUDA-graph kernels); **batched, it's
~22× the naive baseline on the same hardware.**

## 3. FP8 quantization — more throughput, far more memory headroom, same quality

Re-served with `--quantization fp8` (on-the-fly, Ada FP8 tensor cores), identical setup:

| Concurrency | FP16 tok/s | FP8 tok/s |
|---:|---:|---:|
| 8  | 122.8 | 201.2 |
| 16 | 242.2 | 392.5 |
| 32 | 240.5 | **393.1** |

| Startup capacity | FP16 | FP8 |
|---|---:|---:|
| GPU KV-cache size | 21,024 tokens | **81,360 tokens** |
| Max concurrency @ 4k ctx | 5.13× | **19.86×** |

FP8 halves weight memory (~16 → ~8 GB); the freed VRAM goes to KV cache, so cache capacity grows
**~3.9×** (headroom for ~4× longer contexts / more concurrent long sequences). On L4's FP8 tensor
cores it also **speeds compute → ~1.6× throughput** even on this compute-bound workload.

**Does FP8 cost quality?** Two GPT-4o-judged checks over **all 30 scenarios** agree that it does not:

- **Absolute scoring** (data accuracy + reasoning, /10, same rubric as the SFT eval): **FP16 7.18 → FP8 7.08**
  — a **−0.1** difference, i.e. no meaningful loss.
- **Pairwise** (each pair judged twice with order swapped to cancel position bias): FP8 15 / FP16 9 / 6 ties;
  of 24 decisive items, **not statistically significant** vs a 50/50 split (z≈1.2, p≈0.22).

The honest conclusion is **no quality regression** — FP8 buys ~1.6× throughput and ~3.9× KV cache at no
measurable quality cost. (Quantization can't *add* quality; the tiny differences are sampling/judge noise
at temperature 0.1.) Using all 30 examples is fair because FP16 and FP8 are the *same weights* — so,
unlike the fine-tuning study, no unseen-holdout is required.

---

## What this demonstrates

- **Inference serving, not just consumption** — the production pipeline *calls* GPT-4o via API (OpenAI
  hosts the model); here I *own* the serving stack: GPU, batching, quantization, and the container.
- **Continuous batching & the compute/memory-bound distinction** — measured where throughput saturates
  and why, and when quantization's KV-cache headroom does vs doesn't convert to throughput.
- **Quantization trade-off, quantified** — FP8's throughput/memory gains *and* its quality cost, tested
  with the project's own GPT-4o-as-judge methodology.

## Honest caveats

- **Single-node, single L4.** No multi-GPU / tensor-parallel; TensorRT-LLM and Triton (multi-model
  serving) were out of scope by design.
- **Benchmark prompts are identical**, so prefix caching serves the (small) prompt portion; the
  reported throughput is **generation-bound**, which is not prefix-cached, so the batching comparison
  holds. A production benchmark would use varied prompts.
- **Quality: n=30, single generation per prompt at temp 0.1.** Enough to rule out a gross regression;
  detecting a sub-0.5-point difference would need a few hundred examples + multiple seeds.
