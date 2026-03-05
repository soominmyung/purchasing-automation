"""
Merge LoRA adapter into base model and export as GGUF Q4_K_M.

Downloads LoRA adapter from GCS, merges with base Llama-3-8B via Unsloth,
and saves the quantized GGUF file back to GCS.

Usage (Vertex AI Custom Training Job):
    Override CMD with: ["python", "scripts/merge_lora.py"]
"""

import os
from pathlib import Path

from google.cloud import storage as gcs

# ── Config ─────────────────────────────────────────────────────────────────────

GCS_ADAPTER_URI = os.environ.get(
    "GCS_ADAPTER_URI",
    "gs://purchasing-automation-models/sft-runs/lora_adapter",
)
GCS_OUTPUT_URI = os.environ.get(
    "GCS_OUTPUT_URI",
    "gs://purchasing-automation-models/gguf",
)

LOCAL_ADAPTER_PATH = "/app/lora_adapter"
LOCAL_OUTPUT_PATH  = "/app/gguf_output"


# ── GCS helpers ────────────────────────────────────────────────────────────────

def download_gcs_prefix(gcs_uri: str, local_dir: str) -> None:
    uri_parts   = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = uri_parts[0]
    prefix      = uri_parts[1] if len(uri_parts) > 1 else ""

    client  = gcs.Client()
    bucket  = client.bucket(bucket_name)
    blobs   = list(bucket.list_blobs(prefix=prefix))

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    for blob in blobs:
        rel = blob.name[len(prefix):].lstrip("/")
        if not rel:
            continue
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
        print(f"  ↓ {blob.name}")

    print(f"Downloaded {len(blobs)} files → {local_dir}")


def upload_dir_to_gcs(local_dir: str, gcs_uri: str) -> None:
    uri_parts   = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = uri_parts[0]
    prefix      = uri_parts[1] if len(uri_parts) > 1 else ""

    client  = gcs.Client()
    bucket  = client.bucket(bucket_name)
    count   = 0

    for f in Path(local_dir).rglob("*"):
        if f.is_file():
            blob_path = f"{prefix}/{f.relative_to(local_dir)}" if prefix else str(f.relative_to(local_dir))
            bucket.blob(blob_path).upload_from_filename(str(f))
            print(f"  ↑ {f.name} → gs://{bucket_name}/{blob_path}")
            count += 1

    print(f"Uploaded {count} files → {gcs_uri}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Merge LoRA → GGUF Q4_K_M")
    print("=" * 60)

    # 1. Download LoRA adapter
    print(f"\n[1/3] Downloading LoRA adapter from {GCS_ADAPTER_URI} ...")
    download_gcs_prefix(GCS_ADAPTER_URI, LOCAL_ADAPTER_PATH)

    # 2. Load base model + adapter with Unsloth, then save as GGUF
    print(f"\n[2/3] Loading model and exporting GGUF Q4_K_M ...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LOCAL_ADAPTER_PATH,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    Path(LOCAL_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(
        LOCAL_OUTPUT_PATH,
        tokenizer,
        quantization_method="q4_k_m",
    )
    print(f"GGUF saved to {LOCAL_OUTPUT_PATH}")

    # 3. Upload to GCS
    print(f"\n[3/3] Uploading to {GCS_OUTPUT_URI} ...")
    upload_dir_to_gcs(LOCAL_OUTPUT_PATH, GCS_OUTPUT_URI)

    print("\nDone! GGUF model ready at:", GCS_OUTPUT_URI)


if __name__ == "__main__":
    main()
