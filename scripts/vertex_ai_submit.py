"""
Submit SFT training as a Vertex AI Custom Training Job.
Uses Cloud Build to build the Docker image remotely (no local Docker needed).

Prerequisites:
    - gcloud CLI authenticated (`gcloud auth login`)
    - Vertex AI API enabled
    - Artifact Registry API enabled
    - Cloud Build API enabled

Usage:
    python scripts/vertex_ai_submit.py
"""

import os
import subprocess
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # Load .env file

# ── Configuration ──────────────────────────────────────────────────────────────

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "purchasing-automation")
REGION = os.environ.get("GCP_REGION", "us-central1")
REPO_NAME = "purchasing-automation"
IMAGE_NAME = "sft-trainer"
IMAGE_TAG = datetime.now().strftime("%Y%m%d-%H%M")
FULL_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/{IMAGE_NAME}:{IMAGE_TAG}"

# GPU options for Vertex AI:
#   NVIDIA_TESLA_T4   — cheapest, 16GB VRAM, ~$1.40/hr (sufficient for 8B QLoRA)
#   NVIDIA_TESLA_V100 — 16GB VRAM, ~$2.50/hr
#   NVIDIA_TESLA_A100 — 40GB VRAM, ~$3.70/hr (fastest)
MACHINE_TYPE = "n1-standard-8"
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
ACCELERATOR_COUNT = 1

JOB_DISPLAY_NAME = f"sft-llama3-purchasing-{IMAGE_TAG}"


def run_cmd(cmd: list[str], description: str):
    """Run a shell command with error handling."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)
    return result


# ── Step 1: Ensure Artifact Registry repo exists ──────────────────────────────

def ensure_artifact_registry():
    """Create Artifact Registry Docker repo if it doesn't exist."""
    run_cmd([
        "gcloud", "artifacts", "repositories", "describe",
        REPO_NAME,
        f"--project={PROJECT_ID}",
        f"--location={REGION}",
    ], f"Checking if Artifact Registry repo '{REPO_NAME}' exists...")

    print(f"Repo '{REPO_NAME}' exists.")


def create_artifact_registry():
    """Create the Artifact Registry repo."""
    run_cmd([
        "gcloud", "artifacts", "repositories", "create",
        REPO_NAME,
        f"--project={PROJECT_ID}",
        f"--location={REGION}",
        "--repository-format=docker",
        f"--description=Docker images for {REPO_NAME}",
    ], f"Creating Artifact Registry repo '{REPO_NAME}'...")


# ── Step 2: Build image with Cloud Build ──────────────────────────────────────

def build_with_cloud_build():
    """Build Docker image using Cloud Build (no local Docker needed)."""
    run_cmd([
        "gcloud", "builds", "submit",
        f"--project={PROJECT_ID}",
        f"--region={REGION}",
        "--config=cloudbuild.yaml",
        f"--substitutions=_IMAGE_URI={FULL_IMAGE_URI}",
        "--timeout=1800",
        f"--gcs-source-staging-dir=gs://{PROJECT_ID}_cloudbuild/source",
        ".",
    ], f"Building Docker image via Cloud Build: {FULL_IMAGE_URI}")


# ── Step 3: Submit Vertex AI Custom Training Job ──────────────────────────────

def submit_training_job():
    """Submit the training job to Vertex AI."""
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    wandb_entity = os.environ.get("WANDB_ENTITY", "msm1640-")

    env_vars = (
        f"WANDB_API_KEY={wandb_key},"
        f"WANDB_ENTITY={wandb_entity},"
        f"WANDB_PROJECT=purchasing-automation-sft"
    )

    run_cmd([
        "gcloud", "ai", "custom-jobs", "create",
        f"--project={PROJECT_ID}",
        f"--region={REGION}",
        f"--display-name={JOB_DISPLAY_NAME}",
        f"--worker-pool-spec="
        f"machine-type={MACHINE_TYPE},"
        f"replica-count=1,"
        f"accelerator-type={ACCELERATOR_TYPE},"
        f"accelerator-count={ACCELERATOR_COUNT},"
        f"container-image-uri={FULL_IMAGE_URI}",
        f"--env-vars={env_vars}",
    ], f"Submitting Vertex AI Training Job: {JOB_DISPLAY_NAME}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Vertex AI SFT Training Pipeline")
    print(f"  Project: {PROJECT_ID}")
    print(f"  GPU: {ACCELERATOR_COUNT}x {ACCELERATOR_TYPE}")
    print("=" * 60)

    # Step 1: Ensure repo exists
    try:
        ensure_artifact_registry()
    except SystemExit:
        print("Repo not found. Creating...")
        create_artifact_registry()

    # Step 2: Build image via Cloud Build
    build_with_cloud_build()

    # Step 3: Submit training job
    submit_training_job()

    print("\n" + "=" * 60)
    print("  Job submitted! Monitor at:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    print(f"  W&B Dashboard: https://wandb.ai/msm1640-/purchasing-automation-sft")
    print("=" * 60)


if __name__ == "__main__":
    main()
