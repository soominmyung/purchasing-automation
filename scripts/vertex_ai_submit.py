"""
Submit SFT training as a Vertex AI Custom Training Job.
This script packages the training code into a Docker container and
submits it to Vertex AI for execution on GPU instances.

Prerequisites:
    - gcloud CLI authenticated
    - Docker installed
    - Artifact Registry repository created
    - W&B API key set in environment

Usage:
    python scripts/vertex_ai_submit.py
"""

import os
import subprocess
import sys
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
REGION = os.environ.get("GCP_REGION", "us-central1")
REPO_NAME = "purchasing-automation"
IMAGE_NAME = "sft-trainer"
IMAGE_TAG = datetime.now().strftime("%Y%m%d-%H%M")
FULL_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/{IMAGE_NAME}:{IMAGE_TAG}"

# Machine type options for Vertex AI Custom Training
# - n1-standard-8 + 1x NVIDIA T4 (16GB VRAM) — cheapest GPU option (~$1.40/hr)
# - n1-standard-8 + 1x NVIDIA V100 (16GB VRAM) — faster (~$2.50/hr)
# - a2-highgpu-1g + 1x NVIDIA A100 (40GB VRAM) — fastest (~$3.70/hr)
MACHINE_TYPE = "n1-standard-8"
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
ACCELERATOR_COUNT = 1

JOB_DISPLAY_NAME = f"sft-llama3-purchasing-{IMAGE_TAG}"


# ── Docker Build & Push ────────────────────────────────────────────────────────

def build_and_push_image():
    """Build the training Docker image and push to Artifact Registry."""
    dockerfile_path = "Dockerfile.train"

    print(f"Building Docker image: {FULL_IMAGE_URI}")
    subprocess.run(
        ["docker", "build", "-f", dockerfile_path, "-t", FULL_IMAGE_URI, "."],
        check=True,
    )

    print(f"Pushing image to Artifact Registry...")
    subprocess.run(
        ["docker", "push", FULL_IMAGE_URI],
        check=True,
    )
    print("Image pushed successfully.")


# ── Vertex AI Job Submission ───────────────────────────────────────────────────

def submit_training_job():
    """Submit a Vertex AI Custom Training Job."""
    wandb_key = os.environ.get("WANDB_API_KEY", "")

    cmd = [
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
    ]

    # Pass environment variables
    if wandb_key:
        cmd.append(f"--env-vars=WANDB_API_KEY={wandb_key}")

    print(f"Submitting Vertex AI Custom Training Job: {JOB_DISPLAY_NAME}")
    print(f"  Machine: {MACHINE_TYPE} + {ACCELERATOR_COUNT}x {ACCELERATOR_TYPE}")
    print(f"  Image: {FULL_IMAGE_URI}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("Job submitted successfully!")
        print(result.stdout)
    else:
        print("Job submission failed:")
        print(result.stderr)
        sys.exit(1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not PROJECT_ID:
        print("Error: Set GCP_PROJECT_ID environment variable.")
        sys.exit(1)

    print("=" * 60)
    print("Vertex AI Custom Training Job Submission")
    print("=" * 60)
    print()

    # Step 1: Build and push Docker image
    build_and_push_image()

    # Step 2: Submit training job
    submit_training_job()

    print()
    print("Monitor your job at:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")


if __name__ == "__main__":
    main()
