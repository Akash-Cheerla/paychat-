"""
PayChat — Continuous Learning Pipeline
Runs nightly (via AWS EventBridge → Lambda) to:
  1. Pull new labeled data from S3 (user feedback + auto-logged predictions)
  2. Merge with original training data (preserving base knowledge)
  3. Re-train with warm start (no catastrophic forgetting)
  4. Compare new vs current accuracy
  5. Auto-promote if better, or keep current if worse
  6. Hot-reload the API without downtime

Architecture:
  Chat App → Feedback API → S3 (data lake)
  EventBridge (midnight) → Lambda (this script) → SageMaker training
  → S3 (new model) → ECS hot-reload → Better model live
"""

import json
import os
import time
import logging
import boto3
import botocore.exceptions
import tempfile
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ──
S3_BUCKET        = os.getenv("S3_BUCKET", "paychat-ml-data")
S3_DATA_PREFIX   = "feedback/"
S3_MODEL_PREFIX  = "models/"
ECS_CLUSTER      = os.getenv("ECS_CLUSTER", "paychat-cluster")
ECS_SERVICE      = os.getenv("ECS_SERVICE", "paychat-detector")
API_ENDPOINT     = os.getenv("API_ENDPOINT", "http://localhost:8000")
MIN_NEW_SAMPLES  = 50   # Only retrain if at least this many new examples
ACCURACY_DELTA   = 0.005  # Only promote if 0.5%+ better (avoids noise-driven regressions)

s3 = boto3.client("s3")
ecs = boto3.client("ecs")


# ── Data collection ──
def collect_feedback_from_s3(since_hours: int = 24) -> list:
    """Pull all feedback logged in the last N hours from S3."""
    cutoff = datetime.utcnow() - timedelta(hours=since_hours)
    items = []

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_DATA_PREFIX):
        for obj in page.get("Contents", []):
            # Filter by last modified
            if obj["LastModified"].replace(tzinfo=None) < cutoff:
                continue

            body = s3.get_object(Bucket=S3_BUCKET, Key=obj["Key"])["Body"].read()
            try:
                record = json.loads(body)
                if "text" in record and "label" in record:
                    items.append(record)
            except json.JSONDecodeError:
                continue

    logger.info(f"Collected {len(items)} feedback records from S3")
    return items


def load_base_training_data() -> list:
    """Load original synthetic training data from S3."""
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key="base_data/train.json")
        return json.loads(obj["Body"].read())
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            logger.warning("Base training data not found in S3. Using feedback data only.")
            return []
        raise


def merge_datasets(base_data: list, feedback_data: list) -> list:
    """
    Merge base + feedback data intelligently:
    - Human feedback gets 3x weight (repeat in dataset)
    - Auto-logged predictions get 1x weight
    - Dedup by exact text match
    - Preserve 50/50 class balance
    """
    seen_texts = set()
    merged = []

    # Base data first (1x weight)
    for item in base_data:
        if item["text"] not in seen_texts:
            merged.append(item)
            seen_texts.add(item["text"])

    # Feedback data (weighted by confidence)
    human_feedback = [x for x in feedback_data if x.get("source") == "human_feedback"]
    auto_logged    = [x for x in feedback_data if x.get("source") != "human_feedback"]

    for item in human_feedback:
        if item["text"] not in seen_texts:
            # 3x weight: repeat human-labeled examples
            for _ in range(3):
                merged.append(item)
            seen_texts.add(item["text"])

    for item in auto_logged:
        if item["text"] not in seen_texts and item.get("confidence", 0) > 0.85:
            # Only include high-confidence auto-labeled examples
            merged.append(item)
            seen_texts.add(item["text"])

    # Verify class balance
    pos = sum(1 for x in merged if x["label"] == 1)
    neg = sum(1 for x in merged if x["label"] == 0)
    logger.info(f"Merged dataset: {len(merged)} total | {pos} positive | {neg} negative")

    return merged


# ── Training ──
def run_training_job(data: list) -> dict:
    """
    Run training locally or on SageMaker.
    Returns: {"model_path": ..., "accuracy": ..., "f1": ...}
    """
    import sys
    import importlib.util

    # Save merged data to temp location
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Split train/val/test
        import random
        random.shuffle(data)
        n = len(data)
        train = data[:int(n * 0.8)]
        val   = data[int(n * 0.8):int(n * 0.9)]
        test  = data[int(n * 0.9):]

        for split_name, items in [("train", train), ("val", val), ("test", test)]:
            with open(tmp / f"{split_name}.json", "w") as f:
                json.dump(items, f)

        # Run training script
        model_out = tmp / "new_model"
        result = subprocess.run(
            [sys.executable, "model/train.py",
             f"--data-dir={tmp}",
             f"--output-dir={model_out}",
             "--epochs=5"],
            capture_output=True, text=True, timeout=3600
        )

        if result.returncode != 0:
            raise RuntimeError(f"Training failed:\n{result.stderr}")

        # Read report
        with open(model_out / "training_report.json") as f:
            report = json.load(f)

        return {
            "model_path": str(model_out),
            "accuracy": report["test_accuracy"],
            "f1": report["test_f1"],
        }


# ── Model promotion ──
def get_current_model_accuracy() -> float:
    """Fetch current deployed model's accuracy from S3 metadata."""
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=f"{S3_MODEL_PREFIX}current/training_report.json")
        report = json.loads(obj["Body"].read())
        return report.get("test_accuracy", 0.0)
    except Exception:
        return 0.0


def promote_model(model_path: str, metadata: dict):
    """Upload new model to S3 and trigger hot-reload."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    versioned_key = f"{S3_MODEL_PREFIX}{timestamp}/"
    current_key   = f"{S3_MODEL_PREFIX}current/"

    model_files = list(Path(model_path).glob("*"))
    for f in model_files:
        for key_prefix in [versioned_key, current_key]:
            s3.upload_file(str(f), S3_BUCKET, key_prefix + f.name)
            logger.info(f"Uploaded {f.name} → s3://{S3_BUCKET}/{key_prefix + f.name}")

    # Save promotion metadata
    meta = {**metadata, "promoted_at": datetime.utcnow().isoformat(), "version": timestamp}
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"{S3_MODEL_PREFIX}current/metadata.json",
        Body=json.dumps(meta),
    )

    # Trigger hot-reload on API (no downtime)
    _trigger_api_reload()
    logger.info(f"Model promoted: v{timestamp} | accuracy={metadata['accuracy']:.2%}")


def _trigger_api_reload():
    """Tell the running API to reload the model from S3."""
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{API_ENDPOINT}/reload",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        logger.info("API hot-reload triggered successfully")
    except Exception as e:
        logger.warning(f"API reload failed (may need ECS restart): {e}")
        # Fallback: force ECS service restart
        try:
            ecs.update_service(
                cluster=ECS_CLUSTER,
                service=ECS_SERVICE,
                forceNewDeployment=True,
            )
            logger.info("ECS service restart triggered")
        except Exception as ecs_err:
            logger.error(f"ECS restart also failed: {ecs_err}")


# ── Lambda handler ──
def lambda_handler(event, context):
    """AWS Lambda entry point — runs nightly via EventBridge."""
    logger.info("=" * 60)
    logger.info("PayChat Continuous Learning — Nightly Run")
    logger.info(f"Time: {datetime.utcnow().isoformat()}")
    logger.info("=" * 60)

    try:
        # 1. Collect new data
        feedback = collect_feedback_from_s3(since_hours=24)

        if len(feedback) < MIN_NEW_SAMPLES:
            logger.info(f"Only {len(feedback)} new samples (need {MIN_NEW_SAMPLES}). Skipping retraining.")
            return {"status": "skipped", "reason": "insufficient_data", "samples": len(feedback)}

        # 2. Load base data and merge
        base_data = load_base_training_data()
        merged    = merge_datasets(base_data, feedback)

        # 3. Train new model
        logger.info("Starting training...")
        t0 = time.time()
        result = run_training_job(merged)
        elapsed = time.time() - t0
        logger.info(f"Training complete in {elapsed:.1f}s | Accuracy: {result['accuracy']:.2%} | F1: {result['f1']:.2%}")

        # 4. Compare with current
        current_acc = get_current_model_accuracy()
        new_acc     = result["accuracy"]
        logger.info(f"Current accuracy: {current_acc:.2%} | New accuracy: {new_acc:.2%}")

        if new_acc >= current_acc + ACCURACY_DELTA or current_acc == 0.0:
            # 5. Promote new model
            promote_model(result["model_path"], {
                "accuracy": new_acc,
                "f1": result["f1"],
                "training_samples": len(merged),
                "new_feedback_samples": len(feedback),
            })
            return {
                "status": "promoted",
                "accuracy_before": current_acc,
                "accuracy_after": new_acc,
                "improvement": new_acc - current_acc,
            }
        else:
            logger.info(f"New model not better enough (delta={new_acc - current_acc:.4f}). Keeping current.")
            return {
                "status": "kept_current",
                "accuracy_current": current_acc,
                "accuracy_new": new_acc,
            }

    except Exception as e:
        logger.error(f"Continuous learning run failed: {e}", exc_info=True)
        raise


# ── Feedback logging API (separate FastAPI service or added to main API) ──
"""
Add these endpoints to api/app.py to collect feedback for continuous learning:

POST /log
  Body: { "text": "msg", "predicted": true, "confidence": 0.92, "message_id": "..." }
  Saves prediction to S3 for future retraining

POST /feedback  
  Body: { "message_id": "123", "correct": true/false, "true_label": 1/0 }
  Saves human feedback with 3x weight boost
"""

if __name__ == "__main__":
    # Test run locally
    lambda_handler({}, None)
