#!/usr/bin/env python3
"""
worker.py
Consumes RL evaluation jobs from RabbitMQ, runs PPO evaluation via evaluate.py,
stores metrics to MinIO, and publishes a completion message back to RabbitMQ.

Expected job message (JSON) example:
{
  "run_id": "2026-01-20T22-55-00Z",
  "episodes": 3,
  "model_path": "ppo_pioneer_multiinput2.zip"
}
"""

import json
import os
import time
import traceback
from io import BytesIO

import pika
from minio import Minio

from evaluate import evaluate_model  # <- your file


# -------------------------
# Config (env vars)
# -------------------------
# RabbitMQ
RABBIT_HOST = os.getenv("RABBIT_HOST", "rabbitmq")
RABBIT_PORT = int(os.getenv("RABBIT_PORT", "5672"))
RABBIT_USER = os.getenv("RABBIT_USER", "pavlos")
RABBIT_PASS = os.getenv("RABBIT_PASS", "pavlos21")
RABBIT_VHOST = os.getenv("RABBIT_VHOST", "rlVhost")

# Exchange/queue routing
RABBIT_EXCHANGE = os.getenv("RABBIT_EXCHANGE", "rlJobs")  # your topic exchange name
RABBIT_EXCHANGE_TYPE = os.getenv("RABBIT_EXCHANGE_TYPE", "topic")

JOB_QUEUE = os.getenv("RABBIT_JOB_QUEUE", "rlJobs")       # queue you consume from
JOB_BINDING_KEY = os.getenv("RABBIT_JOB_BINDING_KEY", "rl.jobs")  # bind queue to exchange
# (during debugging you can set JOB_BINDING_KEY=rl.#)

RESULT_ROUTING_KEY = os.getenv("RABBIT_RESULT_ROUTING_KEY", "rl.done")

# MinIO
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")  # e.g. "minio:9000" if in docker network
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rlbucket2")  # must be lowercase
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# Evaluation defaults
DEFAULT_EPISODES = int(os.getenv("DEFAULT_EPISODES", "3"))
DEFAULT_MODEL_PATH = os.getenv("DEFAULT_MODEL_PATH", "ppo_pioneer_multiinput2.zip")
SLOWDOWN_SEC = float(os.getenv("SLOWDOWN_SEC", "0.0"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "2000"))

def connect_with_retry(params, attempts=30, delay=2):
    for i in range(attempts):
        try:
            return pika.BlockingConnection(params)
        except pika.exceptions.AMQPConnectionError:
            print(f"RabbitMQ not ready / connection failed. Retry {i+1}/{attempts}...")
            time.sleep(delay)
    raise RuntimeError("RabbitMQ never became ready")
# -------------------------
# Helpers
# -------------------------
def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def ensure_bucket(mc: Minio, bucket: str) -> None:
    if not mc.bucket_exists(bucket):
        mc.make_bucket(bucket)

def put_json(mc: Minio, bucket: str, key: str, data: dict) -> None:
    raw = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    mc.put_object(
        bucket_name=bucket,
        object_name=key,
        data=BytesIO(raw),
        length=len(raw),
        content_type="application/json",
    )

def safe_run_id(provided: str | None) -> str:
    if provided and isinstance(provided, str) and provided.strip():
        return provided.strip()
    # make a filesystem-safe id
    return time.strftime("run-%Y%m%d-%H%M%S", time.gmtime())


# -------------------------
# RabbitMQ callback
# -------------------------
def on_message(ch, method, properties, body: bytes):
    """
    Process one job message.
    ACK only after:
      - evaluation succeeded
      - metrics uploaded to MinIO
      - result message published
    """
    delivery_tag = method.delivery_tag
    try:
        job = json.loads(body.decode("utf-8"))
    except Exception:
        print("❌ Invalid JSON message, rejecting (no requeue). Body:", body[:500])
        ch.basic_nack(delivery_tag=delivery_tag, requeue=False)
        return

    run_id = safe_run_id(job.get("run_id"))
    episodes = int(job.get("episodes", DEFAULT_EPISODES))
    model_path = job.get("model_path") or job.get("model_key") or DEFAULT_MODEL_PATH

    print(f"📥 Job received run_id={run_id} episodes={episodes} model={model_path}")

    # Connect MinIO (create per-message to keep it simple/robust)
    mc = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )
    ensure_bucket(mc, MINIO_BUCKET)

    t0 = time.time()
    try:
        # ---- Run evaluation (your existing logic, now wrapped in evaluate.py) ----
        results = evaluate_model(
            model_path=model_path,
            episodes=episodes,
            slowdown_sec=SLOWDOWN_SEC,
            max_steps=MAX_STEPS,
        )
        # ensure results is a dict
        if not isinstance(results, dict):
            raise RuntimeError("evaluate_model() did not return a dict.")

        duration_sec = time.time() - t0

        # enrich results
        results["run_id"] = run_id
        results["episodes"] = episodes
        results["model_path"] = model_path
        results["job_received_utc"] = job.get("job_received_utc", None)
        results["job_finished_utc"] = utc_now_iso()
        results["job_duration_sec"] = duration_sec

        # ---- Store to MinIO ----
        metrics_key = f"runs/{run_id}/metrics.json"
        put_json(mc, MINIO_BUCKET, metrics_key, results)
        print(f"✅ Uploaded metrics to MinIO: {MINIO_BUCKET}/{metrics_key}")

        # ---- Publish completion event for Node-RED ----
        # Keep message small (Node-RED can fetch full JSON from MinIO if needed)
        done_msg = {
            "run_id": run_id,
            "status": "success",
            "episodes": episodes,
            "model_path": model_path,
            "minio_bucket": MINIO_BUCKET,
            "minio_metrics_key": metrics_key,
            # Include a couple of headline metrics if they exist:
            "success_rate": results.get("success_rate"),
            "success_count": results.get("success_count"),
        }
        ch.basic_publish(
            exchange=RABBIT_EXCHANGE,
            routing_key=RESULT_ROUTING_KEY,
            body=json.dumps(done_msg).encode("utf-8"),
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2,  # persistent
            ),
        )
        print(f"📤 Published result event routing_key={RESULT_ROUTING_KEY}")

        # ---- ACK original job ----
        ch.basic_ack(delivery_tag=delivery_tag)
        print(f"✅ ACK job run_id={run_id}")

    except Exception as e:
        err_text = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print("❌ Job failed:", e)
        print(err_text)

        # Store error to MinIO (best-effort)
        try:
            error_key = f"runs/{run_id}/error.json"
            put_json(mc, MINIO_BUCKET, error_key, {
                "run_id": run_id,
                "status": "error",
                "error": str(e),
                "traceback": err_text,
                "job_finished_utc": utc_now_iso(),
            })
            print(f"⚠️ Uploaded error to MinIO: {MINIO_BUCKET}/{error_key}")
        except Exception as minio_err:
            print("⚠️ Also failed to write error to MinIO:", minio_err)

        # Requeue behavior:
        # - During development: requeue=True can cause infinite loops if the error is permanent.
        # - Safer default: requeue=False (then you can use a DLQ later).
        requeue = os.getenv("REQUEUE_ON_FAIL", "false").lower() == "true"
        ch.basic_nack(delivery_tag=delivery_tag, requeue=requeue)


# -------------------------
# Main
# -------------------------
def main():
    creds = pika.PlainCredentials(RABBIT_USER, RABBIT_PASS)
    params = pika.ConnectionParameters(
    host=RABBIT_HOST,
    virtual_host=RABBIT_VHOST,
    credentials=creds,
    heartbeat=300,
    blocked_connection_timeout=300,)
    

    conn = connect_with_retry(params)
    ch = conn.channel()

    # Exchange + queue + binding (topic routing)
    ch.exchange_declare(
        exchange=RABBIT_EXCHANGE,
        exchange_type=RABBIT_EXCHANGE_TYPE,
        durable=True,
    )
    ch.queue_declare(queue=JOB_QUEUE, durable=True)
    ch.queue_bind(queue=JOB_QUEUE, exchange=RABBIT_EXCHANGE, routing_key=JOB_BINDING_KEY)

    # Fair dispatch: one job at a time per worker
    ch.basic_qos(prefetch_count=1)

    ch.basic_consume(queue=JOB_QUEUE, on_message_callback=on_message)

    print(
        "Worker ready:\n"
        f"  RabbitMQ: {RABBIT_HOST}:{RABBIT_PORT} vhost='{RABBIT_VHOST}'\n"
        f"  Exchange: {RABBIT_EXCHANGE} (type={RABBIT_EXCHANGE_TYPE})\n"
        f"  Job queue: {JOB_QUEUE} (binding key='{JOB_BINDING_KEY}')\n"
        f"  Results routing key: {RESULT_ROUTING_KEY}\n"
        f"  MinIO: {MINIO_ENDPOINT} bucket='{MINIO_BUCKET}' secure={MINIO_SECURE}\n"
    )
    print("Waiting for jobs... (Ctrl+C to stop)",flush=True)
    try:
        print(f"✅ CONSUMING on queue={JOB_QUEUE}", flush=True)
        ch.start_consuming()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        try:
            ch.stop_consuming()
        except Exception:
            pass
        conn.close()


if __name__ == "__main__":
    main()
