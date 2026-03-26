# app/services/verify_job_queue.py

import json
import uuid
import time
import redis
from typing import Dict, Any

redis_client = redis.Redis(host="redis", port=6379, db=0)


class VerifyJobQueue:
    QUEUE_NAME = "face_verify_queue"

    @staticmethod
    def enqueue(payload: Dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())

        job = {
            "job_id": job_id,
            "payload": payload,
            "created_at": time.time()
        }

        redis_client.rpush(VerifyJobQueue.QUEUE_NAME, json.dumps(job))

        # сразу помечаем как processing
        redis_client.setex(
            f"job:{job_id}",
            300,
            json.dumps({"status": "processing"})
        )

        return job_id