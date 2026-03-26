# app/api/routes/verify_async.py

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.verify_job_queue import VerifyJobQueue

router = APIRouter()


class VerifyAsyncRequest(BaseModel):
    image_b64: str
    user_id: str | None = None
    require_liveness: bool = False


@router.post("/verify_async")
async def verify_async(request: VerifyAsyncRequest):

    job_id = VerifyJobQueue.enqueue({
        "image_b64": request.image_b64,
        "user_id": request.user_id,
        "require_liveness": request.require_liveness
    })

    return {
        "job_id": job_id,
        "status": "queued"
    }