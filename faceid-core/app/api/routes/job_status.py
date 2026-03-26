# app/api/routes/job_status.py

from fastapi import APIRouter
from app.services.verify_result_store import VerifyResultStore

router = APIRouter()


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):

    result = VerifyResultStore.get(job_id)

    if not result:
        return {
            "job_id": job_id,
            "status": "not_found"
        }

    return {
        "job_id": job_id,
        **result
    }