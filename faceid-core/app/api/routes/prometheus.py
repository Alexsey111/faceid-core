# faceid-core\app\api\routes\prometheus.py

from fastapi import APIRouter
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, multiprocess
from fastapi.responses import Response

router = APIRouter()


def _prometheus_metrics_response() -> Response:
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


@router.get("/metrics")
def metrics():
    return _prometheus_metrics_response()
