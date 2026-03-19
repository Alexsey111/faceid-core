# faceid-core\app\ml\pipeline_runtime.py

from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import lru_cache

from app.ml.pipeline import FacePipeline

_executor = ThreadPoolExecutor(max_workers=2)

# ограничение одновременных inference
_semaphore = asyncio.Semaphore(2)


@lru_cache(maxsize=1)
def get_pipeline() -> FacePipeline:
    print("PIPELINE INIT")
    pipeline = FacePipeline()
    pipeline._init()
    return pipeline
