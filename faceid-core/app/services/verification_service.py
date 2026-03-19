# app/services/verification_service.py - Сервис верификации

import asyncio
import logging
from typing import Dict, Any, Optional
import numpy as np
import time

from app.ml.pipeline_runtime import get_pipeline
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.verification_repo import VerificationRepository
from app.services.decision_service import DecisionService
from app.services.liveness_service import LivenessService
from app.services.anti_replay_service import AntiReplayService
from app.services.search_service import SearchService

MATCH_THRESHOLD = 0.7  # minimal similarity threshold для уверенного совпадения
LOW_CONFIDENCE_THRESHOLD = 0.5  # нижний порог для режима low_confidence

logger = logging.getLogger("verification")


class VerificationService:

    def __init__(
        self,
        embedding_repo: EmbeddingRepository,
        verification_repo: VerificationRepository,
    ):
        self.embedding_repo = embedding_repo
        self.verification_repo = verification_repo
        self.pipeline = get_pipeline()  # ок, но...
        print("PIPELINE USED")
        self.search_service = SearchService(embedding_repo)

    async def verify_face(
        self,
        image_bytes: bytes,
        user_id: Optional[str] = None,
        require_liveness: bool = False,
        check_replay: bool = True
    ) -> Dict[str, Any]:

        t_start = time.time()

        # Convert user_id to int or None
        try:
            user_id_int: Optional[int] = int(user_id) if user_id else None
        except (ValueError, TypeError):
            user_id_int = None

        # ANTI-REPLAY CHECK (just a signal, don't block)
        replay_detected = False
        if check_replay:
            replay_detected = not AntiReplayService.check(image_bytes)

        logger.info(
            "verify_started",
            extra={
                "user_id": user_id,
                "require_liveness": require_liveness,
            }
        )

        try:
            result = await self.pipeline.process_async(image_bytes)
        except Exception as e:
            logger.exception("pipeline_failed", extra={"error": str(e)})

            await self.verification_repo.create_log(
                user_id=user_id_int,
                similarity=0.0,
                success=False,
                margin=None,
                liveness_score=None,
                is_genuine=None
            )

            return {
                "status": "processing_failed",
                "liveness_passed": False
            }

        embedding: np.ndarray = result["embedding"]
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0.0:
            return {
                "status": "no_match",
                "similarity": 0.0,
                "liveness_passed": None,
                "replay_detected": replay_detected
            }
        embedding = embedding / norm
        liveness_signals: Dict[str, Any] = result.get("liveness", {})
        pipeline_time = result.get("timings", {})

        logger.info(
            "pipeline_completed",
            extra={
                "timings": pipeline_time
            }
        )

        # Compute composite liveness score
        liveness_result = LivenessService.fuse(liveness_signals)
        liveness_score = liveness_result["score"]
        liveness_risk = liveness_result["risk"]

        # LIVENESS CHECK
        liveness_passed = LivenessService.is_passed(liveness_signals)
        if require_liveness and not liveness_passed:

            await self.verification_repo.create_log(
                user_id=user_id_int,
                similarity=0.0,
                success=False,
                margin=None,
                liveness_score=liveness_score,
                is_genuine=None
            )

            return {
                "status": "spoof_detected",
                "liveness_passed": False,
                "liveness": {
                    "score": liveness_score,
                    "risk": liveness_risk
                }
            }

        # VECTOR SEARCH (top-2 for margin calculation)
        t0 = time.time()
        top_k = await self.search_service.search_top_k(embedding, k=2)
        search_time = (time.time() - t0) * 1000

        top1_similarity = top_k[0]["similarity"] if top_k else 0.0
        top2_similarity = top_k[1]["similarity"] if len(top_k) > 1 else 0.0
        margin = top1_similarity - top2_similarity
        logger.info(
            "search_completed",
            extra={
                "top1_similarity": top1_similarity,
                "top2_similarity": top2_similarity,
                "margin": margin,
                "search_ms": search_time
            }
        )

        if not top_k:
            total_time = (time.time() - t_start) * 1000
            logger.info(
                "verify_result",
                extra={
                    "status": "no_match",
                    "similarity": 0.0,
                    "liveness_passed": liveness_passed,
                    "replay_detected": replay_detected,
                    "total_ms": total_time
                }
            )
            return {
                "status": "no_match",
                "similarity": 0.0,
                "margin": 0.0,
                "timings": {
                    "pipeline": pipeline_time,
                    "search_ms": search_time,
                    "total_ms": total_time
                }
            }

        similarity = top1_similarity
        matched_user_id = top_k[0]["user_id"]

        # Check if matches requested user_id (by external_id)
        is_genuine: bool | None = None
        if user_id:
            from app.db.repositories.user_repo import UserRepository
            user_repo = UserRepository(self.embedding_repo.db)
            expected_user = await user_repo.get_by_external_id(user_id)

            if expected_user:
                expected_user_id = getattr(expected_user, 'id')

                # Get all embeddings for this user and find best score
                embeddings = await self.search_service.search_user_embeddings(expected_user_id)
                if embeddings:
                    embedding_vectors = np.stack(embeddings)
                    centroid = np.mean(embedding_vectors, axis=0)
                    centroid_norm = np.linalg.norm(centroid)
                    if centroid_norm != 0:
                        centroid = centroid / centroid_norm
                        similarity = float(np.dot(embedding, centroid))
                    else:
                        similarity = 0.0
                    matched_user_id = expected_user_id
                    is_genuine = True
                else:
                    is_genuine = False
                    similarity = 0.0

                if not is_genuine:
                    await self.verification_repo.create_log(
                        user_id=expected_user_id,
                        similarity=similarity,
                        success=False,
                        margin=margin,
                        liveness_score=liveness_score,
                        is_genuine=is_genuine
                    )
                    return {
                        "status": "no_match",
                        "liveness_passed": liveness_passed,
                        "liveness": {
                            "score": liveness_score,
                            "risk": liveness_risk
                        }
                    }

        # Decision logic
        best_score = similarity
        status, confidence = DecisionService.decide(
            similarity=similarity,
            margin=margin,
            liveness_score=liveness_score
        )
        if best_score >= MATCH_THRESHOLD:
            status = "match"
            confidence = "high"
        elif best_score >= LOW_CONFIDENCE_THRESHOLD:
            status = "low_confidence"
            confidence = "medium"
        else:
            status = "no_match"
            confidence = "low"

        await self.verification_repo.create_log(
            user_id=top_k[0]["user_id"] if top_k else None,
            similarity=similarity,
            margin=margin,
            liveness_score=liveness_score,
            success=(status == "match")
        )

        total_time = (time.time() - t_start) * 1000

        total_time = (time.time() - t_start) * 1000
        logger.info(
            "verify_result",
            extra={
                "status": status,
                "similarity": best_score,
                "liveness_passed": liveness_passed,
                "replay_detected": replay_detected,
                "total_ms": total_time
            }
        )

        return {
            "status": status,
            "user_id": matched_user_id if status == "match" else None,
            "similarity": float(best_score),
            "margin": float(margin),
            "liveness_passed": liveness_passed,
            "replay_detected": replay_detected
        }
