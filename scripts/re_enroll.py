"""Re-enroll: перегенерация эмбеддингов всех пользователей из папки фото.

Смена ML-пайплайна (SCRFD + аффинное выравнивание) меняет embedding-пространство,
поэтому старые эмбеддинги несовместимы с новыми. Исходные фото после enroll не
хранятся (политика 152-ФЗ), поэтому источник фото — папка на диске в структуре:

    <source>/<external_id>/<image>.jpg|.png|.jpeg|.bmp|.webp

Для каждого external_id:
    1. get_or_create пользователя
    2. удалить старые (несовместимые) эмбеддинги
    3. прогнать фото через новый пайплайн (FacePipelineV2, SCRFD + norm_crop)
    4. записать первый успешный эмбеддинг (create_embedding сам шифрует AES-256)
    5. после всех — пересобрать FAISS-индекс из БД (build_faiss_index)

Запуск (из корня репозитория):
    python scripts/re_enroll.py --source tests/data_extended
    python scripts/re_enroll.py --source /path/to/photos --dry-run
    python scripts/re_enroll.py --source tests/data_extended --limit 3

Env: DATABASE_URL, BIOMETRY_AES_KEY_B64, MODELS_DIR — через settings (.env).
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.db.session import engine, AsyncSessionLocal  # noqa: E402
from app.db.repositories.user_repo import UserRepository  # noqa: E402
from app.db.repositories.embedding_repo import EmbeddingRepository  # noqa: E402
from app.ml.pipeline_v2 import FacePipelineV2  # noqa: E402
from app.services.faiss_loader import build_faiss_index  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_persons(source: Path):
    """Итератор (external_id, [image_paths]) по структуре <source>/<id>/*.jpg."""
    for person_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        images = sorted(
            f for f in person_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        yield person_dir.name, images


async def re_enroll(source: Path, dry_run: bool, limit) -> None:
    print(f"DATABASE_URL: {settings.DATABASE_URL}")
    print(f"source: {source}  dry_run={dry_run}  limit={limit}")
    print(
        f"USE_PIPELINE_V2={settings.USE_PIPELINE_V2} "
        f"POSE_QUALITY_MODE={settings.POSE_QUALITY_MODE} "
        f"FAISS_ENABLED={settings.FAISS_ENABLED}"
    )

    pipeline = FacePipelineV2()
    pipeline._init()

    stats = {
        "ok": 0, "users_ok": 0, "users_fail": 0,
        "spoof": 0, "quality_reject": 0, "error": 0,
    }

    async with AsyncSessionLocal() as session:
        user_repo = UserRepository(session)
        embedding_repo = EmbeddingRepository(session)

        processed = 0
        for external_id, images in iter_persons(source):
            if limit is not None and processed >= limit:
                break
            processed += 1

            if not images:
                print(f"[{external_id}] no images, skip")
                stats["users_fail"] += 1
                continue

            if dry_run:
                user_id = None
            else:
                user = await user_repo.get_or_create(external_id)
                await session.commit()  # закрепить нового пользователя
                user_id = user.id
                deleted = await embedding_repo.delete_by_user_id(user_id)
                if deleted:
                    print(f"[{external_id}] deleted {deleted} old embedding(s)")

            enrolled = False
            for img_path in images:
                try:
                    result = pipeline.process(img_path.read_bytes())
                except Exception as exc:
                    print(f"[{external_id}] {img_path.name} error: {exc}")
                    stats["error"] += 1
                    continue

                status = result.get("status")
                if status != "ok":
                    reason = result.get("quality_reason") or result.get("error_code")
                    print(f"[{external_id}] {img_path.name} {status}: {reason}")
                    if status == "spoof":
                        stats["spoof"] += 1
                    elif status == "quality_reject":
                        stats["quality_reject"] += 1
                    else:
                        stats["error"] += 1
                    continue

                embedding = np.asarray(result["embedding"], dtype=np.float32)
                has_landmarks = bool(result.get("landmarks"))
                if not dry_run:
                    await embedding_repo.create_embedding(user_id, embedding)
                print(
                    f"[{external_id}] {img_path.name} ok "
                    f"norm={float(np.linalg.norm(embedding)):.4f} "
                    f"bbox={result.get('bbox')} landmarks={'yes' if has_landmarks else 'no'}"
                )
                stats["ok"] += 1
                enrolled = True
                break  # один эмбеддинг на пользователя

            if enrolled:
                stats["users_ok"] += 1
            else:
                stats["users_fail"] += 1
                print(f"[{external_id}] WARNING: no usable image")

        if not dry_run and settings.FAISS_ENABLED:
            print("rebuilding FAISS index from DB ...")
            index = await build_faiss_index(embedding_repo)
            print(f"FAISS rebuilt: ntotal={index.index.ntotal}")

    await engine.dispose()
    print("done stats=", stats)


def main():
    parser = argparse.ArgumentParser(
        description="Re-enroll embeddings from a photo folder (<id>/*.jpg)"
    )
    parser.add_argument(
        "--source", default="tests/data_extended",
        help="folder with <external_id>/*.jpg subfolders",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="process photos without writing to DB",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="max number of persons to process",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.is_dir():
        print(f"source not found: {source}")
        sys.exit(1)

    asyncio.run(re_enroll(source, args.dry_run, args.limit))


if __name__ == "__main__":
    main()