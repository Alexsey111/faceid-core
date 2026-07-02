import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport

from app.core.config import settings
from app.main import app
from app.services.search_service import SearchService

# Path to test image
TEST_IMAGE = Path(__file__).parent / "images" / "person1.jpg"

# Fallback to insightface test images (t1.jpg has 6 faces)
if not TEST_IMAGE.exists():
    TEST_IMAGE = Path("D:/python projects/faceid-core/venv/insightface/data/images/t1.jpg")


@pytest.mark.asyncio
async def test_enroll_and_verify():
    settings.FAISS_ENABLED = False
    settings.REDIS_ENABLED = False
    SearchService._faiss_index = None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:

        with open(TEST_IMAGE, "rb") as f:

            enroll = await ac.post(
                "/upload?user_id=1",
                files={"file": (TEST_IMAGE.name, f, "image/jpeg")}
            )

        print(f"Enroll status: {enroll.status_code}")
        print(f"Enroll body: {enroll.text}")
        assert enroll.status_code == 200

        with open(TEST_IMAGE, "rb") as f:

            verify = await ac.post(
                "/api/v1/verify",
                files={"file": (TEST_IMAGE.name, f, "image/jpeg")}
            )

        data = verify.json()
        print(f"\nDEBUG verify response: {data}")

        assert verify.status_code == 200
        assert data["status"] == "match"
        assert data["similarity"] > 0.5
