import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport

from app.main import app

# Path to test images (fallback to insightface)
TEST_IMAGE_1 = Path(__file__).parent / "images" / "person1.jpg"
TEST_IMAGE_2 = Path(__file__).parent / "images" / "person2.jpg"

# Fallback to insightface test images
if not TEST_IMAGE_1.exists():
    TEST_IMAGE_1 = Path("D:/python projects/faceid-core/venv/insightface/data/images/t1.jpg")
if not TEST_IMAGE_2.exists():
    TEST_IMAGE_2 = Path("D:/python projects/faceid-core/venv/insightface/data/images/t1.jpg")


@pytest.mark.asyncio
async def test_multi_embedding():
    """Test enrolling multiple photos for same user and verifying both match."""

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:

        # Enroll first photo
        with open(TEST_IMAGE_1, "rb") as f:
            enroll1 = await ac.post(
                "/upload?user_id=1",
                files={"file": (TEST_IMAGE_1.name, f, "image/jpeg")}
            )

        assert enroll1.status_code == 200
        print(f"Enroll 1 status: {enroll1.status_code}")

        # Enroll second photo (same user)
        with open(TEST_IMAGE_2, "rb") as f:
            enroll2 = await ac.post(
                "/upload?user_id=1",
                files={"file": (TEST_IMAGE_2.name, f, "image/jpeg")}
            )

        assert enroll2.status_code == 200
        print(f"Enroll 2 status: {enroll2.status_code}")

        # Verify first photo
        with open(TEST_IMAGE_1, "rb") as f:
            verify1 = await ac.post(
                "/verify?user_id=1",
                files={"file": (TEST_IMAGE_1.name, f, "image/jpeg")}
            )

        data1 = verify1.json()
        print(f"\nDEBUG verify1 response: {data1}")
        assert verify1.status_code == 200

        # Verify second photo
        with open(TEST_IMAGE_2, "rb") as f:
            verify2 = await ac.post(
                "/verify?user_id=1",
                files={"file": (TEST_IMAGE_2.name, f, "image/jpeg")}
            )

        data2 = verify2.json()
        print(f"\nDEBUG verify2 response: {data2}")
        assert verify2.status_code == 200

        # Both should match
        assert data1["status"] in ["match", "low_confidence"], f"Expected match, got {data1['status']}"
        assert data2["status"] in ["match", "low_confidence"], f"Expected match, got {data2['status']}"
