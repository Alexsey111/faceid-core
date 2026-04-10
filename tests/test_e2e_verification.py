import base64
import pytest
from httpx import AsyncClient, ASGITransport
from pathlib import Path

from app.main import app

TEST_IMAGE_1 = Path("tests/data/person1.jpg")
TEST_IMAGE_2 = Path("tests/data/person1_2.jpg")
TEST_IMAGE_OTHER = Path("tests/data/person2.jpg")


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


@pytest.mark.asyncio
async def test_e2e_same_person():

    img1 = encode_image(TEST_IMAGE_1)
    img2 = encode_image(TEST_IMAGE_2)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/upload",
            params={"user_id": "123"},
            files={"file": ("person1.jpg", TEST_IMAGE_1.read_bytes(), "image/jpeg")},
        )
        assert r.status_code == 200

        r = await client.post(
            "/verify",
            params={"user_id": "123"},
            files={"file": ("person1_2.jpg", TEST_IMAGE_2.read_bytes(), "image/jpeg")},
        )

        data = r.json()

        assert r.status_code == 200
        assert data["status"] in {"match", "low_confidence"}


@pytest.mark.asyncio
async def test_e2e_different_person():

    img1 = encode_image(TEST_IMAGE_1)
    img2 = encode_image(TEST_IMAGE_OTHER)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/verify",
            params={"user_id": "456"},
            files={"file": ("person2.jpg", TEST_IMAGE_OTHER.read_bytes(), "image/jpeg")},
        )

        data = r.json()

        assert data["status"] in ["no_match", "low_confidence"]
