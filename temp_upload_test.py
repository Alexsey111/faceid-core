import asyncio
from pathlib import Path
from httpx import AsyncClient, ASGITransport
from app.main import app

TEST_IMAGE_1 = Path(__file__).parent / "tests" / "images" / "person1.jpg"
TEST_IMAGE_2 = Path(__file__).parent / "tests" / "images" / "person2.jpg"

if not TEST_IMAGE_1.exists():
    TEST_IMAGE_1 = Path("D:/python projects/faceid-core/venv/insightface/data/images/t1.jpg")
if not TEST_IMAGE_2.exists():
    TEST_IMAGE_2 = Path("D:/python projects/faceid-core/venv/insightface/data/images/t1.jpg")

async def main():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        for idx, image_path in enumerate([TEST_IMAGE_1, TEST_IMAGE_2], start=1):
            with open(image_path, "rb") as f:
                resp = await ac.post("/upload?user_id=1", files={"file": (image_path.name, f, "image/jpeg")})
            print(idx, resp.status_code, resp.json())

asyncio.run(main())
