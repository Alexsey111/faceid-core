import base64
import requests
from pathlib import Path

API = "http://localhost:8000"

TEST_IMAGE_1 = Path("tests/data/person1.jpg")
TEST_IMAGE_2 = Path("tests/data/person1_2.jpg")
TEST_IMAGE_OTHER = Path("tests/data/person2.jpg")


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def test_e2e_same_person():

    img1 = encode_image(TEST_IMAGE_1)
    img2 = encode_image(TEST_IMAGE_2)

    # enroll
    r = requests.post(
        f"{API}/upload_base64",
        json={"user_id": "123", "image": img1},
    )
    assert r.status_code == 200

    # verify
    r = requests.post(
        f"{API}/verify_base64",
        json={
            "user_id": "123",
            "image": img2,
            "require_liveness": False
        },
    )

    data = r.json()

    assert r.status_code == 200
    assert data["status"] in {"match", "low_confidence"}


def test_e2e_different_person():

    img1 = encode_image(TEST_IMAGE_1)
    img2 = encode_image(TEST_IMAGE_OTHER)

    r = requests.post(
        f"{API}/verify_base64",
        json={
            "user_id": "456",
            "image": img2,
            "require_liveness": False
        },
    )

    data = r.json()

    assert data["status"] in ["no_match", "low_confidence"]
