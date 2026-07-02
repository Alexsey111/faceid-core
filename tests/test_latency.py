# tests\test_latency.py

import time
import requests
import base64


API = "http://localhost:8000"


def test_verify_latency():

    with open("tests/data/person1.jpg", "rb") as f:
        img = base64.b64encode(f.read()).decode()

    start = time.time()

    r = requests.post(
        f"{API}/api/v1/verify_base64",
        json={"user_id": "999", "image": img},
    )

    latency = time.time() - start

    print("Latency:", latency)
    print("Response:", r.json())

    assert r.status_code == 200
    assert latency < 1.0
