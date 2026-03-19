import numpy as np


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_similarity():

    a = np.random.rand(512)
    b = a + np.random.normal(0, 0.01, 512)

    sim = cosine_similarity(a, b)

    assert sim > 0.8