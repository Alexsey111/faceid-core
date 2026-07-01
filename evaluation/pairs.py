# evaluation/pairs.py — построение пар и gallery/probe-сплита для recognition eval.
#
# Чистый numpy, БЕЗ импорта app: позволяет гонять unit-тесты без инфры/моделей.
#
# Контракт:
#   ids — 1D массив «идентификатор личности» для каждого сэмпла (выровнен с
#         массивом эмбеддингов E.shape=(N,512)). Порядок сэмплов детерминирован
#         (datasets.iter_id_images даёт отсортированные по id, затем по файлу).
#   build_pairs_1to1 → (idx_i, idx_j, labels) — индексы в E, labels 1=genuine/0=impostor.
#   gallery_probe_split → (gallery_idx, probe_idx) — индексы; 1-фото ids идут только в gallery.

from __future__ import annotations

import numpy as np

DEFAULT_IMPOSTOR_RATIO = 10
DEFAULT_SEED = 42


def build_pairs_1to1(
    ids: np.ndarray,
    impostor_ratio: int = DEFAULT_IMPOSTOR_RATIO,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Все genuine same-id пары (combinations) + impostor diff-id пары (ratio×n_genuine).

    Детерминизм: np.random.default_rng(seed). Impostor-пары генерируются равномерно
    по всем индексам, дедуп по (min,max); попытки ограничены n_impostor*20 (иначе
    датасет слишком маленький — возвращаем сколько набрали, см. assertion в тестах).
    Возвращает (idx_i, idx_j, labels) — int64, labels 1 genuine / 0 impostor.
    """
    ids = np.asarray(ids)
    n = len(ids)
    if n < 2:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    # ---- genuine: все C(k,2) по каждому id ----
    genu_i: list[int] = []
    genu_j: list[int] = []
    unique_ids, inverse = np.unique(ids, return_inverse=True)
    for gid in range(len(unique_ids)):
        members = np.where(inverse == gid)[0]
        if len(members) < 2:
            continue
        # combinations (i<j) в порядке отсортированных индексов
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                genu_i.append(int(members[a]))
                genu_j.append(int(members[b]))
    genu_i = np.asarray(genu_i, dtype=np.int64)
    genu_j = np.asarray(genu_j, dtype=np.int64)
    n_genuine = len(genu_i)

    # ---- impostor: ratio × n_genuine, diff-id, дедуп ----
    n_impostor = impostor_ratio * n_genuine
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, int]] = set()
    imp_i: list[int] = []
    imp_j: list[int] = []
    max_tries = max(100, n_impostor * 20)
    tries = 0
    while len(imp_i) < n_impostor and tries < max_tries:
        tries += 1
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        if inverse[a] == inverse[b]:
            continue  # same id → не impostor
        seen.add(key)
        imp_i.append(key[0])
        imp_j.append(key[1])

    idx_i = np.concatenate([genu_i, np.asarray(imp_i, dtype=np.int64)]) if imp_i else genu_i
    idx_j = np.concatenate([genu_j, np.asarray(imp_j, dtype=np.int64)]) if imp_i else genu_j
    labels = np.concatenate(
        [np.ones(n_genuine, dtype=np.int64), np.zeros(len(imp_i), dtype=np.int64)]
    )
    return idx_i, idx_j, labels


def gallery_probe_split(
    ids: np.ndarray,
    sort_keys: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gallery = первый сэмпл каждого id (по порядку сортировки), probe = остальные.
    Порядок: лексикографический по (id, sort_key) — по умолчанию sort_key = индекс.

    Ids с одним фото → попадают только в gallery (не queried в 1:N).
    Возвращает (gallery_idx, probe_idx) — int64 индексы в исходный массив.
    """
    ids = np.asarray(ids)
    n = len(ids)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if sort_keys is None:
        sort_keys = np.arange(n, dtype=np.int64)
    sort_keys = np.asarray(sort_keys, dtype=np.int64)

    # Лексикографическая сортировка по (id, sort_key): np.lexsort берёт последний ключ
    # как первичный → порядок аргументов (sort_key, id).
    order = np.lexsort((sort_keys, ids))
    ordered_ids = ids[order]
    # Первый элемент каждой группы id → gallery.
    is_first = np.ones(n, dtype=bool)
    is_first[1:] = ordered_ids[1:] != ordered_ids[:-1]
    gallery_pos = np.where(is_first)[0]
    probe_pos = np.where(~is_first)[0]
    gallery_idx = np.sort(order[gallery_pos])
    probe_idx = np.sort(order[probe_pos])
    return gallery_idx, probe_idx


def pair_scores(
    embeddings: np.ndarray, idx_i: np.ndarray, idx_j: np.ndarray
) -> np.ndarray:
    """
    Косинус-сходство для пар (эмбеддинги L2-нормализованы → cosine = dot).
    scores_k = dot(E[idx_i[k]], E[idx_j[k]]).
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    a = embeddings[np.asarray(idx_i, dtype=np.int64)]
    b = embeddings[np.asarray(idx_j, dtype=np.int64)]
    return np.einsum("ij,ij->i", a, b).astype(np.float64)