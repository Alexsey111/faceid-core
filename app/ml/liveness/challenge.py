# app/ml/liveness/challenge.py — active challenge-response liveness (online access control).
#
# Протокол: сервер выдаёт случайный набор действий (blink / turn_left / turn_right /
# nod / smile); клиент в реалтайме стримит кадры (WS); движок по последовательности
# кадров проверяет, что запрошенные действия выполнены, + 3D-consistency (анти
# jump-cut/replay) + passive-texture (MiniFASNetV2) fuse → is_live + confidence.
#
# Зачем: passive MiniFASNetV2 (AUC 0.97) провалится по cutout (APCER=0.4052) —
# распечатка/фото не выполнит 3D-действие. Random nonce + single-use + TTL бьют
# pre-recorded replay; 3D-consistency бьёт плоский-экран replay (см. memory
# liveness-model-eval, liveness-production-wiring).
#
# Landmarks: 2d106det (106-pt 2D) — для EAR (blink) и pose. SCRFD 5pt даёт только
# центры глаз → EAR невозможен; 5pt используется для coarse yaw/pitch и для
# выбора eye-контура у 106pt (self-calibrating по proximity, без хардкода индексов).
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from app.core.config import settings

# Набор действий. turn_left/turn_right детектятся одинаково (экскурсия yaw ≥ порога,
# любое направление) — MVP принимает любой поворот (print повернуть не может в
# принципе); direction-специфика (anti-replay) — scaling-задача. _sample_actions
# берёт не более одного turn-направления в один challenge, чтобы один поворот
# удовлетворял ровно одному запрошенному turn-действию.
ACTIONS: tuple[str, ...] = ("blink", "turn_left", "turn_right", "nod", "smile")
_TURN_ACTIONS: frozenset[str] = frozenset({"turn_left", "turn_right"})


@dataclass
class FrameObservation:
    """Наблюдение по одному кадру стрима."""

    idx: int
    bbox: tuple[float, float, float, float]      # xyxy главного лица
    lm106: np.ndarray | None                      # (106,2) abs-px или None
    lm5: np.ndarray                               # (5,2) abs-px (SCRFD kpss)
    yaw: float                                    # грубый yaw, градусы
    pitch_signal: float                           # nose_rel (норм. сигнал pitch)
    ear: float | None                             # EAR (min из двух глаз) или None
    mouth_width_ratio: float                      # ширина рта / ширина лица
    passive_score: float                          # MiniFASNetV2 real-score по кадру
    ts_ms: float                                  # метка времени (monotonic, мс)


@dataclass
class ChallengeResult:
    is_live: bool
    confidence: float
    actions_performed: dict[str, bool] = field(default_factory=dict)
    consistency_ok: bool = False
    n_frames: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "is_live": self.is_live,
            "confidence": round(self.confidence, 4),
            "actions_performed": self.actions_performed,
            "consistency_ok": self.consistency_ok,
            "n_frames": self.n_frames,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Геометрия: EAR, yaw, pitch, smile — self-calibrating, без хардкода индексов 106pt
# ---------------------------------------------------------------------------

def _ear_for_eye(lm106: np.ndarray, eye_center: np.ndarray, radius: float) -> float | None:
    """EAR-прокси для одного глаза: bbox-соотношение (h/w) точек 106pt в радиусе
    eye_center. При моргании глаз закрывается → h падает → ratio падает. Радиус
    берётся от межглазного расстояния (self-calibrating), поэтому не зависит от
    порядка индексов 106pt и масштаба лица.
    """
    if lm106 is None or len(lm106) == 0:
        return None
    d = np.linalg.norm(lm106 - eye_center, axis=1)
    pts = lm106[d <= radius]
    if len(pts) < 4:
        return None
    w = float(pts[:, 0].max() - pts[:, 0].min())
    h = float(pts[:, 1].max() - pts[:, 1].min())
    if w < 2.0:
        return None
    return h / w


def _yaw_from_5pt(lm5: np.ndarray) -> float:
    """Грубый yaw (градусы) по асимметрии носа между глазами.

    le/re — центры глаз, nose — нос. При повороте нос смещается к одному глазу.
    Возвращает знаковый угол; для детекции поворота важна экскурсия от baseline,
    не абсолютная калибровка.
    """
    le, re, nose = lm5[0], lm5[1], lm5[2]
    d_le = float(nose[0] - le[0])
    d_re = float(re[0] - nose[0])
    denom = d_le + d_re
    if denom < 1e-6:
        return 0.0
    asym = (d_le - d_re) / denom
    return float(np.degrees(np.arcsin(np.clip(asym * 1.2, -1.0, 1.0))))


def _pitch_signal_from_5pt(lm5: np.ndarray) -> float:
    """Нормированный сигнал pitch: вертикальное положение носа между глазами и ртом.

    0 — нос на уровне глаз, 1 — на уровне рта. При кивке вниз нос приближается
    ко рту → сигнал растёт. Используется как относительный сигнал (экскурсия).
    """
    le, re, nose, ml, mr = lm5[0], lm5[1], lm5[2], lm5[3], lm5[4]
    eye_y = (le[1] + re[1]) / 2.0
    mouth_y = (ml[1] + mr[1]) / 2.0
    face_h = mouth_y - eye_y
    if face_h < 1e-6:
        return 0.0
    return float((nose[1] - eye_y) / face_h)


def _mouth_width_ratio(lm5: np.ndarray, bbox: tuple[float, ...]) -> float:
    """Ширина рта (уголки) / ширина bbox лица. Растёт при улыбке."""
    ml, mr = lm5[3], lm5[4]
    mw = float(abs(ml[0] - mr[0]))
    fw = float(bbox[2] - bbox[0])
    if fw < 1.0:
        return 0.0
    return mw / fw


def _bbox_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------------------
# Построение наблюдения по кадру
# ---------------------------------------------------------------------------

def observe_frame(
    image: np.ndarray,
    detector,
    landmarker,
    passive_checker,
    idx: int,
    t0: float,
) -> FrameObservation | None:
    """Детект → выбор главного лица → 106pt → метрики + passive. None если лица нет.

    Не бросает исключения — пропускает кадр при ошибках (стрим не должен рваться).
    """
    from app.ml.detection.face_selection import select_main_face

    try:
        faces = detector.detect(image)
    except Exception:
        return None
    if not faces:
        return None
    top = select_main_face(faces)
    bbox = tuple(float(v) for v in top["bbox"])
    lm5 = np.asarray(top.get("landmarks") or [[0, 0]] * 5, dtype=np.float32)
    if lm5.shape[0] < 5:
        lm5 = np.asarray([[0, 0]] * 5, dtype=np.float32)

    # 106pt landmarks (может быть None если модель недоступна → blink недоступен)
    lm106 = None
    if landmarker is not None:
        lm106 = landmarker.get(image, bbox)

    # EAR: радиус self-calibrating от межглазного расстояния
    ear = None
    if lm106 is not None and lm5.shape[0] >= 2:
        interocular = float(np.linalg.norm(lm5[0] - lm5[1]))
        radius = max(4.0, interocular * 0.20)
        ear_l = _ear_for_eye(lm106, lm5[0], radius)
        ear_r = _ear_for_eye(lm106, lm5[1], radius)
        if ear_l is not None and ear_r is not None:
            ear = float(min(ear_l, ear_r))

    yaw = _yaw_from_5pt(lm5)
    pitch_sig = _pitch_signal_from_5pt(lm5)
    mouth_w = _mouth_width_ratio(lm5, bbox)

    # passive texture (MiniFASNetV2) по raw bbox детектора (контракт yakhyo_v2)
    passive = 0.0
    if passive_checker is not None:
        try:
            real_score, _ok = passive_checker.predict(image, list(bbox))
            passive = float(real_score) if _ok else 0.0
        except Exception:
            passive = 0.0

    return FrameObservation(
        idx=idx, bbox=bbox, lm106=lm106, lm5=lm5,
        yaw=yaw, pitch_signal=pitch_sig, ear=ear,
        mouth_width_ratio=mouth_w, passive_score=passive,
        ts_ms=(time.monotonic() - t0) * 1000.0,
    )


# ---------------------------------------------------------------------------
# Детекторы действий
# ---------------------------------------------------------------------------

def _baseline(seq: list[float], n: int = 3) -> float:
    """Среднее первых n валидных значений (состояние «покой»)."""
    vals = [v for v in seq if v is not None][:n]
    return float(np.mean(vals)) if vals else 0.0


def _detect_blink(obs: list[FrameObservation]) -> bool:
    ears = [o.ear for o in obs if o.ear is not None]
    if len(ears) < 4:
        return False  # без 106pt/EAR blink не детектим
    base = _baseline(ears, n=3)
    if base <= 1e-6:
        return False
    min_ear = min(ears)
    # падение ≥ LIVENESS_EAR_DIP_RATIO от baseline + возврат (eyes open в конце)
    dipped = min_ear <= base * (1.0 - settings.LIVENESS_EAR_DIP_RATIO)
    recovered = ears[-1] >= base * (1.0 - settings.LIVENESS_EAR_DIP_RATIO * 0.5)
    return bool(dipped and recovered)


def _detect_turn(obs: list[FrameObservation]) -> tuple[bool, str]:
    """Возвращает (произошёл_поворот, направление 'left'|'right'|'').
    Экскурсия yaw ≥ LIVENESS_YAW_MIN_DEG от baseline. Направление — по знаку
    пикового отклонения (image-coord convention; MVP не требует совпадения с
    клиентским 'left/right' — достаточно факта поворота).
    """
    yaws = [o.yaw for o in obs]
    if len(yaws) < 4:
        return False, ""
    base = _baseline(yaws, n=3)
    deltas = [y - base for y in yaws]
    max_d = max(deltas)
    min_d = min(deltas)
    if max_d >= settings.LIVENESS_YAW_MIN_DEG:
        return True, "right" if max_d >= abs(min_d) else "left"
    if abs(min_d) >= settings.LIVENESS_YAW_MIN_DEG:
        return True, "left"
    return False, ""


def _detect_nod(obs: list[FrameObservation]) -> bool:
    sig = [o.pitch_signal for o in obs]
    if len(sig) < 4:
        return False
    base = _baseline(sig, n=3)
    excursion = max(abs(s - base) for s in sig)
    return bool(excursion >= settings.LIVENESS_PITCH_MIN_EXCURSION)


def _detect_smile(obs: list[FrameObservation]) -> bool:
    mw = [o.mouth_width_ratio for o in obs]
    if len(mw) < 4:
        return False
    base = _baseline(mw, n=3)
    return bool(max(mw) - base >= settings.LIVENESS_SMILE_DELTA)


def _check_3d_consistency(obs: list[FrameObservation]) -> bool:
    """Анти jump-cut/replay: стабильность размера bbox + непрерывность трека.

    - CV площади bbox по последовательности ≤ LIVENESS_CONSISTENCY_AREA_CV
      (плоский экран с поворотом / loop дают скачки размера);
    - IoU bbox кадр-к-кадру ≥ LIVENESS_CONSISTENCY_IOU_MIN (нет телепорта лица);
    - достаточно кадров с лицом.
    """
    if len(obs) < settings.LIVENESS_MIN_FRAMES:
        return False
    areas = np.array([(o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1]) for o in obs])
    mean_a = float(areas.mean())
    if mean_a <= 1e-6:
        return False
    cv = float(areas.std() / mean_a)
    if cv > settings.LIVENESS_CONSISTENCY_AREA_CV:
        return False
    for a, b in zip(obs, obs[1:]):
        if _bbox_iou(a.bbox, b.bbox) < settings.LIVENESS_CONSISTENCY_IOU_MIN:
            return False
    return True


# ---------------------------------------------------------------------------
# Главный entry: verify по последовательности наблюдений
# ---------------------------------------------------------------------------

def verify_challenge_stream(
    obs: list[FrameObservation],
    actions: list[str],
    threshold: float | None = None,
) -> ChallengeResult:
    """Свести наблюдения в вердикт liveness по запрошенным действиям.

    score = 0.5*active_frac + 0.3*passive_mean + 0.2*consistency,
    где active_frac — доля выполненных запрошенных действий. Жёсткие гейты:
    consistency_ok И все запрошенные действия выполнены, иначе is_live=False.
    """
    thr = float(settings.LIVENESS_THRESHOLD if threshold is None else threshold)
    if len(obs) < settings.LIVENESS_MIN_FRAMES:
        return ChallengeResult(False, 0.0, {}, False, len(obs), "too_few_frames")

    turn_ok, _dir = _detect_turn(obs)
    performed = {
        "blink": _detect_blink(obs),
        "turn_left": turn_ok,
        "turn_right": turn_ok,
        "nod": _detect_nod(obs),
        "smile": _detect_smile(obs),
    }

    consistency = _check_3d_consistency(obs)
    requested = {a: bool(performed.get(a, False)) for a in actions}
    active_frac = float(sum(1 for v in requested.values() if v) / max(1, len(requested)))
    passive_mean = float(np.mean([o.passive_score for o in obs])) if obs else 0.0
    score = 0.5 * active_frac + 0.3 * passive_mean + 0.2 * (1.0 if consistency else 0.0)

    all_actions = all(requested.values())
    if not consistency:
        reason = "consistency_fail"
        is_live = False
    elif not all_actions:
        reason = "actions_incomplete:" + ",".join(a for a, ok in requested.items() if not ok)
        is_live = False
    else:
        is_live = score >= thr
        reason = "ok" if is_live else "below_threshold"

    return ChallengeResult(
        is_live=bool(is_live),
        confidence=float(min(1.0, max(0.0, score))),
        actions_performed=requested,
        consistency_ok=bool(consistency),
        n_frames=len(obs),
        reason=reason,
    )


def sample_actions(n: int | None = None, *, rng=None) -> list[str]:
    """Случайный набор действий для challenge.

    n = settings.LIVENESS_CHALLENGE_ACTIONS. Не более одного turn-направления
    (turn_left/turn_right) — чтобы один поворот удовлетворял ровно одному
    запрошенному turn-действию (MVP принимает любой поворот).

    Кандидаты = не-turn действия + один случайный turn-направление; выборка n
    из них → turn появляется с разумной частотой, но никогда не два turn'а сразу.
    """
    import random as _r

    n = int(settings.LIVENESS_CHALLENGE_ACTIONS if n is None else n)
    r = rng or _r.Random()
    turn = r.choice(list(_TURN_ACTIONS))
    candidates = [a for a in ACTIONS if a not in _TURN_ACTIONS] + [turn]
    pick = r.sample(candidates, k=min(n, len(candidates)))
    r.shuffle(pick)
    return pick