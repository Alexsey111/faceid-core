# demo/desktop_demo.py — desktop-демо FaceID Core (Windows, tkinter + OpenCV).
#
# Нативное окно: авто-подъём demo-стека Docker Compose (с отсечением prod-сервисов),
# загрузка эталона (файл/веб-камера), верификация веб-камерой, авто-обработка retry
# (снять очки/маску) и active challenge (поворот/наклон головы при неуверенной
# идентификации). Минимум действий пользователя.
#
# 152-ФЗ: кадры только в памяти (cv2.imencode → bytes/b64 в локальных переменных),
# не пишутся на диск/логи. Остановка сервиса → docker compose down -v (чистит volumes
# с биометрией). Не production: AUTH_ENABLED=false через docker-compose.demo.yml.
#
# Threading: tkinter не thread-safe → CameraThread + worker-потоки + ChallengeSession
# кладут сообщения в queue.Queue; main-цикл drain'ит через root.after(30) и только он
# трогает виджеты.
#
# Запуск: demo/run_demo.bat (двойной клик) — проверяет python/docker/зависимости.

from __future__ import annotations

import base64
import json
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import requests
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, scrolledtext, ttk

try:
    import websocket  # websocket-client
except ImportError:  # pragma: no cover
    print("[ОШИБКА] Не установлен websocket-client. Запустите demo/run_demo.bat")
    sys.exit(1)


# ================================ Константы ================================

REPO_ROOT = Path(__file__).resolve().parent.parent
API_BASE = "http://localhost:8000"
COMPOSE_CMD = ["docker", "compose", "-f", "docker-compose.yml", "-f", "docker-compose.demo.yml"]
SERVICES = ["postgres", "redis", "minio", "api", "worker"]

READY_TIMEOUT_S = 90
READY_INTERVAL_S = 5
COMPOSE_UP_TIMEOUT_S = 300
COMPOSE_DOWN_TIMEOUT_S = 60

WS_FRAME_INTERVAL_S = 0.6
WS_MAX_FRAMES = 30
WS_RESULT_TIMEOUT_S = 60

# actions из /liveness/challenge/init → русские подписи для оверлея.
ACTION_LABELS: dict[str, str] = {
    "blink": "моргните",
    "turn_left": "поверните голову влево",
    "turn_right": "поверните голову вправо",
    "nod": "кивните головой",
    "smile": "улыбнитесь",
}

# status → цвет бейджа (палитра из web-демо styles.css).
BADGE_COLORS: dict[str, str] = {
    "match": "#3ec47e",
    "no_match": "#e2554a",
    "spoof_detected": "#e2554a",
    "quality_reject": "#e0a93a",
    "retry": "#e0a93a",
    "low_confidence": "#9a9a9a",
    "processing_failed": "#e2554a",
    "no_face": "#e0a93a",
    "ok": "#3ec47e",
    "error": "#e2554a",
    "info": "#4a7de2",
}

TERMINAL_JOB_STATUSES = {"done", "error", "expired", "failed"}

# Куда пишем traceback при падении (лог из окна не копируется — файл можно прислать).
_CRASH_LOG = REPO_ROOT / "demo" / "_demo_crash.log"
# Дубликат лог-зоны окна в файл (лог из окна не копируется — файл можно прислать).
# Только человекочитаемые строки (status/score/reason), без кадров/эмбеддингов (152-ФЗ).
_UI_LOG = REPO_ROOT / "demo" / "_demo_ui.log"


def _write_crash(exc_type: Any, exc: Any, tb: Any) -> None:
    """Записать traceback в demo/_demo_crash.log (перезапись). 152-ФЗ: только текст
    исключения, без кадров/эмбеддингов (их в traceback от демо не бывает)."""
    try:
        with open(_CRASH_LOG, "w", encoding="utf-8") as f:
            f.write("".join(traceback.format_exception(exc_type, exc, tb)))
    except Exception:
        pass


def _threading_excepthook(args: Any) -> None:
    """Ловим падения worker-потоков (иначе они гибнут молча)."""
    _write_crash(args.exc_type, args.exc_value, args.exc_traceback)


# ================================= Утилиты =================================

def b64_jpeg(frame: Any, quality: int = 90) -> str | None:
    """Кадр (np.ndarray BGR) → base64 JPEG (без data:-prefix). Только в памяти."""
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def bytes_jpeg(frame: Any, quality: int = 90) -> bytes | None:
    """Кадр → JPEG bytes (для multipart /liveness и WS-стрима). Только в памяти."""
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return buf.tobytes()


def ws_close_reason(code: int | None) -> str:
    """Карта WS close-кодов → человекочитаемое сообщение (из liveness_challenge.py)."""
    return {
        4410: "Challenge истёк или неизвестен — начните заново",
        4401: "Неверный ws_token — начните заново",
        4503: "Active liveness отключён или сервер занят (проверьте LIVENESS_ACTIVE_ENABLED)",
        4409: "Challenge уже стримится (конфликт) — начните заново",
        4400: "Невалидное состояние challenge",
        1006: "Соединение разорвано",
    }.get(code, f"Соединение закрыто (код {code})")


def read_image_file(path: str) -> Any:
    """Прочитать изображение через np.fromfile + cv2.imdecode (поддержка кириллицы в
    путях Windows — cv2.imread её ломает). Возвращает BGR ndarray или None. Только в
    память, без записи."""
    try:
        buf = np.fromfile(path, dtype="uint8")
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ============================ Docker-оркестрация ============================

class DockerError(Exception):
    """Ошибка запуска/остановки demo-стека."""


class DockerOrchestrator:
    """Обёртка над `docker compose` для demo-стека. Все вызовы блокирующие —
    вызывать только в worker-потоке, не из main (tkinter)."""

    def up(self) -> None:
        """Поднять 5 сервисов (отсекает api_lb/prometheus/postgres_test)."""
        try:
            res = subprocess.run(
                COMPOSE_CMD + ["up", "-d", "--build"] + SERVICES,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=COMPOSE_UP_TIMEOUT_S,
            )
        except FileNotFoundError as exc:
            raise DockerError("Docker не найден в PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise DockerError(f"docker compose up превысил {COMPOSE_UP_TIMEOUT_S}с") from exc
        if res.returncode != 0:
            raise DockerError(self._format_stderr(res.stderr))

    def down_v(self) -> None:
        """Остановить стек и удалить volumes (биометрия, 152-ФЗ)."""
        try:
            subprocess.run(
                COMPOSE_CMD + ["down", "-v"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=COMPOSE_DOWN_TIMEOUT_S,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # best-effort: не блокируем выход

    def wait_ready(self) -> bool:
        """Poll GET /ready до 200 (db+redis ok)."""
        deadline = time.monotonic() + READY_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                r = requests.get(f"{API_BASE}/ready", timeout=3)
                if r.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(READY_INTERVAL_S)
        return False

    @staticmethod
    def _format_stderr(stderr: str) -> str:
        head = (stderr or "").strip().splitlines()
        msg = head[-1] if head else "docker compose up завершился с ошибкой"
        if "port is already allocated" in stderr or "address already in use" in stderr:
            return "Порты заняты (8000/5432/6379/9000). Остановите конфликтующие контейнеры."
        return msg


# ================================ ApiClient ================================

class ApiError(Exception):
    """HTTP-ошибка API с человекочитаемым detail."""


class ApiClient:
    """Синхронный REST-клиент к /api/v1. Все вызовы блокирующие — в worker-потоке."""

    def __init__(self) -> None:
        self._s = requests.Session()
        # 1 ретрай на сетевые сбои (демо: сервис мог кратко перезапуститься).
        adapter = requests.adapters.HTTPAdapter(max_retries=1)
        self._s.mount("http://", adapter)

    @staticmethod
    def _detail(resp: requests.Response) -> str:
        try:
            data = resp.json()
            return str(data.get("detail") or data)
        except ValueError:
            return f"HTTP {resp.status_code}"

    def upload_base64(self, user_id: str, image_b64: str) -> dict[str, Any]:
        resp = self._s.post(
            f"{API_BASE}/api/v1/upload_base64",
            json={"user_id": user_id, "image": image_b64},
            timeout=30,
        )
        if resp.status_code >= 400:
            raise ApiError(self._detail(resp))
        return resp.json()

    def verify_base64(
        self,
        user_id: str,
        image_b64: str,
        require_liveness: bool = True,
        liveness_mode: str = "passive",
        liveness_token: str | None = None,
    ) -> dict[str, Any]:
        """Verify (sync fast-path) + async-fallback через /jobs/{id}/wait long-poll.
        Возвращает финальный dict с полями VerifyResponse (или {error:...})."""
        payload: dict[str, Any] = {
            "user_id": user_id,
            "image": image_b64,
            "require_liveness": require_liveness,
            "liveness_mode": liveness_mode,
        }
        if liveness_token:
            payload["liveness_token"] = liveness_token
        resp = self._s.post(
            f"{API_BASE}/api/v1/verify_base64", json=payload, timeout=30
        )
        if resp.status_code >= 400:
            raise ApiError(self._detail(resp))
        data = resp.json()
        # async-fallback: {job_id, status:"pending"} → long-poll результата.
        if data.get("status") == "pending" and "job_id" in data:
            return self._poll_job(data["job_id"])
        return data

    def _poll_job(self, job_id: str) -> dict[str, Any]:
        """Long-poll /jobs/{id}/wait до терминального статуса, max 60с.

        60с (вместо 30с) — запас под cold-start: первый verify после подъёма стека
        ждёт загрузки ONNX-моделей (ArcFace/SCRFD/MiniFASNet) в worker на CPU,
        что может занимать >30с. Тёплые запросы укладываются в секунды."""
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                r = self._s.get(
                    f"{API_BASE}/api/v1/jobs/{job_id}/wait",
                    params={"timeout": 2000},
                    timeout=5,
                )
            except requests.RequestException as exc:
                raise ApiError(f"poll /jobs/{job_id}: {exc}")
            if r.status_code >= 400:
                raise ApiError(self._detail(r))
            d = r.json()
            if d.get("status") in TERMINAL_JOB_STATUSES:
                # envelope: {job_id, status, result:{...VerifyResponse...}, ...}
                return d.get("result") or d
        return {"status": "processing_failed", "reason": "timeout", "error_code": "poll_timeout"}

    def challenge_init(self) -> dict[str, Any]:
        """POST /liveness/challenge/init → {challenge_id, actions, ws_token, ws_url, expires_at}."""
        resp = self._s.post(f"{API_BASE}/api/v1/liveness/challenge/init", timeout=10)
        if resp.status_code >= 400:
            raise ApiError(self._detail(resp))
        return resp.json()

    def get_config(self) -> dict[str, Any]:
        """GET /config → 6 публичных порогов (read-only, без секретов)."""
        try:
            r = self._s.get(f"{API_BASE}/api/v1/config", timeout=5)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        return {}

    def is_ready(self) -> bool:
        try:
            return self._s.get(f"{API_BASE}/ready", timeout=3).status_code == 200
        except requests.RequestException:
            return False


# ============================== CameraThread ===============================

class CameraThread(threading.Thread):
    """Фоновый захват cv2.VideoCapture(0). Хранит latest-кадр под Lock; отдаёт
    JPEG-bytes/base64 по запросу. Ничего не пишет на диск (152-ФЗ)."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Any = None
        self.error: str | None = None

    def run(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error = "Камера недоступна (индекс 0). Закройте другие приложения с камерой."
            return
        while not self._stop.is_set():
            ok, frame = cap.read()
            if ok:
                with self._lock:
                    self._latest = frame
            time.sleep(0.02)  # ~50fps опрос
        cap.release()

    def stop(self) -> None:
        self._stop.set()

    def grab_frame(self) -> Any:
        """Копия текущего кадра (np.ndarray BGR) для превью."""
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def grab_jpeg_b64(self) -> str | None:
        return b64_jpeg(self.grab_frame())

    def grab_jpeg_bytes(self) -> bytes | None:
        return bytes_jpeg(self.grab_frame())


# ============================= ChallengeSession ============================

class ChallengeSession:
    """Active-challenge WS-сессия. Стримит JPEG-кадры с камеры каждые 600мс (max 30),
    реагирует на done/cancel, возвращает result через callback. Кадры берёт у
    CameraThread (не открывает свой capture — OpenCV не любит concurrent access)."""

    def __init__(
        self,
        camera: CameraThread,
        on_result: Callable[[dict[str, Any]], None],
        on_close: Callable[[int | None, str], None],
        on_challenge: Callable[[dict[str, Any]], None],
    ) -> None:
        self._camera = camera
        self._on_result = on_result
        self._on_close = on_close
        self._on_challenge = on_challenge
        self._done = threading.Event()
        self._cancel = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.liveness_token: str | None = None

    def start(self, init_data: dict[str, Any]) -> None:
        ws_url = f"ws://localhost:8000{init_data['ws_url']}"
        self._thread = threading.Thread(
            target=self._run, args=(ws_url,), daemon=True
        )
        self._thread.start()

    def send_done(self) -> None:
        self._done.set()

    def send_cancel(self) -> None:
        self._cancel.set()

    def stop(self) -> None:
        self._stop.set()
        self._cancel.set()

    def _run(self, ws_url: str) -> None:
        ws = websocket.WebSocket()
        try:
            ws.connect(ws_url, timeout=10)
            # 1) сервер сразу шлёт {type:"challenge", actions, deadline_ms}.
            ws.settimeout(10)
            first = self._recv_json(ws)
            if first is not None:
                self._on_challenge(first)
            # 2) цикл стрима кадров + неблокирующий приём result.
            frames = 0
            deadline = time.monotonic() + 60
            while not self._stop.is_set() and frames < WS_MAX_FRAMES and time.monotonic() < deadline:
                if self._cancel.is_set():
                    self._send_json(ws, {"cmd": "cancel"})
                    break
                frame = self._camera.grab_jpeg_bytes()
                if frame is not None:
                    ws.settimeout(5)
                    try:
                        ws.send_binary(frame)
                        frames += 1
                    except Exception:
                        break
                # короткий неблокирующий приём (result мог прийти между кадрами)
                msg = self._recv_json(ws, timeout=0.1)
                if msg is not None and msg.get("type") == "result":
                    self._handle_result(msg)
                    return
                if self._done.is_set():
                    self._send_json(ws, {"cmd": "done"})
                    self._done.clear()
                    # блокирующе ждём result
                    msg = self._recv_json(ws, timeout=WS_RESULT_TIMEOUT_S)
                    if msg is not None and msg.get("type") == "result":
                        self._handle_result(msg)
                        return
                    break
                time.sleep(WS_FRAME_INTERVAL_S)
            # поток кадров исчерпан — запросим вердикт явно.
            if not self._cancel.is_set() and not self._stop.is_set():
                self._send_json(ws, {"cmd": "done"})
                msg = self._recv_json(ws, timeout=WS_RESULT_TIMEOUT_S)
                if msg is not None and msg.get("type") == "result":
                    self._handle_result(msg)
        except Exception as exc:
            self._on_close(None, f"WS-ошибка: {exc}")
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def _handle_result(self, msg: dict[str, Any]) -> None:
        if msg.get("is_live") and msg.get("liveness_token"):
            self.liveness_token = msg["liveness_token"]
        self._on_result(msg)

    @staticmethod
    def _recv_json(ws: websocket.WebSocket, timeout: float = 10.0) -> dict[str, Any] | None:
        ws.settimeout(timeout)
        try:
            raw = ws.recv()
        except Exception:
            return None
        if raw is None or raw == "":
            return None
        if isinstance(raw, bytes):
            return None  # сервер не шлёт бинарные клиенту в этом протоколе
        try:
            return json.loads(raw)
        except ValueError:
            return None

    @staticmethod
    def _send_json(ws: websocket.WebSocket, obj: dict[str, Any]) -> None:
        ws.settimeout(5)
        ws.send(json.dumps(obj))


# ============================= DesktopDemoApp ==============================

# Виды сообщений очереди (worker → main).
M_LOG = "log"
M_READY = "ready"
M_READY_FAIL = "ready_fail"
M_STOPPED = "stopped"
M_CONFIG = "config"
M_UPLOAD = "upload"
M_VERIFY = "verify"
M_CHALLENGE_INIT_OK = "ch_init_ok"
M_CHALLENGE_INIT_ERR = "ch_init_err"
M_CHALLENGE_MSG = "ch_msg"
M_CHALLENGE_RESULT = "ch_result"
M_CHALLENGE_CLOSED = "ch_closed"


class DesktopDemoApp(tk.Tk):
    """Главное окно демо: панель камеры + управление/результат/лог."""

    def __init__(self) -> None:
        super().__init__()
        self.title("FaceID Core — desktop demo")
        self.geometry("1120x740")
        self.minsize(960, 640)
        self.configure(bg="#1e1e1e")

        self.msg_q: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.api = ApiClient()
        self.orchestrator = DockerOrchestrator()
        self.camera = CameraThread()
        self.challenge: ChallengeSession | None = None
        self.busy = False
        self.service_ready = False
        self.closing = False
        self._photo: ImageTk.PhotoImage | None = None  # анти-GC для превью

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Камера стартует сразу — превью живёт независимо от сервиса.
        self.camera.start()
        self._set_overlay("Нажмите «Запустить сервис» →", "#4a7de2")
        self.after(30, self._pump_ui)

    # ---------------------------- построение UI ----------------------------

    def _build_ui(self) -> None:
        # Левая панель: камера.
        left = tk.Frame(self, bg="#1e1e1e", width=500)
        left.pack(side="left", fill="y", padx=8, pady=8)
        left.pack_propagate(False)

        self.cam_label = tk.Label(left, bg="#000000", width=480, height=360)
        self.cam_label.pack(anchor="n", padx=4, pady=4)
        # оверлей-инструкции поверх превью.
        self.overlay_var = tk.StringVar(value="")
        self.overlay = tk.Label(
            left, textvariable=self.overlay_var, bg="#000000", fg="#ffffff",
            font=("Segoe UI", 13, "bold"), wraplength=460, justify="center",
            padx=10, pady=6,
        )
        self.overlay.place(relx=0.5, rely=0.82, anchor="center")
        self.cam_status_var = tk.StringVar(value="Камера: запуск…")
        tk.Label(left, textvariable=self.cam_status_var, bg="#1e1e1e", fg="#aaaaaa",
                 font=("Segoe UI", 9)).pack(anchor="w", padx=6)

        # Правая панель: управление.
        right = tk.Frame(self, bg="#1e1e1e")
        right.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        # Шапка: user_id + сервис.
        top = tk.Frame(right, bg="#1e1e1e")
        top.pack(fill="x", pady=(0, 6))
        tk.Label(top, text="user_id:", bg="#1e1e1e", fg="#dddddd",
                 font=("Segoe UI", 10)).pack(side="left")
        # Серверный /upload_base64 делает int(user_id) — нужен ЧИСЛОВОЙ id.
        self.user_var = tk.StringVar(value="1001")
        ttk.Entry(top, textvariable=self.user_var, width=18).pack(side="left", padx=4)
        self.btn_start = ttk.Button(top, text="Запустить сервис", command=self._on_start)
        self.btn_start.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(top, text="Остановить сервис", command=self._on_stop,
                                   state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        self.btn_config = ttk.Button(top, text="Config…", command=self._on_show_config,
                                     state="disabled")
        self.btn_config.pack(side="left", padx=4)

        # Параметры верификации.
        params = tk.Frame(right, bg="#1e1e1e")
        params.pack(fill="x", pady=4)
        self.req_live_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params, text="требовать liveness (passive)",
                        variable=self.req_live_var).pack(side="left")
        self.auto_engage_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params, text="авто active-challenge при неуверенной",
                        variable=self.auto_engage_var).pack(side="left", padx=8)

        # Эталон.
        f_ref = ttk.LabelFrame(right, text="Эталон", padding=6)
        f_ref.pack(fill="x", pady=4)
        self.btn_upload_file = ttk.Button(f_ref, text="Из файла…", command=self._on_upload_file)
        self.btn_upload_file.pack(side="left")
        self.btn_upload_cam = ttk.Button(f_ref, text="Снять с камеры", command=self._on_upload_cam)
        self.btn_upload_cam.pack(side="left", padx=4)
        self.ref_status = tk.Label(f_ref, text="эталон не загружен", bg="#1e1e1e", fg="#aaaaaa",
                                   font=("Segoe UI", 9))
        self.ref_status.pack(side="left", padx=8)

        # Верификация.
        f_ver = ttk.LabelFrame(right, text="Верификация", padding=6)
        f_ver.pack(fill="x", pady=4)
        self.verify_btn = ttk.Button(f_ver, text="Верифицировать (камера)",
                                     command=lambda: self._on_verify(from_cam=True))
        self.verify_btn.pack(side="left")
        self.btn_verify_file = ttk.Button(f_ver, text="Из файла…",
                                          command=lambda: self._on_verify(from_cam=False))
        self.btn_verify_file.pack(side="left", padx=4)

        # Результат.
        f_res = ttk.LabelFrame(right, text="Результат", padding=6)
        f_res.pack(fill="both", expand=False, pady=4)
        self.badge = tk.Label(f_res, text="—", bg="#333333", fg="#ffffff",
                              font=("Segoe UI", 12, "bold"), padx=10, pady=4)
        self.badge.pack(anchor="w")
        self.score_bar = self._add_bar(f_res, "match_score")
        self.real_bar = self._add_bar(f_res, "real_prob")
        self.spoof_bar = self._add_bar(f_res, "spoof_prob")
        self.result_detail = tk.Label(
            f_res, text="", bg="#1e1e1e", fg="#cccccc", font=("Segoe UI", 9),
            justify="left", anchor="w", wraplength=560,
        )
        self.result_detail.pack(anchor="w", fill="x")

        # Active challenge.
        f_ch = ttk.LabelFrame(right, text="Active challenge", padding=6)
        f_ch.pack(fill="x", pady=4)
        self.ch_status = tk.Label(f_ch, text="не запущен", bg="#1e1e1e", fg="#aaaaaa",
                                  font=("Segoe UI", 9), anchor="w")
        self.ch_status.pack(anchor="w")
        ch_btns = tk.Frame(f_ch, bg="#1e1e1e")
        ch_btns.pack(fill="x", pady=2)
        ttk.Button(ch_btns, text="Запустить вручную", command=self._on_challenge_start).pack(side="left")
        self.btn_ch_done = ttk.Button(ch_btns, text="Готово", command=self._on_challenge_done,
                                      state="disabled")
        self.btn_ch_done.pack(side="left", padx=4)
        self.btn_ch_cancel = ttk.Button(ch_btns, text="Отмена", command=self._on_challenge_cancel,
                                        state="disabled")
        self.btn_ch_cancel.pack(side="left")

        # Лог.
        f_log = ttk.LabelFrame(right, text="Лог", padding=4)
        f_log.pack(fill="both", expand=True, pady=4)
        self.log = scrolledtext.ScrolledText(f_log, height=8, state="disabled",
                                             bg="#111111", fg="#cfcfcf",
                                             font=("Consolas", 9), wrap="word")
        self.log.pack(fill="both", expand=True)

        self._set_actions_state()

    def _add_bar(self, parent: tk.Widget, label: str) -> ttk.Progressbar:
        row = tk.Frame(parent, bg="#1e1e1e")
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label, bg="#1e1e1e", fg="#aaaaaa", width=12, anchor="w",
                 font=("Segoe UI", 9)).pack(side="left")
        bar = ttk.Progressbar(row, length=260, maximum=1.0, value=0)
        bar.pack(side="left", padx=4)
        return bar

    # ------------------------------ насос UI -------------------------------

    def _pump_ui(self) -> None:
        if self.closing:
            return
        # превью камеры
        frame = self.camera.grab_frame()
        if frame is not None:
            self._render_preview(frame)
        elif self.camera.error and not self.service_ready:
            self.cam_status_var.set(f"Камера: {self.camera.error}")
        # drain очереди сообщений
        while True:
            try:
                kind, payload = self.msg_q.get_nowait()
            except queue.Empty:
                break
            self._dispatch(kind, payload)
        self.after(30, self._pump_ui)

    def _render_preview(self, frame: Any) -> None:
        # BGR → RGB → PIL → ImageTk. Только в памяти.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((480, 360))
        self._photo = ImageTk.PhotoImage(img)
        self.cam_label.configure(image=self._photo)

    def _dispatch(self, kind: str, payload: Any) -> None:
        if kind == M_LOG:
            self._append_log(str(payload))
        elif kind == M_READY:
            self._on_ready_main()
        elif kind == M_READY_FAIL:
            self._on_ready_fail_main(str(payload))
        elif kind == M_STOPPED:
            self._on_stopped_main()
        elif kind == M_CONFIG:
            self._show_config_dialog(payload)
        elif kind == M_UPLOAD:
            self._render_upload(payload)
        elif kind == M_VERIFY:
            self._render_verify(payload)
        elif kind == M_CHALLENGE_INIT_OK:
            self._on_challenge_init_ok(payload)
        elif kind == M_CHALLENGE_INIT_ERR:
            self._release_busy()
            self.ch_status.configure(text=f"ошибка: {payload}", foreground="#e2554a")
            self._append_log(f"[challenge] init fail: {payload}")
        elif kind == M_CHALLENGE_MSG:
            self._on_challenge_msg(payload)
        elif kind == M_CHALLENGE_RESULT:
            self._on_challenge_result_main(payload)
        elif kind == M_CHALLENGE_CLOSED:
            self._on_challenge_closed_main(payload)

    # --------------------------- лог-зона (152-ФЗ) --------------------------
    def _append_log(self, text: str) -> None:
        # Только человекочитаемые строки; никогда — кадры/эмбеддинги/base64.
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        # Дублируем в файл (диагностика: лог из окна не копируется).
        try:
            with open(_UI_LOG, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            pass

    # ------------------------------ сервис ---------------------------------

    def _on_start(self) -> None:
        self.btn_start.configure(state="disabled")
        self._append_log("[сервис] подъём demo-стека (postgres redis minio api worker)…")
        self._set_overlay("Поднимаю сервис…", "#4a7de2")

        def worker() -> None:
            try:
                self.orchestrator.up()
                self.msg_q.put((M_LOG, "[сервис] контейнеры запущены, жду готовность (/ready)…"))
                if self.orchestrator.wait_ready():
                    self.msg_q.put((M_READY, True))
                else:
                    self.msg_q.put((M_READY_FAIL, "Сервис не стал готов за 90с. См.: docker compose logs api"))
            except DockerError as exc:
                self.msg_q.put((M_READY_FAIL, str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_ready_main(self) -> None:
        self.service_ready = True
        self._busy = False
        self._set_actions_state()
        self._set_overlay("", "#000000")
        self._append_log("[сервис] готов. Камера активна — можно загружать эталон и верифицировать.")
        # подгрузить публичные пороги (неблокирующе)
        threading.Thread(target=self._fetch_config, daemon=True).start()

    def _on_ready_fail_main(self, reason: str) -> None:
        self.service_ready = False
        self._busy = False
        self.btn_start.configure(state="normal")
        self._set_actions_state()
        self._set_overlay("Сервис не готов", "#e2554a")
        self._append_log(f"[сервис] ОШИБКА: {reason}")

    def _fetch_config(self) -> None:
        cfg = self.api.get_config()
        if cfg:
            self.msg_q.put((M_CONFIG, cfg))

    def _on_stop(self) -> None:
        self.btn_stop.configure(state="disabled")
        self._append_log("[сервис] остановка + очистка volumes (down -v, 152-ФЗ)…")

        def worker() -> None:
            self.orchestrator.down_v()
            self.msg_q.put((M_STOPPED, True))

        threading.Thread(target=worker, daemon=True).start()

    def _on_stopped_main(self) -> None:
        self.service_ready = False
        self._busy = False
        self._set_actions_state()
        self._set_overlay("Сервис остановлен. Нажмите «Запустить сервис»", "#4a7de2")
        self._append_log("[сервис] остановлен, volumes очищены.")

    # ------------------------------ config ---------------------------------

    def _on_show_config(self) -> None:
        threading.Thread(target=self._fetch_config, daemon=True).start()

    def _show_config_dialog(self, cfg: dict[str, Any]) -> None:
        if not cfg:
            self._append_log("[config] не удалось получить /api/v1/config")
            return
        win = tk.Toplevel(self)
        win.title("Публичные пороги (/api/v1/config)")
        win.configure(bg="#1e1e1e")
        for key in ("FACE_MATCH_THRESHOLD", "LIVENESS_THRESHOLD", "LIVENESS_ENABLED",
                    "LIVENESS_ACTIVE_ENABLED", "LIVENESS_ACTIVE_REQUIRED", "QUALITY_GATE_MODE"):
            row = tk.Frame(win, bg="#1e1e1e")
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=key, bg="#1e1e1e", fg="#aaaaaa", width=28, anchor="w",
                     font=("Consolas", 9)).pack(side="left")
            tk.Label(row, text=str(cfg.get(key)), bg="#1e1e1e", fg="#ffffff", anchor="w",
                     font=("Consolas", 9)).pack(side="left")

    # ------------------------------- эталон --------------------------------

    def _on_upload_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Выбрать эталон", filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return
        frame = read_image_file(path)
        if frame is None:
            self._append_log("[эталон] не удалось прочитать файл")
            return
        self._do_upload(b64_jpeg(frame))

    def _on_upload_cam(self) -> None:
        b64 = self.camera.grab_jpeg_b64()
        if b64 is None:
            self._append_log("[эталон] камера не готова")
            return
        self._do_upload(b64)

    def _do_upload(self, image_b64: str | None) -> None:
        if not self._guard_busy():
            return
        user_id = self._numeric_user_id()
        if user_id is None:
            self._release_busy()
            return
        self.ref_status.configure(text="загружаю…", foreground="#e0a93a")
        self._append_log(f"[эталон] upload user_id={user_id}…")

        def worker() -> None:
            try:
                resp = self.api.upload_base64(user_id, image_b64 or "")
                self.msg_q.put((M_UPLOAD, resp))
            except Exception as exc:  # любой сбой → UI получает error, busy не залипает
                _write_crash(type(exc), exc, exc.__traceback__)
                self.msg_q.put((M_UPLOAD, {"error": repr(exc)}))

        threading.Thread(target=worker, daemon=True).start()

    def _render_upload(self, payload: dict[str, Any]) -> None:
        self._release_busy()
        if "error" in payload:
            self.ref_status.configure(text=f"ошибка: {payload['error']}", foreground="#e2554a")
            self._append_log(f"[эталон] ОШИБКА: {payload['error']}")
            return
        data = payload.get("data") or {}
        eid = data.get("embedding_id")
        self.ref_status.configure(text=f"эталон сохранён (embedding_id={eid})", foreground="#3ec47e")
        # Полный дамп ответа upload (метаданные, без биометрии) — для диагностики no_match.
        self._append_log(f"[эталон] OK, embedding_id={eid}, full={data}")

    # ------------------------------ верификация ----------------------------

    def _on_verify(self, from_cam: bool) -> None:
        if not self._guard_busy():
            return
        user_id = self._numeric_user_id()
        if user_id is None:
            self._release_busy()
            return
        if from_cam:
            b64 = self.camera.grab_jpeg_b64()
            if b64 is None:
                self._release_busy()
                self._append_log("[verify] камера не готова")
                return
        else:
            path = filedialog.askopenfilename(
                title="Выбрать фото для верификации",
                filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp")],
            )
            if not path:
                self._release_busy()
                return
            frame = read_image_file(path)
            if frame is None:
                self._release_busy()
                self._append_log("[verify] не удалось прочитать файл")
                return
            b64 = b64_jpeg(frame)
        self.verify_btn.configure(text="Верифицировать (камера)")
        self._set_overlay("", "#000000")
        self._append_log(
            f"[verify] {user_id} (require_liveness={self.req_live_var.get()}) — "
            f"обработка, подождите (до ~30с на CPU)…"
        )

        require_liveness = self.req_live_var.get()

        def worker() -> None:
            try:
                resp = self.api.verify_base64(
                    user_id, b64 or "",
                    require_liveness=require_liveness, liveness_mode="passive",
                )
                self.msg_q.put((M_VERIFY, resp))
            except Exception as exc:  # любой сбой → UI получает error, busy не залипает
                _write_crash(type(exc), exc, exc.__traceback__)
                self.msg_q.put((M_VERIFY, {"error": repr(exc)}))

        threading.Thread(target=worker, daemon=True).start()

    def _render_verify(self, payload: dict[str, Any]) -> None:
        self._release_busy()
        if "error" in payload:
            self._set_badge("error", f"ошибка: {payload['error']}")
            self._append_log(f"[verify] ОШИБКА: {payload['error']}")
            return
        status = str(payload.get("status") or "processing_failed")
        score = payload.get("match_score") or payload.get("similarity")
        confidence = payload.get("confidence")
        # Полный дамп результата в лог-файл (только метаданные — 152-ФЗ, без кадров/эмбеддингов).
        # Нужно для диагностики «ни одну верификацию не прошёл»: статус/оценка/живость/причина.
        indicators = payload.get("spoofing_indicators") or {}
        qd = payload.get("quality_details")
        self._append_log(
            "[verify] результат: "
            f"status={status} match_score={score} confidence={confidence} "
            f"liveness_passed={payload.get('liveness_passed')} "
            f"liveness_score={payload.get('liveness_score')} "
            f"reason={payload.get('reason')} error_code={payload.get('error_code')} "
            f"challenge_recommended={payload.get('challenge_recommended')} "
            f"spoofing={indicators} quality={qd}"
        )
        self._set_badge(status, self._status_text(status, score, confidence))
        self._set_bar(self.score_bar, score)
        self._set_bar(self.real_bar, indicators.get("real_prob"))
        self._set_bar(self.spoof_bar, indicators.get("spoof_prob"))
        self._render_detail(payload)
        self._after_verify_branching(payload)

    def _render_detail(self, payload: dict[str, Any]) -> None:
        parts: list[str] = []
        if payload.get("liveness_passed") is not None:
            parts.append(f"liveness_passed={payload['liveness_passed']}")
        if payload.get("liveness_score") is not None:
            parts.append(f"liveness_score={payload['liveness_score']:.3f}")
        if payload.get("confidence"):
            parts.append(f"confidence={payload['confidence']}")
        if payload.get("reason"):
            parts.append(f"reason={payload['reason']}")
        if payload.get("error_code"):
            parts.append(f"error_code={payload['error_code']}")
        qd = payload.get("quality_details")
        if isinstance(qd, dict) and qd:
            occ = qd.get("occlusion_flags")
            if isinstance(occ, dict) and occ:
                parts.append(f"occlusion={occ}")
            else:
                parts.append(f"quality={qd}")
        self.result_detail.configure(text="  ".join(parts) if parts else "—")

    def _after_verify_branching(self, payload: dict[str, Any]) -> None:
        """Минимум действий: авто-реакция на retry и low_confidence/challenge_recommended."""
        status = str(payload.get("status") or "")
        # retry (окклюзия) → оверлей «снимите …» + переименовать кнопку.
        if status == "retry":
            flags = ((payload.get("quality_details") or {}).get("occlusion_flags") or {})
            msg = self._occlusion_text(flags)
            self._set_overlay(msg, "#e0a93a")
            self.verify_btn.configure(text="Переснять")
            self._append_log(f"[verify] retry: {msg} (verification_log не пишется — это не исход)")
            return
        # неуверенная идентификация → предложить/запустить active challenge.
        recommend = bool(payload.get("challenge_recommended"))
        low = payload.get("confidence") == "low" or status == "low_confidence"
        if recommend or low:
            self._set_overlay("Неуверенная идентификация — пройдите active challenge", "#e0a93a")
            self._append_log(f"[verify] рекомендован active-challenge (challenge_recommended={recommend})")
            if self.auto_engage_var.get():
                self.after(1200, self._on_challenge_start)
        else:
            self._set_overlay("", "#000000")

    @staticmethod
    def _occlusion_text(flags: dict[str, Any]) -> str:
        mask = flags.get("mask_detected")
        glasses = flags.get("glasses_detected")
        if mask and glasses:
            return "Снимите маску и очки, затем «Переснять»"
        if mask:
            return "Снимите маску, затем «Переснять»"
        if glasses:
            return "Снимите очки, затем «Переснять»"
        return "Уберите окклюзию с лица, затем «Переснять»"

    @staticmethod
    def _status_text(status: str, score: Any, confidence: Any) -> str:
        base = {
            "match": "СВОЙ — доступ открыт",
            "no_match": "ЧУЖОЙ — отказ",
            "low_confidence": "НЕ УВЕРЕН — серая зона",
            "spoof_detected": "ПОДМЕНА — отказ (spoof)",
            "quality_reject": "ПЛОХОЙ КАДР — переснимите",
            "retry": "ПЕРЕСНЯТЬ — окклюзия",
            "processing_failed": "СБОЙ обработки",
            "no_face": "НЕТ ЛИЦА",
        }.get(status, status)
        extra = []
        if score is not None:
            extra.append(f"score={float(score):.3f}")
        if confidence:
            extra.append(f"conf={confidence}")
        return f"{base}  " + " ".join(extra) if extra else base

    # --------------------------- active challenge --------------------------

    def _on_challenge_start(self) -> None:
        if not self._guard_busy():
            return
        self.ch_status.configure(text="init…", foreground="#e0a93a")
        self._append_log("[challenge] init /liveness/challenge/init…")

        def worker() -> None:
            try:
                init = self.api.challenge_init()
                self.msg_q.put((M_CHALLENGE_INIT_OK, init))
            except Exception as exc:  # любой сбой → UI получает error, busy не залипает
                _write_crash(type(exc), exc, exc.__traceback__)
                self.msg_q.put((M_CHALLENGE_INIT_ERR, repr(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_challenge_init_ok(self, init: dict[str, Any]) -> None:
        actions = init.get("actions") or []
        labels = [ACTION_LABELS.get(a, a) for a in actions]
        text = "Active challenge: " + ", затем ".join(labels)
        self._set_overlay(text, "#4a7de2")
        self.ch_status.configure(text=text, foreground="#4a7de2")
        self.btn_ch_done.configure(state="normal")
        self.btn_ch_cancel.configure(state="normal")
        self._append_log(f"[challenge] действия: {actions}")
        # запускаем WS-стрим
        self.challenge = ChallengeSession(
            camera=self.camera,
            on_result=lambda r: self.msg_q.put((M_CHALLENGE_RESULT, r)),
            on_close=lambda c, r: self.msg_q.put((M_CHALLENGE_CLOSED, (c, r))),
            on_challenge=lambda m: self.msg_q.put((M_CHALLENGE_MSG, m)),
        )
        self.challenge.start(init)

    def _on_challenge_msg(self, msg: dict[str, Any]) -> None:
        # {type:"challenge", actions, deadline_ms} — сервер подтвердил старт.
        if msg.get("type") == "challenge":
            self._append_log(f"[challenge] стрим запущен (deadline={msg.get('deadline_ms')}мс)")

    def _on_challenge_done(self) -> None:
        if self.challenge:
            self.challenge.send_done()
            self.btn_ch_done.configure(state="disabled")
            self._append_log("[challenge] отправлено {cmd:done}, жду вердикт…")

    def _on_challenge_cancel(self) -> None:
        if not self.challenge:
            return
        self.challenge.send_cancel()
        # Штатная отмена: помечаем сессию закрытой, чтобы нормальный WS-close
        # после {cmd:cancel} не показался аварийной ошибкой.
        self.challenge = None
        self.btn_ch_done.configure(state="disabled")
        self.btn_ch_cancel.configure(state="disabled")
        self.ch_status.configure(text="отменено", foreground="#aaaaaa")
        self._set_overlay("", "#000000")
        self._release_busy()
        self._append_log("[challenge] отменено пользователем")

    def _on_challenge_result_main(self, msg: dict[str, Any]) -> None:
        is_live = bool(msg.get("is_live"))
        reason = msg.get("reason") or ("ok" if is_live else "below_threshold")
        self.btn_ch_done.configure(state="disabled")
        self.btn_ch_cancel.configure(state="disabled")
        # Сессия получила вердикт — помечаем завершённой, чтобы последующий
        # нормальный WS-close (сервер шлёт его после result) не сбросил busy
        # во время ещё летящего авто-verify.
        self.challenge = None
        if is_live and msg.get("liveness_token"):
            self.ch_status.configure(text="пройден! verify с active-токеном…", foreground="#3ec47e")
            self._append_log(f"[challenge] ОК (is_live, n_frames={msg.get('n_frames')}) — авто-verify active")
            self._auto_verify_active(msg["liveness_token"])
        else:
            self.ch_status.configure(text=f"не пройден: {reason}", foreground="#e2554a")
            self._set_overlay("Challenge не пройден — повторите", "#e2554a")
            self._append_log(f"[challenge] FAIL is_live=False reason={reason}")
            self._release_busy()

    def _on_challenge_closed_main(self, payload: tuple) -> None:
        # Аварийный разрыв WS БЕЗ вердикта (expired/bad token/конфликт). Если
        # сессия уже закрыта штатно (result/cancel) — self.challenge is None,
        # игнорируем этот нормальный close.
        if self.challenge is None:
            return
        code, reason = payload
        self.challenge = None
        self.btn_ch_done.configure(state="disabled")
        self.btn_ch_cancel.configure(state="disabled")
        self.ch_status.configure(text=ws_close_reason(code), foreground="#e2554a")
        self._set_overlay(ws_close_reason(code), "#e2554a")
        self._append_log(f"[challenge] закрыто: {ws_close_reason(code)} ({reason})")
        self._release_busy()

    def _auto_verify_active(self, token: str) -> None:
        """Авто-переход: verify_base64 с liveness_mode=active + токен. Токен single-use,
        TTL 120с — уходит в запрос немедленно; после отправки остаётся только в локальной
        переменной worker-потока и собирается GC (не файл/env)."""
        user_id = self._numeric_user_id()
        b64 = self.camera.grab_jpeg_b64()
        if user_id is None or not b64:
            self._release_busy()
            self._append_log("[challenge] нет user_id/кадра для авто-verify")
            return

        def worker() -> None:
            try:
                resp = self.api.verify_base64(
                    user_id, b64, require_liveness=True,
                    liveness_mode="active", liveness_token=token,
                )
                self.msg_q.put((M_VERIFY, resp))
            except Exception as exc:  # любой сбой → UI получает error, busy не залипает
                _write_crash(type(exc), exc, exc.__traceback__)
                self.msg_q.put((M_VERIFY, {"error": repr(exc)}))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------- helpers -------------------------------

    def _numeric_user_id(self) -> str | None:
        """Серверный /upload_base64 делает int(user_id) — нужен целочисленный id.
        Возвращает строку-число или None (с логом причины)."""
        uid = self.user_var.get().strip()
        if not uid:
            self._append_log("[демо] укажите user_id (целое число)")
            return None
        if not uid.isdigit():
            self._append_log(
                f"[демо] user_id должен быть целым числом (сервер: int(user_id)). "
                f"Текущее значение не подходит: «{uid}»"
            )
            return None
        return uid

    def _guard_busy(self) -> bool:
        if not self.service_ready:
            self._append_log("[демо] сначала запустите сервис")
            return False
        if self.busy:
            self._append_log("[демо] операция уже выполняется…")
            return False
        self.busy = True
        self._set_actions_state()
        return True

    def _release_busy(self) -> None:
        """Сброс busy + обновление состояния кнопок. Единая точка: гарантирует, что
        после ЛЮБОГО завершения операции (успех/ошибка/отмена/ранний return) кнопки
        снова активны — не бывает залипшего disabled."""
        self.busy = False
        self._set_actions_state()

    def _set_actions_state(self) -> None:
        ready = self.service_ready and not self.busy
        self.btn_config.configure(state="normal" if self.service_ready else "disabled")
        # Действия (эталон/verify/challenge-старт) — по общему флагу ready рекурсивно
        # по дереву виджетов. Сервисные и challenge done/cancel исключены — ими
        # управляем явно в обработчиках.
        for widget in self.winfo_children():
            self._recursive_enable(widget, ready)
        # сервисные кнопки — вручную
        self.btn_start.configure(state="disabled" if self.service_ready else "normal")
        self.btn_stop.configure(state="normal" if self.service_ready else "disabled")

    def _recursive_enable(self, widget: tk.Widget, ready: bool) -> None:
        # Не трогаем сервисные кнопки и challenge done/cancel (управляются вручную).
        if widget in (self.btn_start, self.btn_stop, self.btn_ch_done, self.btn_ch_cancel):
            return
        if isinstance(widget, ttk.Button):
            widget.configure(state="normal" if ready else "disabled")
        for child in widget.winfo_children():
            self._recursive_enable(child, ready)

    def _set_badge(self, status: str, text: str) -> None:
        color = BADGE_COLORS.get(status, "#333333")
        self.badge.configure(text=text, background=color)

    @staticmethod
    def _set_bar(bar: ttk.Progressbar, value: Any) -> None:
        try:
            bar.configure(value=float(value) if value is not None else 0.0)
        except (TypeError, ValueError):
            bar.configure(value=0.0)

    def _set_overlay(self, text: str, color: str) -> None:
        self.overlay_var.set(text)
        self.overlay.configure(fg=color if text else "#000000")

    # ------------------------------- закрытие ------------------------------

    def _on_close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self._append_log("[закрытие] останавливаю сервис (down -v) и камеру…")
        if self.challenge:
            self.challenge.stop()
        self.camera.stop()
        # down -v в daemon-потоке — не блокирует выход (контейнеры получат команду).
        threading.Thread(target=self.orchestrator.down_v, daemon=True).start()
        self.after(800, self.destroy)


def main() -> None:
    sys.excepthook = _write_crash
    threading.excepthook = _threading_excepthook
    app = DesktopDemoApp()
    app.mainloop()


if __name__ == "__main__":
    main()