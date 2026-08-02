from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import (
    Frame,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    WebSocket,
    sync_playwright,
)


APP_URL = os.environ.get(
    "APP_URL",
    "https://f8d5dqg4evu9tvthrrffkl.streamlit.app/",
).strip()
ARTIFACT_DIR = Path(os.environ.get("KEEPALIVE_ARTIFACT_DIR", "keepalive-artifacts"))
READY_SELECTORS = (
    '[data-testid="stAppViewContainer"]',
    '[data-testid="stApp"]',
    "div.stApp",
)
SLEEP_PATTERN = re.compile(
    r"gone to sleep|app has gone to sleep|zzz|wake (?:this|the) app",
    flags=re.IGNORECASE,
)


def _frame_body_text(frame: Frame) -> str:
    try:
        return frame.locator("body").inner_text(timeout=5_000)
    except Exception:
        return ""


def _all_frame_text(page: Page, limit: int = 4_000) -> str:
    sections: list[str] = []
    for index, frame in enumerate(page.frames):
        text = _frame_body_text(frame).strip()
        if text:
            sections.append(f"[frame {index}] {frame.url}\n{text}")
    return "\n\n".join(sections)[:limit]


def _save_diagnostics(page: Page, label: str) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        page.screenshot(
            path=str(ARTIFACT_DIR / f"{label}.png"),
            full_page=True,
            timeout=20_000,
        )
    except Exception as exc:
        print(f"No se pudo guardar la captura: {exc}")

    try:
        (ARTIFACT_DIR / f"{label}-top.html").write_text(
            page.content(),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"No se pudo guardar el HTML principal: {exc}")

    frame_report: list[str] = []
    for index, frame in enumerate(page.frames):
        frame_report.append(
            f"FRAME {index}\nURL: {frame.url}\n"
            f"TEXTO:\n{_frame_body_text(frame)[:3_000]}\n"
        )
        try:
            (ARTIFACT_DIR / f"{label}-frame-{index}.html").write_text(
                frame.content(),
                encoding="utf-8",
            )
        except Exception as exc:
            frame_report.append(f"No se pudo guardar HTML: {exc}\n")

    try:
        (ARTIFACT_DIR / f"{label}-frames.txt").write_text(
            "\n".join(frame_report),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"No se pudo guardar el informe de frames: {exc}")


def _wake_sleeping_app(page: Page) -> bool:
    for frame in page.frames:
        body = _frame_body_text(frame)
        if not SLEEP_PATTERN.search(body):
            continue

        print(f"Se detectó la pantalla de hibernación en: {frame.url}")
        candidates = (
            frame.get_by_role(
                "button",
                name=re.compile(
                    r"get this app back up|wake|yes|reactivate",
                    re.IGNORECASE,
                ),
            ),
            frame.get_by_role(
                "link",
                name=re.compile(
                    r"get this app back up|wake|yes|reactivate",
                    re.IGNORECASE,
                ),
            ),
            frame.get_by_text(
                re.compile(
                    r"yes,? get this app back up|wake (?:this|the) app|reactivate",
                    re.IGNORECASE,
                )
            ),
        )

        for locator in candidates:
            try:
                if locator.count() > 0 and locator.first.is_visible(timeout=2_000):
                    locator.first.click(timeout=10_000)
                    print("Se solicitó reactivar la aplicación.")
                    return True
            except Exception:
                continue

        print(
            "La pantalla de hibernación fue detectada, pero no se encontró "
            "un control de reactivación visible."
        )

    return False


def _find_streamlit_interface(page: Page) -> tuple[Frame, str] | None:
    for frame in page.frames:
        for selector in READY_SELECTORS:
            try:
                if frame.locator(selector).count() > 0:
                    return frame, selector
            except Exception:
                continue
    return None


def _wait_until_ready(
    page: Page,
    stream_state: dict[str, Any],
    timeout_seconds: int = 240,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    last_frame_signature: tuple[str, ...] = ()
    stream_stable_since: float | None = None
    reload_done = False

    while time.monotonic() < deadline:
        frame_signature = tuple(frame.url for frame in page.frames)
        if frame_signature != last_frame_signature:
            print(f"Frames detectados ({len(frame_signature)}):")
            for index, url in enumerate(frame_signature):
                print(f"  [{index}] {url or 'about:blank'}")
            last_frame_signature = frame_signature

        _wake_sleeping_app(page)

        interface = _find_streamlit_interface(page)
        if interface is not None:
            frame, selector = interface
            body = _frame_body_text(frame)
            if not SLEEP_PATTERN.search(body):
                print(
                    "Interfaz Streamlit detectada en "
                    f"{frame.url or 'frame sin URL'} mediante {selector}."
                )
                return True

        if int(stream_state.get("active", 0)) > 0:
            if stream_stable_since is None:
                stream_stable_since = time.monotonic()
                print("Conexión WebSocket _stcore/stream activa; verificando estabilidad...")
            elif time.monotonic() - stream_stable_since >= 15:
                print("La sesión WebSocket de Streamlit permaneció activa durante 15 segundos.")
                return True
        else:
            stream_stable_since = None

        remaining = deadline - time.monotonic()
        if not reload_done and remaining < timeout_seconds / 2:
            print("La interfaz aún no apareció; recargando una vez la página...")
            try:
                page.reload(wait_until="domcontentloaded", timeout=120_000)
            except Exception as exc:
                print(f"La recarga no finalizó normalmente: {exc}")
            reload_done = True

        page.wait_for_timeout(3_000)

    return False


def main() -> int:
    if not APP_URL.startswith(("https://", "http://")):
        print(f"APP_URL inválida: {APP_URL}", file=sys.stderr)
        return 2

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Abriendo la aplicación con Chromium: {APP_URL}")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--disable-dev-shm-usage", "--no-sandbox"],
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000},
            locale="es-AR",
        )
        page = context.new_page()
        page.set_default_timeout(20_000)

        stream_state: dict[str, Any] = {
            "active": 0,
            "ever_opened": False,
        }

        def handle_websocket(websocket: WebSocket) -> None:
            print(f"[websocket] abierta: {websocket.url}")
            if "_stcore/stream" not in websocket.url:
                return

            stream_state["active"] = int(stream_state["active"]) + 1
            stream_state["ever_opened"] = True

            def handle_close() -> None:
                stream_state["active"] = max(
                    0,
                    int(stream_state["active"]) - 1,
                )
                print(f"[websocket] cerrada: {websocket.url}")

            websocket.on("close", handle_close)
            websocket.on(
                "socketerror",
                lambda error: print(
                    f"[websocket:error] {websocket.url}: {error}"
                ),
            )

        page.on(
            "console",
            lambda message: print(f"[browser:{message.type}] {message.text}"),
        )
        page.on(
            "pageerror",
            lambda error: print(f"[browser:error] {error}"),
        )
        page.on("websocket", handle_websocket)

        try:
            response = page.goto(
                APP_URL,
                wait_until="domcontentloaded",
                timeout=120_000,
            )
            if response is not None:
                print(f"Respuesta inicial del navegador: HTTP {response.status}")

            ready = _wait_until_ready(page, stream_state)
            if not ready:
                print("La interfaz o sesión de Streamlit no quedó disponible.")
                print(f"URL final: {page.url}")
                print("Frames finales:")
                for index, frame in enumerate(page.frames):
                    print(f"  [{index}] {frame.url or 'about:blank'}")
                print("Contenido visible en todos los frames:")
                print(_all_frame_text(page) or "(sin texto visible)")
                print(
                    "WebSocket Streamlit abierta alguna vez: "
                    f"{bool(stream_state['ever_opened'])}"
                )
                _save_diagnostics(page, "keepalive-error")
                return 1

            print("La aplicación está activa y la sesión de Streamlit fue iniciada.")
            print("Manteniendo el navegador conectado durante 60 segundos...")
            page.wait_for_timeout(60_000)
            _save_diagnostics(page, "keepalive-success")
            return 0

        except PlaywrightTimeoutError as exc:
            print(f"Tiempo de espera agotado: {exc}")
            _save_diagnostics(page, "keepalive-timeout")
            return 1
        except Exception as exc:
            print(f"Error inesperado: {exc}")
            _save_diagnostics(page, "keepalive-error")
            return 1
        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    raise SystemExit(main())
