"""Dashboard settings and runtime controls for the ElevenLabs app mode."""

from __future__ import annotations
import os
import copy
import logging
import argparse
import threading
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable


logger = logging.getLogger(__name__)


ENV_KEYS = (
    "REACHY_AGENT_PROVIDER",
    "ELEVENLABS_AGENT_ID",
    "ELEVENLABS_API_KEY",
    "REACHY_ELEVEN_USER_ID",
    "REACHY_ELEVEN_MEDIA_BACKEND",
)


@dataclass(frozen=True)
class ElevenLabsDashboardSettings:
    """Settings shown and edited through the Reachy Mini dashboard UI."""

    agent_id: str = ""
    api_key_configured: bool = False
    user_id: str = ""


@dataclass(frozen=True)
class ElevenLabsRuntimeStatus:
    """Small runtime snapshot for the settings UI."""

    state: str
    detail: str
    agent_id: str
    api_key_configured: bool
    user_id: str


def _env_path(instance_path: str | Path | None) -> Path | None:
    if instance_path is None:
        return None
    return Path(instance_path) / ".env"


def _parse_env_lines(lines: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw_value = stripped.partition("=")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _read_env_values(env_path: Path | None) -> dict[str, str]:
    if env_path is None or not env_path.exists():
        return {}
    try:
        return _parse_env_lines(env_path.read_text(encoding="utf-8").splitlines())
    except Exception:
        logger.warning("Failed to read dashboard settings from %s", env_path, exc_info=True)
        return {}


def _load_env_template(env_path: Path | None) -> list[str]:
    candidates: list[Path] = []
    if env_path is not None:
        candidates.append(env_path)
        candidates.append(env_path.with_name(".env.example"))
    candidates.append(Path.cwd() / ".env.example")
    candidates.append(Path(__file__).resolve().parents[2] / ".env.example")

    for candidate in candidates:
        if candidate.exists():
            try:
                return candidate.read_text(encoding="utf-8").splitlines()
            except Exception:
                logger.debug("Could not load env template from %s", candidate, exc_info=True)
    return []


def _write_env_values(env_path: Path | None, values: dict[str, str | None]) -> None:
    if env_path is None:
        return

    lines = _load_env_template(env_path)
    seen: set[str] = set()
    result: list[str] = []

    for line in lines:
        stripped = line.strip()
        key = stripped.partition("=")[0].strip() if "=" in stripped and not stripped.startswith("#") else None
        if key in values:
            seen.add(key)
            value = values[key]
            if value is None:
                continue
            result.append(f"{key}={value}")
        else:
            result.append(line)

    for key in ENV_KEYS:
        if key in values and key not in seen:
            value = values[key]
            if value is not None:
                result.append(f"{key}={value}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(result).rstrip() + "\n", encoding="utf-8")


class ElevenLabsDashboardRuntime:
    """Own the dashboard-controlled ElevenLabs conversation lifecycle."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        deps: Any,
        movement_manager: Any,
        head_wobbler: Any,
        camera_worker: Any | None,
        vision_manager: Any | None,
        app_stop_event: threading.Event,
        instance_path: str | Path | None,
        conversation_runner: Callable[[Any, Any, threading.Event], None],
    ) -> None:
        """Initialize the supervisor with already-created app dependencies."""
        self._args = args
        self._deps = deps
        self._movement_manager = movement_manager
        self._head_wobbler = head_wobbler
        self._camera_worker = camera_worker
        self._vision_manager = vision_manager
        self._app_stop_event = app_stop_event
        self._env_path = _env_path(instance_path)
        self._conversation_runner = conversation_runner

        self._lock = threading.RLock()
        self._conversation_stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._state = "idle"
        self._detail = "Configure ElevenLabs, then start the conversation."
        self._last_error = ""

        self._load_saved_settings_into_process()

    def _load_saved_settings_into_process(self) -> None:
        saved = _read_env_values(self._env_path)
        for key in ENV_KEYS:
            value = saved.get(key)
            if value is not None:
                os.environ[key] = value

        self._args.provider = "elevenlabs"
        self._args.eleven_agent_id = os.getenv("ELEVENLABS_AGENT_ID") or getattr(self._args, "eleven_agent_id", None)
        self._args.eleven_api_key = os.getenv("ELEVENLABS_API_KEY") or getattr(self._args, "eleven_api_key", None)
        self._args.eleven_user_id = os.getenv("REACHY_ELEVEN_USER_ID") or getattr(self._args, "eleven_user_id", None)

    def settings(self) -> ElevenLabsDashboardSettings:
        """Return the current persisted/live settings snapshot."""
        saved = _read_env_values(self._env_path)
        agent_id = (
            saved.get("ELEVENLABS_AGENT_ID")
            or os.getenv("ELEVENLABS_AGENT_ID")
            or getattr(self._args, "eleven_agent_id", None)
            or ""
        )
        api_key = saved.get("ELEVENLABS_API_KEY") or os.getenv("ELEVENLABS_API_KEY") or getattr(
            self._args, "eleven_api_key", None
        )
        user_id = (
            saved.get("REACHY_ELEVEN_USER_ID")
            or os.getenv("REACHY_ELEVEN_USER_ID")
            or getattr(self._args, "eleven_user_id", None)
            or ""
        )
        return ElevenLabsDashboardSettings(
            agent_id=str(agent_id).strip(),
            api_key_configured=bool(str(api_key or "").strip()),
            user_id=str(user_id).strip(),
        )

    def save_settings(
        self,
        agent_id: str,
        api_key: str = "",
        user_id: str = "",
        clear_api_key: bool = False,
    ) -> ElevenLabsRuntimeStatus:
        """Persist dashboard settings and update the running process environment."""
        clean_agent_id = (agent_id or "").strip()
        clean_api_key = (api_key or "").strip()
        clean_user_id = (user_id or "").strip()

        if not clean_agent_id:
            with self._lock:
                self._state = "needs_config"
                self._detail = "ElevenLabs agent ID is required."
            return self.status()

        values: dict[str, str | None] = {
            "REACHY_AGENT_PROVIDER": "elevenlabs",
            "ELEVENLABS_AGENT_ID": clean_agent_id,
            "REACHY_ELEVEN_USER_ID": clean_user_id or None,
            "REACHY_ELEVEN_MEDIA_BACKEND": "no_media",
        }
        if clear_api_key:
            values["ELEVENLABS_API_KEY"] = None
        elif clean_api_key:
            values["ELEVENLABS_API_KEY"] = clean_api_key

        _write_env_values(self._env_path, values)

        os.environ["REACHY_AGENT_PROVIDER"] = "elevenlabs"
        os.environ["ELEVENLABS_AGENT_ID"] = clean_agent_id
        os.environ["REACHY_ELEVEN_MEDIA_BACKEND"] = "no_media"
        if clean_user_id:
            os.environ["REACHY_ELEVEN_USER_ID"] = clean_user_id
        else:
            os.environ.pop("REACHY_ELEVEN_USER_ID", None)
        if clear_api_key:
            os.environ.pop("ELEVENLABS_API_KEY", None)
            self._args.eleven_api_key = None
        elif clean_api_key:
            os.environ["ELEVENLABS_API_KEY"] = clean_api_key
            self._args.eleven_api_key = clean_api_key

        self._args.provider = "elevenlabs"
        self._args.eleven_agent_id = clean_agent_id
        self._args.eleven_user_id = clean_user_id or None

        with self._lock:
            if self._state in {"idle", "needs_config"}:
                self._state = "idle"
            self._detail = "Settings saved. Start the conversation when ready."
            self._last_error = ""

        return self.status()

    def status(self) -> ElevenLabsRuntimeStatus:
        """Return a UI-friendly runtime status snapshot."""
        settings = self.settings()
        with self._lock:
            state = self._state
            detail = self._detail
            if state != "running" and not settings.agent_id:
                state = "needs_config"
                detail = "Enter an ElevenLabs agent ID to enable Start."
        return ElevenLabsRuntimeStatus(
            state=state,
            detail=detail,
            agent_id=settings.agent_id,
            api_key_configured=settings.api_key_configured,
            user_id=settings.user_id,
        )

    def start(self) -> ElevenLabsRuntimeStatus:
        """Start the conversation thread if configured and not already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._state = "running"
                self._detail = "Conversation is already running."
                return self.status()

            settings = self.settings()
            if not settings.agent_id:
                self._state = "needs_config"
                self._detail = "Enter an ElevenLabs agent ID before starting."
                return self.status()

            stop_event = threading.Event()
            self._conversation_stop_event = stop_event
            self._state = "starting"
            self._detail = "Starting ElevenLabs conversation..."
            self._last_error = ""
            runtime_args = copy.copy(self._args)
            runtime_args.provider = "elevenlabs"
            runtime_args.eleven_agent_id = settings.agent_id
            runtime_args.eleven_api_key = os.getenv("ELEVENLABS_API_KEY") or None
            runtime_args.eleven_user_id = settings.user_id or None

            self._thread = threading.Thread(
                target=self._run_conversation,
                args=(runtime_args, stop_event),
                name="reachy-eleven-dashboard-conversation",
                daemon=True,
            )
            self._thread.start()
            return self.status()

    def stop(self, timeout_s: float = 8.0) -> ElevenLabsRuntimeStatus:
        """Stop the conversation thread if it is active."""
        thread: threading.Thread | None
        with self._lock:
            thread = self._thread
            event = self._conversation_stop_event
            if thread is None or not thread.is_alive() or event is None:
                self._state = "idle" if self.settings().agent_id else "needs_config"
                self._detail = "Conversation is not running."
                return self.status()
            self._state = "stopping"
            self._detail = "Stopping conversation..."
            event.set()

        thread.join(timeout=timeout_s)
        if thread.is_alive():
            with self._lock:
                self._state = "stopping"
                self._detail = "Stop requested; waiting for ElevenLabs session to close."
        return self.status()

    def restart(self) -> ElevenLabsRuntimeStatus:
        """Restart the conversation with the current saved settings."""
        self.stop()
        return self.start()

    def close(self) -> None:
        """Stop the conversation during app shutdown."""
        self.stop(timeout_s=10.0)

    def _run_conversation(self, runtime_args: argparse.Namespace, stop_event: threading.Event) -> None:
        resources_started = False
        try:
            self._movement_manager.start()
            self._head_wobbler.start()
            if self._camera_worker is not None:
                self._camera_worker.start()
            if self._vision_manager is not None:
                self._vision_manager.start()
            resources_started = True

            with self._lock:
                self._state = "running"
                self._detail = "Conversation is running."

            self._conversation_runner(runtime_args, self._deps, stop_event)
        except Exception as exc:
            logger.exception("ElevenLabs dashboard conversation failed")
            with self._lock:
                self._state = "error"
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._detail = self._last_error
        finally:
            stop_event.set()
            if resources_started:
                self._stop_resources()
            with self._lock:
                if self._state not in {"error"}:
                    self._state = "idle" if self.settings().agent_id else "needs_config"
                    self._detail = "Conversation stopped."
                if self._conversation_stop_event is stop_event:
                    self._conversation_stop_event = None
                if self._thread is threading.current_thread():
                    self._thread = None

    def _stop_resources(self) -> None:
        try:
            self._movement_manager.stop()
        except Exception:
            logger.debug("Error stopping movement manager", exc_info=True)
        try:
            self._head_wobbler.stop()
        except Exception:
            logger.debug("Error stopping head wobbler", exc_info=True)
        if self._camera_worker is not None:
            try:
                self._camera_worker.stop()
            except Exception:
                logger.debug("Error stopping camera worker", exc_info=True)
        if self._vision_manager is not None:
            try:
                self._vision_manager.stop()
            except Exception:
                logger.debug("Error stopping vision manager", exc_info=True)


def _format_status(status: ElevenLabsRuntimeStatus) -> str:
    key_status = "configured" if status.api_key_configured else "not set"
    return (
        f"State: {status.state}\n"
        f"Details: {status.detail}\n"
        f"Agent ID: {status.agent_id or 'not set'}\n"
        f"Private API key: {key_status}"
    )


def _remove_autogenerated_root_route(settings_app: Any) -> None:
    """Remove the template static root so Gradio can own '/'.

    ReachyMiniApp registers a package static index at '/' before app-specific
    run() code executes. This app uses that settings server for an ElevenLabs
    Gradio control panel, so the earlier static route must be removed or users
    see the stale OpenAI setup page instead.
    """
    router = getattr(settings_app, "router", None)
    routes = getattr(router, "routes", None)
    if routes is None:
        return

    routes[:] = [
        route
        for route in routes
        if not (
            getattr(route, "path", None) == "/"
            and "GET" in (getattr(route, "methods", set()) or set())
        )
    ]


def mount_elevenlabs_dashboard(settings_app: Any, runtime: ElevenLabsDashboardRuntime) -> None:
    """Mount the Gradio dashboard UI on the Reachy Mini app settings server."""
    import gradio as gr

    _remove_autogenerated_root_route(settings_app)

    def load_values() -> tuple[str, str, str, bool, str]:
        status = runtime.status()
        return status.agent_id, "", status.user_id, False, _format_status(status)

    def save(agent_id: str, api_key: str, user_id: str, clear_api_key: bool) -> tuple[str, str, str, bool, str]:
        status = runtime.save_settings(agent_id, api_key, user_id, clear_api_key)
        return status.agent_id, "", status.user_id, False, _format_status(status)

    def start() -> str:
        return _format_status(runtime.start())

    def stop() -> str:
        return _format_status(runtime.stop())

    def restart() -> str:
        return _format_status(runtime.restart())

    def refresh() -> str:
        return _format_status(runtime.status())

    with gr.Blocks(title="Reachy Eleven Agent", fill_width=True) as dashboard:
        gr.Markdown("# Reachy Eleven Agent")
        gr.Markdown("Configure ElevenLabs, then start or stop the voice conversation from this page.")
        with gr.Row():
            with gr.Column(scale=2):
                agent_id = gr.Textbox(label="ElevenLabs Agent ID", placeholder="agent_...", autofocus=True)
                api_key = gr.Textbox(
                    label="ElevenLabs API Key (optional)",
                    type="password",
                    placeholder="Leave blank to keep the current saved key",
                )
                user_id = gr.Textbox(label="Conversation User ID (optional)", placeholder="Optional history user id")
                clear_api_key = gr.Checkbox(label="Clear saved API key", value=False)
                with gr.Row():
                    save_button = gr.Button("Save settings", variant="primary")
                    refresh_button = gr.Button("Refresh status")
            with gr.Column(scale=1):
                status_box = gr.Textbox(label="Status", lines=7, interactive=False)
                with gr.Row():
                    start_button = gr.Button("Start", variant="primary")
                    stop_button = gr.Button("Stop", variant="stop")
                restart_button = gr.Button("Restart conversation")

        dashboard.load(
            load_values,
            outputs=[agent_id, api_key, user_id, clear_api_key, status_box],
            queue=False,
            show_progress="hidden",
        )
        save_button.click(
            save,
            inputs=[agent_id, api_key, user_id, clear_api_key],
            outputs=[agent_id, api_key, user_id, clear_api_key, status_box],
            queue=False,
        )
        start_button.click(start, outputs=[status_box], queue=False)
        stop_button.click(stop, outputs=[status_box], queue=False)
        restart_button.click(restart, outputs=[status_box], queue=False)
        refresh_button.click(refresh, outputs=[status_box], queue=False)

    gr.mount_gradio_app(settings_app, dashboard, path="/")
