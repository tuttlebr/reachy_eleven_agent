from __future__ import annotations
# ruff: noqa: D101,D102,D103,D107
import time
import argparse
import threading

from reachy_eleven_agent.elevenlabs_dashboard import ElevenLabsDashboardRuntime


class FakeService:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        provider="elevenlabs",
        eleven_agent_id=None,
        eleven_api_key=None,
        eleven_user_id=None,
        no_speech_motion=False,
    )


def make_runtime(tmp_path, runner=None) -> ElevenLabsDashboardRuntime:
    if runner is None:
        def runner(_args, _deps, stop_event) -> None:
            stop_event.wait(0.1)

    service = FakeService()
    return ElevenLabsDashboardRuntime(
        args=make_args(),
        deps=object(),
        movement_manager=service,
        head_wobbler=FakeService(),
        camera_worker=None,
        vision_manager=None,
        app_stop_event=threading.Event(),
        instance_path=tmp_path,
        conversation_runner=runner,
    )


def wait_for(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_dashboard_settings_persist_to_instance_env(tmp_path, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_AGENT_ID", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("REACHY_ELEVEN_USER_ID", raising=False)
    runtime = make_runtime(tmp_path)

    status = runtime.save_settings("agent_abc", "secret", "user-1")

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "REACHY_AGENT_PROVIDER=elevenlabs" in env_text
    assert "ELEVENLABS_AGENT_ID=agent_abc" in env_text
    assert "ELEVENLABS_API_KEY=secret" in env_text
    assert "REACHY_ELEVEN_USER_ID=user-1" in env_text
    assert status.agent_id == "agent_abc"
    assert status.api_key_configured is True


def test_dashboard_settings_can_clear_saved_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "old-secret")
    runtime = make_runtime(tmp_path)
    runtime.save_settings("agent_abc", "secret", "user-1")

    status = runtime.save_settings("agent_abc", "", "", clear_api_key=True)

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "ELEVENLABS_API_KEY=" not in env_text
    assert "REACHY_ELEVEN_USER_ID=" not in env_text
    assert status.api_key_configured is False


def test_dashboard_start_requires_agent_id(tmp_path, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_AGENT_ID", raising=False)
    runtime = make_runtime(tmp_path)

    status = runtime.start()

    assert status.state == "needs_config"
    assert "agent ID" in status.detail


def test_dashboard_runtime_start_stop_lifecycle(tmp_path):
    runner_started = threading.Event()
    runner_stopped = threading.Event()
    seen_agent_ids = []

    def runner(args, _deps, stop_event) -> None:
        seen_agent_ids.append(args.eleven_agent_id)
        runner_started.set()
        stop_event.wait(2.0)
        runner_stopped.set()

    movement = FakeService()
    head_wobbler = FakeService()
    runtime = ElevenLabsDashboardRuntime(
        args=make_args(),
        deps=object(),
        movement_manager=movement,
        head_wobbler=head_wobbler,
        camera_worker=None,
        vision_manager=None,
        app_stop_event=threading.Event(),
        instance_path=tmp_path,
        conversation_runner=runner,
    )
    runtime.save_settings("agent_abc")

    start_status = runtime.start()
    assert start_status.state in {"starting", "running"}
    assert runner_started.wait(timeout=1.0)
    assert wait_for(lambda: runtime.status().state == "running")

    stop_status = runtime.stop()
    assert runner_stopped.wait(timeout=1.0)
    assert wait_for(lambda: runtime.status().state == "idle")
    assert stop_status.state in {"stopping", "idle"}
    assert seen_agent_ids == ["agent_abc"]
    assert movement.started == 1
    assert movement.stopped == 1
    assert head_wobbler.started == 1
    assert head_wobbler.stopped == 1
