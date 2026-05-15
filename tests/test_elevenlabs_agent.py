from __future__ import annotations
# ruff: noqa: D101,D102,D103,D107
import argparse
import threading

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from reachy_eleven_agent.elevenlabs_agent import (
    ReachyElevenTools,
    ElevenLabsAgentSettings,
    ReachyMediaAudioInterface,
    _float_audio_to_pcm16_bytes,
    _pcm16_bytes_to_float_audio,
)
from reachy_eleven_agent.tools.core_tools import ToolDependencies


class FakeMedia:
    def __init__(self) -> None:
        self.samples = []
        self.pushed = []
        self.recording_started = 0
        self.playing_started = 0
        self.recording_stopped = 0
        self.playing_stopped = 0

    def start_recording(self) -> None:
        self.recording_started += 1

    def start_playing(self) -> None:
        self.playing_started += 1

    def stop_recording(self) -> None:
        self.recording_stopped += 1

    def stop_playing(self) -> None:
        self.playing_stopped += 1

    def get_input_audio_samplerate(self) -> int:
        return 16000

    def get_output_audio_samplerate(self) -> int:
        return 16000

    def get_audio_sample(self):
        if self.samples:
            return self.samples.pop(0)
        return None

    def push_audio_sample(self, audio_frame) -> None:
        self.pushed.append(audio_frame)


class FakeMovementManager:
    def __init__(self) -> None:
        self.moves = []
        self.cleared = 0
        self.paused = None
        self.speech_offsets = None
        self.listening = None

    def queue_move(self, move) -> None:
        self.moves.append(move)

    def clear_move_queue(self) -> None:
        self.cleared += 1

    def set_moving_state(self, duration: float) -> None:
        self.duration = duration

    def set_control_paused(self, paused: bool) -> None:
        self.paused = paused

    def set_speech_offsets(self, offsets) -> None:
        self.speech_offsets = offsets

    def set_listening(self, listening: bool) -> None:
        self.listening = listening


class FakeRobot:
    def __init__(self) -> None:
        self.sleep_calls = 0
        self.wake_calls = 0
        self.media = FakeMedia()

    def get_current_head_pose(self):
        return create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)

    def get_current_joint_positions(self):
        return [0.0], [0.0, 0.0]

    def goto_sleep(self) -> None:
        self.sleep_calls += 1

    def wake_up(self) -> None:
        self.wake_calls += 1


@pytest.fixture
def tool_context():
    robot = FakeRobot()
    movement = FakeMovementManager()
    deps = ToolDependencies(reachy_mini=robot, movement_manager=movement)
    return robot, movement, ReachyElevenTools(deps)


def test_settings_require_agent_id(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_AGENT_ID", raising=False)
    args = argparse.Namespace(eleven_agent_id=None, eleven_api_key=None, eleven_user_id=None, no_speech_motion=False)

    with pytest.raises(ValueError, match="ELEVENLABS_AGENT_ID"):
        ElevenLabsAgentSettings.from_args(args)


def test_move_head_clamps_values_and_queues_move(tool_context):
    _, movement, tools = tool_context

    result = tools.reachy_move_head({"roll_deg": 99, "pitch_deg": -99, "yaw_deg": 120, "duration_s": 99})

    assert "roll=40" in result
    assert "pitch=-40" in result
    assert "yaw=65" in result
    assert movement.cleared == 1
    assert len(movement.moves) == 1
    assert movement.moves[0].duration == 4.0


def test_sleep_pauses_movement_loop(tool_context):
    robot, movement, tools = tool_context

    result = tools.reachy_sleep({})

    assert result == "Sleep command sent."
    assert movement.cleared == 1
    assert movement.paused is True
    assert movement.speech_offsets == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert robot.sleep_calls == 1


def test_reachy_media_audio_converts_output_to_float_audio():
    robot = FakeRobot()
    audio = np.array([0, 32767, -32768], dtype=np.int16).tobytes()
    interface = ReachyMediaAudioInterface(robot, head_wobbler=None)

    interface.output(audio)

    assert len(robot.media.pushed) == 1
    np.testing.assert_allclose(
        robot.media.pushed[0],
        np.array([0.0, 32767 / 32768, -1.0], dtype=np.float32),
        rtol=1e-6,
    )


def test_reachy_media_audio_stop_is_safe_from_input_callback():
    robot = FakeRobot()
    robot.media.samples.append(np.ones((160, 1), dtype=np.float32) * 0.25)
    interface = ReachyMediaAudioInterface(robot, head_wobbler=None)
    callback_called = threading.Event()
    chunks = []

    def callback(audio: bytes) -> None:
        chunks.append(audio)
        callback_called.set()
        interface.stop()

    interface.start(callback)

    assert callback_called.wait(timeout=1.0)
    interface.stop()
    assert chunks
    assert robot.media.recording_started == 1
    assert robot.media.playing_started == 1
    assert robot.media.recording_stopped >= 1
    assert robot.media.playing_stopped >= 1


def test_audio_conversion_resamples_and_mixes_to_mono():
    stereo = np.column_stack(
        [
            np.linspace(-1.0, 1.0, num=1600, dtype=np.float32),
            np.linspace(1.0, -1.0, num=1600, dtype=np.float32),
        ]
    )

    pcm = _float_audio_to_pcm16_bytes(stereo, input_sample_rate=16000, output_sample_rate=8000)
    restored = _pcm16_bytes_to_float_audio(pcm, input_sample_rate=8000, output_sample_rate=16000)

    assert len(pcm) == 800 * 2
    assert restored.shape == (1600,)
    np.testing.assert_allclose(restored, np.zeros(1600, dtype=np.float32), atol=1e-4)
