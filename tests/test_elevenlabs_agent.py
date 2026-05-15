from __future__ import annotations

import argparse

import pytest

from reachy_mini.utils import create_head_pose

from reachy_eleven_agent.tools.core_tools import ToolDependencies
from reachy_eleven_agent.elevenlabs_agent import ElevenLabsAgentSettings, ReachyElevenTools


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
