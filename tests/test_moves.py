from __future__ import annotations

# ruff: noqa: D101,D102,D103,D107
from reachy_mini.utils import create_head_pose
from reachy_eleven_agent.moves import MovementManager


class DisconnectedRobot:
    def __init__(self) -> None:
        self.calls = 0

    def set_target(self, **_kwargs) -> None:
        self.calls += 1
        raise ConnectionError("Lost connection with the server.")


def test_movement_manager_pauses_after_connection_loss():
    robot = DisconnectedRobot()
    errors = []
    manager = MovementManager(robot, on_connection_lost=errors.append)

    manager._issue_control_command(create_head_pose(0, 0, 0, 0, 0, 0, degrees=True), (0.0, 0.0), 0.0)
    manager._issue_control_command(create_head_pose(0, 0, 0, 0, 0, 0, degrees=True), (0.0, 0.0), 0.0)

    status = manager.get_status()
    assert robot.calls == 1
    assert status["connection_lost"] is True
    assert status["control_paused"] is True
    assert len(errors) == 1
