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


class ConnectedRobot:
    def __init__(self) -> None:
        self.calls = 0

    def set_target(self, **_kwargs) -> None:
        self.calls += 1

    def get_current_joint_positions(self):
        return [0.0], [0.1, -0.1]

    def get_current_head_pose(self):
        return create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)


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


def test_movement_manager_can_resume_with_replacement_robot():
    robot = DisconnectedRobot()
    errors = []
    manager = MovementManager(robot, on_connection_lost=errors.append)

    manager._issue_control_command(create_head_pose(0, 0, 0, 0, 0, 0, degrees=True), (0.0, 0.0), 0.0)

    replacement = ConnectedRobot()
    manager.replace_robot(replacement)
    manager._poll_signals(manager._now())
    manager._issue_control_command(create_head_pose(0, 0, 0, 0, 0, 0, degrees=True), (0.0, 0.0), 0.0)

    status = manager.get_status()
    assert status["connection_lost"] is False
    assert status["control_paused"] is False
    assert replacement.calls == 1
    assert len(errors) == 1
