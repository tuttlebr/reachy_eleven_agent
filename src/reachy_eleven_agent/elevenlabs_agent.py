"""ElevenLabs ElevenAgents integration for Reachy Mini."""

from __future__ import annotations

import os
import time
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import linear_pose_interpolation

from reachy_eleven_agent.tools.core_tools import ToolDependencies
from reachy_eleven_agent.audio.head_wobbler import HeadWobbler


logger = logging.getLogger(__name__)

ELEVENLABS_PCM_SAMPLE_RATE = 16000


class ElevenGotoMove(Move):  # type: ignore[misc]
    """Small goto move used by ElevenLabs client tools."""

    def __init__(
        self,
        target_head_pose: NDArray[np.float32],
        start_head_pose: NDArray[np.float32],
        target_antennas: tuple[float, float],
        start_antennas: tuple[float, float],
        target_body_yaw: float,
        start_body_yaw: float,
        duration: float,
    ) -> None:
        self.target_head_pose = target_head_pose
        self.start_head_pose = start_head_pose
        self.target_antennas = target_antennas
        self.start_antennas = start_antennas
        self.target_body_yaw = target_body_yaw
        self.start_body_yaw = start_body_yaw
        self._duration = duration

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self._duration

    def evaluate(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Evaluate interpolated head, antennas, and body yaw at time t."""
        ratio = max(0.0, min(1.0, t / self.duration))
        head = linear_pose_interpolation(self.start_head_pose, self.target_head_pose, ratio)
        antennas = np.array(
            [
                self.start_antennas[0] + (self.target_antennas[0] - self.start_antennas[0]) * ratio,
                self.start_antennas[1] + (self.target_antennas[1] - self.start_antennas[1]) * ratio,
            ],
            dtype=np.float64,
        )
        body_yaw = self.start_body_yaw + (self.target_body_yaw - self.start_body_yaw) * ratio
        return head, antennas, body_yaw


@dataclass(frozen=True)
class ElevenLabsAgentSettings:
    """Runtime settings for the ElevenLabs conversation."""

    agent_id: str
    api_key: str | None = None
    user_id: str | None = None
    speech_motion_enabled: bool = True
    keepalive_interval_s: float = 20.0
    reconnect_delay_s: float = 2.0

    @classmethod
    def from_args(cls, args: Any) -> "ElevenLabsAgentSettings":
        """Build settings from CLI args and environment variables."""
        agent_id = getattr(args, "eleven_agent_id", None) or os.getenv("ELEVENLABS_AGENT_ID")
        if not agent_id:
            raise ValueError(
                "Missing ElevenLabs agent ID. Set ELEVENLABS_AGENT_ID or pass --eleven-agent-id."
            )

        return cls(
            agent_id=agent_id,
            api_key=getattr(args, "eleven_api_key", None) or os.getenv("ELEVENLABS_API_KEY") or None,
            user_id=getattr(args, "eleven_user_id", None) or os.getenv("REACHY_ELEVEN_USER_ID") or None,
            speech_motion_enabled=not bool(getattr(args, "no_speech_motion", False)),
            keepalive_interval_s=_float_env("REACHY_ELEVEN_KEEPALIVE_INTERVAL_S", 20.0),
            reconnect_delay_s=_float_env("REACHY_ELEVEN_RECONNECT_DELAY_S", 2.0),
        )


class TappedDefaultAudioInterface:
    """Default ElevenLabs audio I/O with a tap for speech-synced robot motion."""

    def __init__(self, head_wobbler: HeadWobbler | None, speech_motion_enabled: bool = True) -> None:
        try:
            from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

            self._delegate = DefaultAudioInterface()
        except ImportError as exc:
            raise RuntimeError(
                "ElevenLabs live audio needs PyAudio. Install system PortAudio headers first "
                "(Debian/Ubuntu: sudo apt-get install portaudio19-dev libasound-dev), "
                "then install pyaudio or elevenlabs[pyaudio]."
            ) from exc

        self._head_wobbler = head_wobbler
        self._speech_motion_enabled = speech_motion_enabled

    def start(self, input_callback: Callable[[bytes], None]) -> None:
        """Start microphone input and speaker output."""
        self._delegate.start(input_callback)

    def stop(self) -> None:
        """Stop microphone input and speaker output."""
        self._delegate.stop()

    def output(self, audio: bytes) -> None:
        """Play agent audio and feed a copy into the head wobbler."""
        if self._speech_motion_enabled and self._head_wobbler is not None and audio:
            pcm = np.frombuffer(audio, dtype=np.int16).reshape(1, -1)
            self._head_wobbler.feed_pcm(pcm, ELEVENLABS_PCM_SAMPLE_RATE)
        self._delegate.output(audio)

    def interrupt(self) -> None:
        """Clear pending output and speech motion when the user interrupts."""
        if self._head_wobbler is not None:
            self._head_wobbler.reset()
        self._delegate.interrupt()


class ReachyElevenTools:
    """Client tools exposed to the ElevenLabs agent."""

    def __init__(self, deps: ToolDependencies) -> None:
        self.deps = deps

    def register_with(self, client_tools: Any) -> None:
        """Register all Reachy tools with ElevenLabs ClientTools."""
        client_tools.register("reachy_move_head", self.reachy_move_head)
        client_tools.register("reachy_set_antennas", self.reachy_set_antennas)
        client_tools.register("reachy_set_body_yaw", self.reachy_set_body_yaw)
        client_tools.register("reachy_expression", self.reachy_expression)
        client_tools.register("reachy_sleep", self.reachy_sleep)
        client_tools.register("reachy_wake", self.reachy_wake)

    def reachy_move_head(self, parameters: dict[str, Any]) -> str:
        """Move Reachy's head to an absolute roll/pitch/yaw pose in degrees."""
        roll = _number(parameters, "roll_deg", _direction_default(parameters, "roll_deg"), -40.0, 40.0)
        pitch = _number(parameters, "pitch_deg", _direction_default(parameters, "pitch_deg"), -40.0, 40.0)
        yaw = _number(parameters, "yaw_deg", _direction_default(parameters, "yaw_deg"), -65.0, 65.0)
        duration = _number(parameters, "duration_s", 0.8, 0.3, 4.0)
        self._queue_goto(roll, pitch, yaw, duration_s=duration)
        return f"Head move queued: roll={roll:.0f}, pitch={pitch:.0f}, yaw={yaw:.0f} degrees."

    def reachy_set_antennas(self, parameters: dict[str, Any]) -> str:
        """Move Reachy's right and left antennas to absolute degree positions."""
        right_deg = _number(parameters, "right_deg", 0.0, -175.0, 175.0)
        left_deg = _number(parameters, "left_deg", 0.0, -175.0, 175.0)
        duration = _number(parameters, "duration_s", 0.7, 0.3, 4.0)
        self._queue_goto(
            0.0,
            0.0,
            0.0,
            antennas=(np.deg2rad(right_deg), np.deg2rad(left_deg)),
            duration_s=duration,
            preserve_head=True,
        )
        return f"Antenna move queued: right={right_deg:.0f}, left={left_deg:.0f} degrees."

    def reachy_set_body_yaw(self, parameters: dict[str, Any]) -> str:
        """Rotate Reachy's body around its vertical axis."""
        yaw_deg = _number(parameters, "yaw_deg", 0.0, -160.0, 160.0)
        duration = _number(parameters, "duration_s", 1.0, 0.3, 5.0)
        self._queue_goto(
            0.0,
            0.0,
            0.0,
            body_yaw=np.deg2rad(yaw_deg),
            duration_s=duration,
            preserve_head=True,
            preserve_antennas=True,
        )
        return f"Body yaw queued: {yaw_deg:.0f} degrees."

    def reachy_expression(self, parameters: dict[str, Any]) -> str:
        """Play a short built-in expressive gesture."""
        name = str(parameters.get("name", "curious")).strip().lower()
        if name not in {"nod", "shake_no", "happy", "curious", "reset"}:
            return "Unknown expression. Available: nod, shake_no, happy, curious, reset."

        if name == "nod":
            steps = [(0.0, 18.0, 0.0, 0.35), (0.0, -8.0, 0.0, 0.35), (0.0, 0.0, 0.0, 0.35)]
        elif name == "shake_no":
            steps = [(0.0, 0.0, -25.0, 0.35), (0.0, 0.0, 25.0, 0.45), (0.0, 0.0, 0.0, 0.35)]
        elif name == "happy":
            steps = [(0.0, -12.0, 0.0, 0.5), (0.0, -4.0, 0.0, 0.5), (0.0, 0.0, 0.0, 0.5)]
        elif name == "reset":
            steps = [(0.0, 0.0, 0.0, 0.8)]
        else:
            steps = [(12.0, -4.0, 10.0, 0.6), (0.0, 0.0, 0.0, 0.6)]

        self._queue_expression_steps(steps, expression=name)
        return f"Expression queued: {name}."

    def reachy_sleep(self, parameters: dict[str, Any]) -> str:
        """Put Reachy into sleep pose and pause app-level movement commands."""
        del parameters
        self.deps.movement_manager.clear_move_queue()
        self.deps.movement_manager.set_speech_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        self.deps.movement_manager.set_control_paused(True)
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()
        time.sleep(0.05)
        self.deps.reachy_mini.goto_sleep()
        return "Sleep command sent."

    def reachy_wake(self, parameters: dict[str, Any]) -> str:
        """Wake Reachy and resume app-level movement commands."""
        del parameters
        self.deps.reachy_mini.wake_up()
        self.deps.movement_manager.set_control_paused(False)
        self.deps.movement_manager.set_listening(False)
        self._queue_goto(0.0, 0.0, 0.0, duration_s=0.8)
        return "Wake command sent."

    def _queue_expression_steps(self, steps: list[tuple[float, float, float, float]], expression: str) -> None:
        head_pose = self.deps.reachy_mini.get_current_head_pose()
        body_yaw, antennas = _current_body_and_antennas(self.deps.reachy_mini)
        for roll, pitch, yaw, duration in steps:
            target_head = create_head_pose(0, 0, 0, roll, pitch, yaw, degrees=True)
            target_antennas = antennas
            if expression == "happy":
                target_antennas = (np.deg2rad(35.0), np.deg2rad(-35.0))
            elif expression == "curious":
                target_antennas = (np.deg2rad(20.0), np.deg2rad(-20.0))
            elif expression == "reset":
                target_antennas = (0.0, 0.0)

            self.deps.movement_manager.queue_move(
                ElevenGotoMove(
                    target_head_pose=target_head,
                    start_head_pose=head_pose,
                    target_antennas=target_antennas,
                    start_antennas=antennas,
                    target_body_yaw=body_yaw,
                    start_body_yaw=body_yaw,
                    duration=duration,
                )
            )
            head_pose = target_head
            antennas = target_antennas

    def _queue_goto(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        *,
        antennas: tuple[float, float] | None = None,
        body_yaw: float | None = None,
        duration_s: float,
        preserve_head: bool = False,
        preserve_antennas: bool = False,
    ) -> None:
        current_head = self.deps.reachy_mini.get_current_head_pose()
        current_body_yaw, current_antennas = _current_body_and_antennas(self.deps.reachy_mini)
        target_head = current_head if preserve_head else create_head_pose(0, 0, 0, roll, pitch, yaw, degrees=True)
        target_antennas = current_antennas if preserve_antennas else antennas or (0.0, 0.0)
        target_body_yaw = current_body_yaw if body_yaw is None else body_yaw

        self.deps.movement_manager.clear_move_queue()
        self.deps.movement_manager.queue_move(
            ElevenGotoMove(
                target_head_pose=target_head,
                start_head_pose=current_head,
                target_antennas=target_antennas,
                start_antennas=current_antennas,
                target_body_yaw=target_body_yaw,
                start_body_yaw=current_body_yaw,
                duration=duration_s,
            )
        )
        self.deps.movement_manager.set_moving_state(duration_s)


def run_elevenlabs_agent(
    args: Any,
    deps: ToolDependencies,
    stop_event: threading.Event | None = None,
) -> None:
    """Run the ElevenLabs conversation until stopped."""
    from elevenlabs.client import ElevenLabs
    from elevenlabs.conversational_ai.conversation import Conversation, ClientTools

    settings = ElevenLabsAgentSettings.from_args(args)
    client = ElevenLabs(api_key=settings.api_key)

    while stop_event is None or not stop_event.is_set():
        ended_event = threading.Event()
        client_tools = ClientTools()
        ReachyElevenTools(deps).register_with(client_tools)

        audio_interface = TappedDefaultAudioInterface(
            head_wobbler=deps.head_wobbler,
            speech_motion_enabled=settings.speech_motion_enabled,
        )

        def on_end_session() -> None:
            logger.warning("ElevenLabs conversation ended")
            ended_event.set()

        conversation = Conversation(
            client,
            settings.agent_id,
            user_id=settings.user_id,
            requires_auth=bool(settings.api_key),
            audio_interface=audio_interface,
            client_tools=client_tools,
            callback_agent_response=lambda response: logger.info("Agent: %s", response),
            callback_user_transcript=lambda transcript: logger.info("User: %s", transcript),
            callback_agent_response_correction=lambda original, corrected: logger.debug(
                "Agent correction: %s -> %s",
                original,
                corrected,
            ),
            callback_end_session=on_end_session,
        )

        logger.info("Starting ElevenLabs conversation with agent %s", settings.agent_id)
        try:
            conversation.start_session()
            next_keepalive = time.monotonic() + max(0.0, settings.keepalive_interval_s)
            while stop_event is None or not stop_event.is_set():
                if ended_event.wait(timeout=0.25):
                    break

                now = time.monotonic()
                if settings.keepalive_interval_s > 0 and now >= next_keepalive:
                    try:
                        conversation.register_user_activity()
                    except RuntimeError:
                        logger.warning("ElevenLabs session is no longer active; reconnecting", exc_info=True)
                        ended_event.set()
                        break
                    except Exception:
                        logger.warning("Failed to send ElevenLabs keepalive; reconnecting", exc_info=True)
                        ended_event.set()
                        break
                    next_keepalive = now + settings.keepalive_interval_s
        except KeyboardInterrupt:
            logger.info("Keyboard interruption; ending ElevenLabs conversation")
            break
        finally:
            if not ended_event.is_set():
                try:
                    conversation.end_session()
                except Exception:
                    logger.debug("Error ending ElevenLabs conversation", exc_info=True)
                    ended_event.set()
            try:
                conversation.wait_for_session_end()
            except RuntimeError:
                logger.debug("ElevenLabs conversation thread was not started", exc_info=True)

        if stop_event is not None and stop_event.is_set():
            break

        delay = max(0.0, settings.reconnect_delay_s)
        logger.warning("Restarting ElevenLabs conversation in %.1fs", delay)
        time.sleep(delay)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.1f", name, raw, default)
        return default


def _current_body_and_antennas(robot: ReachyMini) -> tuple[float, tuple[float, float]]:
    try:
        head_joints, antenna_joints = robot.get_current_joint_positions()
        body_yaw = float(head_joints[0]) if head_joints is not None and len(head_joints) else 0.0
        antennas = (
            float(antenna_joints[0]) if antenna_joints is not None and len(antenna_joints) > 0 else 0.0,
            float(antenna_joints[1]) if antenna_joints is not None and len(antenna_joints) > 1 else 0.0,
        )
        return body_yaw, antennas
    except Exception:
        logger.debug("Could not read current joints; falling back to neutral", exc_info=True)
        return 0.0, (0.0, 0.0)


def _number(parameters: dict[str, Any], key: str, default: float, lower: float, upper: float) -> float:
    try:
        value = float(parameters.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(lower, min(upper, value))


def _direction_default(parameters: dict[str, Any], key: str) -> float:
    direction = str(parameters.get("direction", "")).strip().lower()
    defaults = {
        "left": {"yaw_deg": 25.0},
        "right": {"yaw_deg": -25.0},
        "up": {"pitch_deg": -18.0},
        "down": {"pitch_deg": 18.0},
        "front": {"roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0},
        "center": {"roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0},
    }
    return defaults.get(direction, {}).get(key, 0.0)


def tool_schema_markdown() -> str:
    """Return dashboard tool schema documentation for README/plan reuse."""
    schemas = [
        {
            "name": "reachy_move_head",
            "parameters": {
                "direction": "optional enum: left, right, up, down, front",
                "roll_deg": "optional number -40..40",
                "pitch_deg": "optional number -40..40",
                "yaw_deg": "optional number -65..65",
                "duration_s": "optional number 0.3..4.0",
            },
        },
        {
            "name": "reachy_set_antennas",
            "parameters": {
                "right_deg": "number -175..175",
                "left_deg": "number -175..175",
                "duration_s": "optional number 0.3..4.0",
            },
        },
        {
            "name": "reachy_set_body_yaw",
            "parameters": {"yaw_deg": "number -160..160", "duration_s": "optional number 0.3..5.0"},
        },
        {"name": "reachy_expression", "parameters": {"name": "enum: nod, shake_no, happy, curious, reset"}},
        {"name": "reachy_sleep", "parameters": {}},
        {"name": "reachy_wake", "parameters": {}},
    ]
    return "\n".join(f"- `{schema['name']}`: `{json.dumps(schema['parameters'])}`" for schema in schemas)
