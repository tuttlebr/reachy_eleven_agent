"""Entrypoint for the Reachy Mini conversation app."""

import os
import sys
import time
import asyncio
import logging
import argparse
import threading
from typing import Any, Dict, List, Optional

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_eleven_agent.utils import (
    parse_args,
    setup_logger,
    handle_vision_stuff,
    log_connection_troubleshooting,
)


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def _robot_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Build ReachyMini constructor kwargs from app arguments."""
    robot_kwargs: Dict[str, Any] = {}
    if args.robot_name is not None:
        robot_kwargs["robot_name"] = args.robot_name

    if args.provider == "elevenlabs":
        robot_kwargs["media_backend"] = os.getenv("REACHY_ELEVEN_MEDIA_BACKEND", "no_media")

    return robot_kwargs


def _close_robot_connection(robot: ReachyMini, logger: logging.Logger) -> None:
    """Best-effort cleanup for an SDK connection."""
    try:
        robot.media.close()
    except Exception as e:
        logger.debug("Error closing media during shutdown: %s", e)

    try:
        if getattr(robot, "_media_released", False):
            robot.client.acquire_media()
    except Exception as e:
        logger.debug("Error returning media to daemon during shutdown: %s", e)

    try:
        robot.client.disconnect()
    except Exception as e:
        logger.debug("Error disconnecting Reachy Mini client: %s", e)


def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args, _ = parse_args()
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[Any] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    # Putting these dependencies here makes the dashboard faster to load when the conversation app is installed
    from reachy_eleven_agent.moves import MovementManager
    from reachy_eleven_agent.tools.core_tools import ToolDependencies
    from reachy_eleven_agent.audio.head_wobbler import HeadWobbler

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Eleven Agent app with provider=%s", args.provider)
    stop_event = app_stop_event or threading.Event()

    if args.no_camera and args.head_tracker is not None:
        logger.warning(
            "Head tracking disabled: --no-camera flag is set. "
            "Remove --no-camera to enable head tracking."
        )

    robot_kwargs = _robot_kwargs_from_args(args)

    if robot is None:
        try:
            logger.info("Initializing ReachyMini with options: %s", robot_kwargs)
            robot = ReachyMini(**robot_kwargs)

        except TimeoutError as e:
            logger.error(
                "Connection timeout: Failed to connect to Reachy Mini daemon. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(
                "Connection failed: Unable to establish connection to Reachy Mini. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(
                f"Unexpected error during robot initialization: {type(e).__name__}: {e}"
            )
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    # Auto-enable Gradio in simulation mode (both MuJoCo for daemon and mockup-sim for desktop app)
    status = robot.client.get_status()
    if isinstance(status, dict):
        simulation_enabled = status.get("simulation_enabled", False)
        mockup_sim_enabled = status.get("mockup_sim_enabled", False)
    else:
        simulation_enabled = getattr(status, "simulation_enabled", False)
        mockup_sim_enabled = getattr(status, "mockup_sim_enabled", False)

    is_simulation = simulation_enabled or mockup_sim_enabled

    if is_simulation and not args.gradio:
        logger.info("Simulation mode detected. Automatically enabling gradio flag.")
        args.gradio = True

    if args.provider == "elevenlabs" and not args.no_camera:
        logger.info("ElevenLabs v1 does not use camera context; disabling camera worker.")
        args.no_camera = True

    camera_worker, _, vision_manager = handle_vision_stuff(args, robot)
    robot_ref = {"robot": robot}
    reconnect_lock = threading.Lock()
    reconnect_thread: threading.Thread | None = None

    def should_stop_reconnect() -> bool:
        return stop_event.is_set()

    def stop_on_connection_lost(error: BaseException) -> None:
        logger.error("Lost connection to Reachy Mini daemon; stopping app. Details: %s", error)
        stop_event.set()

    def reconnect_on_connection_lost(error: BaseException) -> None:
        nonlocal reconnect_thread

        logger.error(
            "Lost connection to Reachy Mini daemon; keeping voice app alive and reconnecting. Details: %s",
            error,
        )

        with reconnect_lock:
            if reconnect_thread is not None and reconnect_thread.is_alive():
                logger.debug("Reachy Mini reconnect already in progress")
                return

            reconnect_thread = threading.Thread(
                target=reconnect_robot_loop,
                name="reachy-mini-reconnect",
                daemon=True,
            )
            reconnect_thread.start()

    def reconnect_robot_loop() -> None:
        nonlocal reconnect_thread

        delay = max(0.5, float(os.getenv("REACHY_ELEVEN_ROBOT_RECONNECT_DELAY_S", "2.0")))
        max_delay = max(delay, float(os.getenv("REACHY_ELEVEN_ROBOT_RECONNECT_MAX_DELAY_S", "30.0")))
        attempt = 1

        _close_robot_connection(robot_ref["robot"], logger)

        while not should_stop_reconnect():
            try:
                logger.warning("Attempting Reachy Mini daemon reconnect (attempt %d)", attempt)
                new_robot = ReachyMini(**robot_kwargs)
            except (ConnectionError, TimeoutError) as e:
                logger.warning(
                    "Reachy Mini reconnect attempt %d failed: %s. Retrying in %.1fs",
                    attempt,
                    e,
                    delay,
                )
            except Exception:
                logger.warning(
                    "Unexpected error during Reachy Mini reconnect attempt %d. Retrying in %.1fs",
                    attempt,
                    delay,
                    exc_info=True,
                )
            else:
                robot_ref["robot"] = new_robot
                deps.reachy_mini = new_robot
                movement_manager.replace_robot(new_robot)
                if camera_worker is not None:
                    camera_worker.reachy_mini = new_robot
                logger.info("Reconnected to Reachy Mini daemon; app remains running")
                break

            attempt += 1
            if stop_event.wait(delay):
                break
            delay = min(max_delay, delay * 1.5)

        with reconnect_lock:
            reconnect_thread = None

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
        on_connection_lost=reconnect_on_connection_lost if args.provider == "elevenlabs" else stop_on_connection_lost,
    )

    head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_manager=vision_manager,
        head_wobbler=head_wobbler,
    )

    if args.provider == "elevenlabs":
        from reachy_eleven_agent.elevenlabs_agent import run_elevenlabs_agent

        if settings_app is not None:
            from reachy_eleven_agent.elevenlabs_dashboard import (
                ElevenLabsDashboardRuntime,
                mount_elevenlabs_dashboard,
            )

            dashboard_runtime = ElevenLabsDashboardRuntime(
                args=args,
                deps=deps,
                movement_manager=movement_manager,
                head_wobbler=head_wobbler,
                camera_worker=camera_worker,
                vision_manager=vision_manager,
                app_stop_event=stop_event,
                instance_path=instance_path,
                conversation_runner=run_elevenlabs_agent,
            )
            mount_elevenlabs_dashboard(settings_app, dashboard_runtime)
            logger.info("ElevenLabs dashboard controls are ready at %s", ReachyElevenAgent.custom_app_url)

            try:
                stop_event.wait()
            finally:
                dashboard_runtime.close()
                with reconnect_lock:
                    thread_to_join = reconnect_thread
                if thread_to_join is not None and thread_to_join.is_alive():
                    thread_to_join.join(timeout=5.0)
                _close_robot_connection(robot_ref["robot"], logger)
                time.sleep(1)
                logger.info("Shutdown complete.")
            return

        movement_manager.start()
        head_wobbler.start()
        if camera_worker:
            camera_worker.start()
        if vision_manager:
            vision_manager.start()

        try:
            run_elevenlabs_agent(args, deps, stop_event=stop_event)
        finally:
            stop_event.set()
            movement_manager.stop()
            head_wobbler.stop()
            if camera_worker:
                camera_worker.stop()
            if vision_manager:
                vision_manager.stop()
            with reconnect_lock:
                thread_to_join = reconnect_thread
            if thread_to_join is not None and thread_to_join.is_alive():
                thread_to_join.join(timeout=5.0)
            _close_robot_connection(robot_ref["robot"], logger)
            time.sleep(1)
            logger.info("Shutdown complete.")
        return

    import gradio as gr
    from fastapi import FastAPI
    from fastrtc import Stream
    from gradio.utils import get_space

    from reachy_eleven_agent.console import LocalStream
    from reachy_eleven_agent.openai_realtime import OpenaiRealtimeHandler

    current_file_path = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"Current file absolute path: {current_file_path}")
    chatbot = gr.Chatbot(
        type="messages",
        resizable=True,
        avatar_images=(
            os.path.join(current_file_path, "images", "user_avatar.png"),
            os.path.join(current_file_path, "images", "reachymini_avatar.png"),
        ),
    )
    logger.debug(f"Chatbot avatar images: {chatbot.avatar_images}")

    handler = OpenaiRealtimeHandler(deps, gradio_mode=args.gradio, instance_path=instance_path)

    stream_manager: gr.Blocks | LocalStream | None = None

    if args.gradio:
        api_key_textbox = gr.Textbox(
            label="OPENAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY") if not get_space() else "",
        )

        from reachy_eleven_agent.gradio_personality import PersonalityUI

        personality_ui = PersonalityUI()
        personality_ui.create_components()

        stream = Stream(
            handler=handler,
            mode="send-receive",
            modality="audio",
            additional_inputs=[
                chatbot,
                api_key_textbox,
                *personality_ui.additional_inputs_ordered(),
            ],
            additional_outputs=[chatbot],
            additional_outputs_handler=update_chatbot,
            ui_args={"title": "Talk with Reachy Mini"},
        )
        stream_manager = stream.ui
        if not settings_app:
            app = FastAPI()
        else:
            app = settings_app

        personality_ui.wire_events(handler, stream_manager)

        app = gr.mount_gradio_app(app, stream.ui, path="/")
    else:
        # In headless mode, wire settings_app + instance_path to console LocalStream
        stream_manager = LocalStream(
            handler,
            robot,
            settings_app=settings_app,
            instance_path=instance_path,
        )

    # Each async service → its own thread/loop
    movement_manager.start()
    head_wobbler.start()
    if camera_worker:
        camera_worker.start()
    if vision_manager:
        vision_manager.start()

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        stop_event.set()
        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()
        if vision_manager:
            vision_manager.stop()

        # prevent connection to keep alive some threads
        _close_robot_connection(robot_ref["robot"], logger)
        time.sleep(1)
        logger.info("Shutdown complete.")


class ReachyElevenAgent(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False
    request_media_backend = "no_media"

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini conversation app."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        args, _ = parse_args()

        # is_wireless = reachy_mini.client.get_status()["wireless_version"]
        # args.head_tracker = None if is_wireless else "mediapipe"

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = ReachyElevenAgent()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
