---
title: Reachy Eleven Agent
emoji: 🤖
colorFrom: purple
colorTo: gray
sdk: static
pinned: false
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# Reachy Eleven Agent

Reachy Mini voice interaction app powered by ElevenLabs ElevenAgents.

The app listens through the system default microphone, sends the conversation to
an ElevenLabs agent, plays the agent response through the system default speaker,
and taps that audio stream so Reachy moves while speaking. The app also exposes
client tools that let the ElevenLabs agent trigger simple Reachy gestures.

## Requirements

- A Reachy Mini with `reachy-mini-daemon` running.
- Python 3.12.
- `uv` for dependency management.
- An ElevenLabs Conversational AI agent ID.
- System audio devices configured for the microphone and speaker you want the app
  to use.
- PortAudio headers if you want live voice audio.

## Quick Start

From the repository root, which is the directory containing `pyproject.toml`:

```bash
export UV_LINK_MODE=copy  # optional on mounted volumes where hardlinks are not supported
uv sync --dev --extra audio
cp .env.example .env
```

Edit `.env` and set:

```bash
ELEVENLABS_AGENT_ID=agent_your_public_agent_id
```

Start `reachy-mini-daemon`, then launch the app:

```bash
uv run reachy-eleven-agent
```

If your daemon uses a robot name:

```bash
uv run reachy-eleven-agent --robot-name <robot-name>
```

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

Install PortAudio build dependencies before syncing the `audio` extra:

```bash
# Reachy OS
sudo apt-get update
sudo apt-get install -y build-essential python3-dev portaudio19-dev libasound2-dev
```

## Setup

Clone this repository, then work from the directory that contains
`pyproject.toml`:

```bash
git clone <repository-url>
cd <repository-directory>
```

Create the synchronized environment:

```bash
# Optional on mounted volumes where hardlinks are not supported.
export UV_LINK_MODE=copy

# Use --extra audio for live ElevenLabs microphone/speaker support.
uv sync --dev --extra audio
```

Create your local environment file:

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```bash
REACHY_AGENT_PROVIDER=elevenlabs
ELEVENLABS_AGENT_ID=agent_your_public_agent_id
REACHY_ELEVEN_KEEPALIVE_INTERVAL_S=20
REACHY_ELEVEN_RECONNECT_DELAY_S=2
```

For a private ElevenLabs agent, also set:

```bash
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

Do not commit `.env`.

## ElevenLabs Agent Setup

In the ElevenLabs dashboard, create or open a Conversational AI agent. Add the
Reachy controls as **client tools**. ElevenLabs' client tool documentation is
available at <https://elevenlabs.io/docs/conversational-ai/customization/tools/client-tools>.

1. Open your agent in ElevenLabs.
2. Go to **Tools**.
3. Add a **Client** tool.
4. Click **Edit as JSON**.
5. Replace the generated JSON with one of the blocks below.
6. Save the tool, then repeat for the remaining tools.
7. Make sure each saved tool is enabled for the agent.

Tool names are case-sensitive. They must match the names registered by this app:
`reachy_move_head`, `reachy_set_antennas`, `reachy_set_body_yaw`,
`reachy_expression`, `reachy_sleep`, and `reachy_wake`.

These tools use `expects_response: false` because they are non-blocking robot
gestures. The app still returns short status strings internally, but the agent
does not need to pause the conversation waiting for them.

`reachy_move_head`

```json
{
  "type": "client",
  "name": "reachy_move_head",
  "description": "Move Reachy's head. Use small, expressive movements to look left, right, up, down, forward, or to set roll/pitch/yaw angles directly.",
  "expects_response": false,
  "response_timeout_secs": 1,
  "parameters": [
    {
      "id": "direction",
      "type": "string",
      "value_type": "llm_prompt",
      "description": "Optional preset direction. Use one of: left, right, up, down, front.",
      "dynamic_variable": "",
      "constant_value": "",
      "enum": ["left", "right", "up", "down", "front"],
      "required": false
    },
    {
      "id": "roll_deg",
      "type": "number",
      "value_type": "llm_prompt",
      "description": "Optional roll angle in degrees from -40 to 40.",
      "dynamic_variable": "",
      "constant_value": "",
      "enum": null,
      "required": false
    },
    {
      "id": "pitch_deg",
      "type": "number",
      "value_type": "llm_prompt",
      "description": "Optional pitch angle in degrees from -40 to 40.",
      "dynamic_variable": "",
      "constant_value": "",
      "enum": null,
      "required": false
    },
    {
      "id": "yaw_deg",
      "type": "number",
      "value_type": "llm_prompt",
      "description": "Optional yaw angle in degrees from -65 to 65.",
      "dynamic_variable": "",
      "constant_value": "",
      "enum": null,
      "required": false
    },
    {
      "id": "duration_s",
      "type": "number",
      "value_type": "llm_prompt",
      "description": "Optional movement duration in seconds from 0.3 to 4.0. Use 0.8 for a normal gesture.",
      "dynamic_variable": "",
      "constant_value": "",
      "enum": null,
      "required": false
    }
  ],
  "dynamic_variables": {
    "dynamic_variable_placeholders": {}
  },
  "assignments": [],
  "disable_interruptions": false,
  "pre_tool_speech": "auto",
  "tool_call_sound": null,
  "tool_call_sound_behavior": "auto",
  "execution_mode": "immediate",
  "response_mocks": []
}
```

`reachy_set_antennas`

```json
{
  "type": "client",
  "name": "reachy_set_antennas",
  "description": "Move Reachy's right and left antennas. Use this for small expressive ear-like gestures.",
  "expects_response": false,
  "response_timeout_secs": 1,
  "parameters": [
    {
      "id": "right_deg",
      "type": "number",
      "value_type": "llm_prompt",
      "description": "Right antenna angle in degrees from -175 to 175.",
      "dynamic_variable": "",
      "constant_value": "",
      "enum": null,
      "required": true
    },
    {
      "id": "left_deg",
      "type": "number",
      "value_type": "llm_prompt",
      "description": "Left antenna angle in degrees from -175 to 175.",
      "dynamic_variable": "",
      "constant_value": "",
      "enum": null,
      "required": true
    },
    {
      "id": "duration_s",
      "type": "number",
      "value_type": "llm_prompt",
      "description": "Optional movement duration in seconds from 0.3 to 4.0. Use 0.7 for a normal gesture.",
      "dynamic_variable": "",
      "constant_value": "",
      "enum": null,
      "required": false
    }
  ],
  "dynamic_variables": {
    "dynamic_variable_placeholders": {}
  },
  "assignments": [],
  "disable_interruptions": false,
  "pre_tool_speech": "auto",
  "tool_call_sound": null,
  "tool_call_sound_behavior": "auto",
  "execution_mode": "immediate",
  "response_mocks": []
}
```

`reachy_set_body_yaw`

```json
{
  "type": "client",
  "name": "reachy_set_body_yaw",
  "description": "Rotate Reachy's body around its vertical axis. Use sparingly for larger orientation changes.",
  "expects_response": false,
  "response_timeout_secs": 1,
  "parameters": [
    {
      "id": "yaw_deg",
      "type": "number",
      "value_type": "llm_prompt",
      "description": "Body yaw angle in degrees from -160 to 160.",
      "dynamic_variable": "",
      "constant_value": "",
      "enum": null,
      "required": true
    },
    {
      "id": "duration_s",
      "type": "number",
      "value_type": "llm_prompt",
      "description": "Optional movement duration in seconds from 0.3 to 5.0. Use 1.0 for a normal turn.",
      "dynamic_variable": "",
      "constant_value": "",
      "enum": null,
      "required": false
    }
  ],
  "dynamic_variables": {
    "dynamic_variable_placeholders": {}
  },
  "assignments": [],
  "disable_interruptions": false,
  "pre_tool_speech": "auto",
  "tool_call_sound": null,
  "tool_call_sound_behavior": "auto",
  "execution_mode": "immediate",
  "response_mocks": []
}
```

`reachy_expression`

```json
{
  "type": "client",
  "name": "reachy_expression",
  "description": "Play a short built-in Reachy expression. Prefer this tool for common conversational gestures.",
  "expects_response": false,
  "response_timeout_secs": 1,
  "parameters": [
    {
      "id": "name",
      "type": "string",
      "value_type": "llm_prompt",
      "description": "Expression name. Use one of: nod, shake_no, happy, curious, reset.",
      "dynamic_variable": "",
      "constant_value": "",
      "enum": ["nod", "shake_no", "happy", "curious", "reset"],
      "required": true
    }
  ],
  "dynamic_variables": {
    "dynamic_variable_placeholders": {}
  },
  "assignments": [],
  "disable_interruptions": false,
  "pre_tool_speech": "auto",
  "tool_call_sound": null,
  "tool_call_sound_behavior": "auto",
  "execution_mode": "immediate",
  "response_mocks": []
}
```

`reachy_sleep`

```json
{
  "type": "client",
  "name": "reachy_sleep",
  "description": "Put Reachy into its sleep pose and pause app-level motion. Use only when the user asks Reachy to sleep, rest, or shut down.",
  "expects_response": false,
  "response_timeout_secs": 1,
  "parameters": [],
  "dynamic_variables": {
    "dynamic_variable_placeholders": {}
  },
  "assignments": [],
  "disable_interruptions": false,
  "pre_tool_speech": "auto",
  "tool_call_sound": null,
  "tool_call_sound_behavior": "auto",
  "execution_mode": "immediate",
  "response_mocks": []
}
```

`reachy_wake`

```json
{
  "type": "client",
  "name": "reachy_wake",
  "description": "Wake Reachy and resume app-level motion. Use when the user asks Reachy to wake up or continue.",
  "expects_response": false,
  "response_timeout_secs": 1,
  "parameters": [],
  "dynamic_variables": {
    "dynamic_variable_placeholders": {}
  },
  "assignments": [],
  "disable_interruptions": false,
  "pre_tool_speech": "auto",
  "tool_call_sound": null,
  "tool_call_sound_behavior": "auto",
  "execution_mode": "immediate",
  "response_mocks": []
}
```

Prompt guidance for the agent: speak as Reachy, keep spoken answers concise, and
call movement tools sparingly to punctuate the conversation. Use `reachy_expression`
for common gestures before composing lower-level head or antenna motions.

Suggested system prompt addition:

```text
You are embodied as Reachy Mini. Keep replies concise and conversational.
Use Reachy client tools only when motion adds meaning to the interaction.
Prefer reachy_expression for simple gestures such as nodding, curiosity, or happiness.
Do not call motion tools repeatedly or continuously. Use small movements and let speech remain the main interaction.
Use reachy_sleep only when the user asks you to sleep, rest, or shut down.
Use reachy_wake only when the user asks you to wake up or resume.
```

## Run Locally

Start the Reachy Mini daemon first, then run:

```bash
uv run reachy-eleven-agent
```

If your daemon was started with a robot name:

```bash
uv run reachy-eleven-agent --robot-name <robot-name>
```

The default provider is ElevenLabs. The original OpenAI realtime conversation
template remains available for development:

```bash
uv run reachy-eleven-agent --provider openai --gradio
```

## Validate

Run the focused tests and Reachy app packaging check:

```bash
uv run pytest tests/test_elevenlabs_agent.py -q
uv run reachy-mini-app-assistant check .
```

## Publish

Authenticate with Hugging Face using a token that can create Spaces:

```bash
uv run hf auth login
uv run hf auth whoami
```

Publish the app:

```bash
uv run reachy-mini-app-assistant publish . "Publish Reachy Eleven Agent"
```

Use `--private` or `--public` if you need to control Space visibility:

```bash
uv run reachy-mini-app-assistant publish --private . "Publish private Reachy Eleven Agent"
```

## Updating

After pulling changes:

```bash
git pull
uv sync --dev --extra audio
uv run reachy-mini-app-assistant check .
```

## Troubleshooting

- `ModuleNotFoundError: No module named 'reachy_mini'`: run commands with `uv run`
  from the directory containing `pyproject.toml`, or rerun `uv sync --dev --extra audio`.
- PyAudio or `portaudio.h` errors: install the PortAudio system packages listed in
  Requirements, then rerun `uv sync --dev --extra audio`.
- No microphone or speaker audio: verify the system default input/output device
  before starting the app.
- Robot keeps moving but no longer responds to voice: the motion loop may still
  be running after the ElevenLabs websocket ended. The app sends keepalives and
  automatically reconnects; check logs for `ElevenLabs conversation ended` and
  `Restarting ElevenLabs conversation`. Lower `REACHY_ELEVEN_KEEPALIVE_INTERVAL_S`
  if your agent times out quickly.
- Hugging Face `403 Forbidden` while publishing: the token is valid but does not
  have permission to create Spaces in the target namespace.
- Reachy connection timeout: verify `reachy-mini-daemon` is running and pass
  `--robot-name` if the daemon uses one.
