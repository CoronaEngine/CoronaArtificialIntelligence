# Quasar

[简体中文](README.md) | English

Quasar, also known as CAI, is a general-purpose AI runtime for host applications. It exposes chat streaming, tools, workflows, plugins, and runtime capabilities through a small, stable public facade while preserving compatibility with the legacy module loader and entrance protocol.

Quasar can be installed as a standalone Python package or embedded in a host such as the CoronaEngine editor. New code should import the public API from `Quasar.cai`, with required capabilities assembled explicitly by the host.

## Requirements and Installation

- Python 3.10 or later
- Run the following command from this directory for an editable development install:

```powershell
python -m pip install -e .
```

Install optional dependencies as needed:

```powershell
# LangChain, workflow, and media capabilities
python -m pip install -e ".[langchain,workflow,media]"

# Model/storage providers and plugin configuration
python -m pip install -e ".[providers,redis,mongo,sql,plugins]"

# Web integration or object recognition
python -m pip install -e ".[web]"
python -m pip install -e ".[object-recognition]"

# All runtime dependencies, or development test dependencies
python -m pip install -e ".[all]"
python -m pip install -e ".[dev]"
```

The core package does not require third-party runtime dependencies. The `cabbage` dependency group is currently empty and is retained for compatibility with host installation profiles.

## Quick Start

Create a request through the public facade and consume its stream events:

```python
from Quasar.cai import CAIApp, ChatRequest, StreamEvent

app = CAIApp()
request = ChatRequest.from_text("Summarize this text", session_id="demo")

try:
    for chunk in app.chat_stream(request):
        event = StreamEvent.from_legacy_chunk(chunk)
        print(event.to_dict())
finally:
    app.shutdown()
```

`CAIApp()` uses the default runtime. Before calling `chat_stream()`, the host must register an entrance handler or inject a configured `CAIRuntime`. New integrations should use `RuntimeBuilder` to assemble models, stores, tools, and plugins explicitly. To connect the legacy entrance, use `CAIApp.from_legacy_entrance()`.

## Examples and CLI

- [`examples/cli_chat.py`](examples/cli_chat.py): minimal command-line integration.
- [`examples/fastapi_websocket.py`](examples/fastapi_websocket.py): FastAPI WebSocket integration.
- `quasar-chat`: console command installed through `pyproject.toml`.

```powershell
quasar-chat "Introduce Quasar in one sentence" --session-id demo
```

The console entry point and code examples also require the host to configure the runtime entrance first.

## Architecture

- `cai/`: public facade, runtime, protocols, capability interfaces, and plugin management.
- `adapters/`: host and infrastructure adapters.
- `ai_service/entrance.py`: legacy entrance compatibility layer.
- `ai_modules/`: feature modules loadable through `ai_service/module_settings.yaml`.
- `ai_tools/`: tool registration, response adapters, session helpers, and tool loading.
- `ai_workflow/`: workflow registries and LangGraph execution helpers.
- `ai_media_resource/`: media resource registries and storage adapters.
- `ai_agent/`: agent execution and conversation history.
- `compat/`: compatibility code for legacy host interfaces.

The core runtime uses explicit assembly and does not automatically discover models, persistence backends, or the legacy AI entrance.

## Documentation

- [API Reference](docs/API_REFERENCE.md): public API and packaging notes.
- [Quasar Generalization Guide (Chinese)](docs/QUASAR_GENERALIZATION_ZH_CN.md): runtime assembly, capability interfaces, and migration guidance.
