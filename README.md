# Quasar

Quasar, or CAI, is a general-purpose AI runtime for host applications. It exposes a small public facade while keeping the legacy module loading system compatible.

The package can be used standalone or embedded by a host application. Standalone usage is supported through editable installation or by importing the package from its parent directory.

Host applications import the facade through `from Quasar.cai import CAIApp`. Quasar package internals use package-relative imports to keep internal modules movable within the package.

## Install For Development

From this directory:

```powershell
python -m pip install -e .
```

Optional dependency groups:

```powershell
python -m pip install -e ".[langchain,workflow,media]"
python -m pip install -e ".[web]"
python -m pip install -e ".[object-recognition]"
```

Optional host integrations can be installed separately from the core runtime.

## Public Facade

```python
from Quasar.cai import CAIApp, ChatRequest, StreamEvent

app = CAIApp()
request = ChatRequest.from_text("请总结这段文字", session_id="demo")

for chunk in app.chat_stream(request):
    event = StreamEvent.from_legacy_chunk(chunk)
    print(event.to_dict())
```

The facade currently wraps the legacy integrated stream handler. New host code should call `CAIApp`; old code can continue to use `ai_service.entrance.get_ai_entrance()`.

## Examples

- `examples/cli_chat.py`: minimal CLI-style script.
- `examples/fastapi_websocket.py`: FastAPI WebSocket integration sketch.
- `quasar-chat`: console script installed by `pyproject.toml`.

## Architecture Notes

- `cai/`: public facade, runtime, protocol, and plugin manager.
- `ai_service/entrance.py`: legacy entrance compatibility layer.
- `ai_modules/`: feature modules loaded from `ai_service/module_settings.yaml`.
- `ai_tools/`: tool registry, response adapter, session helpers, and tool loading.
- `ai_workflow/`: workflow registries and LangGraph execution helpers.
- `ai_media_resource/`: media registry and storage adapters.
- `ai_agent/`: agent execution and conversation history.

## Documentation

See `docs/API_REFERENCE.md` for the public API surface and packaging notes.
