# Quasar

简体中文 | [English](README_EN.md)

Quasar（亦称 CAI）是一个面向宿主应用的通用 AI 运行时。它通过小型、稳定的公共门面提供会话流、工具、工作流、插件和运行时能力，同时保留对旧模块加载与入口协议的兼容。

Quasar 既可以作为独立 Python 包安装，也可以嵌入 CoronaEngine 编辑器等宿主应用。新代码应从 `Quasar.cai` 导入公共 API，并由宿主显式装配所需能力。

## 环境要求与安装

- Python 3.10 或更高版本
- 在本目录执行以下命令进行开发模式安装：

```powershell
python -m pip install -e .
```

按需安装可选依赖：

```powershell
# LangChain、工作流与媒体能力
python -m pip install -e ".[langchain,workflow,media]"

# 模型/存储提供方与插件配置
python -m pip install -e ".[providers,redis,mongo,sql,plugins]"

# Web 集成或目标识别
python -m pip install -e ".[web]"
python -m pip install -e ".[object-recognition]"

# 全部运行时依赖，或开发测试依赖
python -m pip install -e ".[all]"
python -m pip install -e ".[dev]"
```

核心包本身不强制安装第三方运行时依赖。`cabbage` 依赖组目前为空，用于兼容宿主安装配置。

## 快速开始

通过公共门面创建请求并消费流式事件：

```python
from Quasar.cai import CAIApp, ChatRequest, StreamEvent

app = CAIApp()
request = ChatRequest.from_text("请总结这段文字", session_id="demo")

try:
    for chunk in app.chat_stream(request):
        event = StreamEvent.from_legacy_chunk(chunk)
        print(event.to_dict())
finally:
    app.shutdown()
```

`CAIApp()` 使用默认运行时。宿主必须在调用 `chat_stream()` 前注册入口处理器，或注入已配置的 `CAIRuntime`。新集成建议使用 `RuntimeBuilder` 显式装配模型、存储、工具与插件；需要连接旧入口时，可使用 `CAIApp.from_legacy_entrance()`。

## 示例与命令行

- [`examples/cli_chat.py`](examples/cli_chat.py)：最小命令行集成示例。
- [`examples/fastapi_websocket.py`](examples/fastapi_websocket.py)：FastAPI WebSocket 集成示例。
- `quasar-chat`：由 `pyproject.toml` 安装的控制台命令。

```powershell
quasar-chat "请用一句话介绍 Quasar" --session-id demo
```

命令行入口与代码示例同样需要宿主先完成运行时入口配置。

## 架构

- `cai/`：公共门面、运行时、协议、能力接口与插件管理。
- `adapters/`：宿主或基础设施适配器。
- `ai_service/entrance.py`：旧入口兼容层。
- `ai_modules/`：可由 `ai_service/module_settings.yaml` 加载的功能模块。
- `ai_tools/`：工具注册、响应适配、会话辅助与工具加载。
- `ai_workflow/`：工作流注册与 LangGraph 执行辅助。
- `ai_media_resource/`：媒体资源注册与存储适配。
- `ai_agent/`：Agent 执行与对话历史。
- `compat/`：旧宿主接口的兼容代码。

核心运行时采用显式装配，不会自动发现模型、持久化后端或旧 AI 入口。

## 文档

- [API Reference](docs/API_REFERENCE.md)：公共 API 与打包说明。
- [Quasar 通用化指南](docs/QUASAR_GENERALIZATION_ZH_CN.md)：运行时装配、能力接口和迁移说明。
