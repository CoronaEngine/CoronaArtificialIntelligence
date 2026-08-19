# Quasar 通用运行时中文指南

## 1. 定位

Quasar 是一个与 Flask、FastAPI、数据库和云厂商无关的 AI 运行时核心。默认安装只使用 Python 标准库；默认 Runtime 只创建进程内存能力，不读取环境变量、不创建目录、不连接网络、不导入数据库驱动，也不启动后台线程。

Quasar 采用显式装配：宿主负责解析环境变量、配置文件和密钥，再通过 `RuntimeBuilder` 或 `CAIRuntime.set_capability()` 注入模型、工具、Store 和插件。Core 不自动发现任何后端。

## 2. 安装方式

只安装核心：

```bash
pip install -e ./Quasar
```

按需安装能力：

```bash
pip install -e "./Quasar[langchain]"
pip install -e "./Quasar[redis]"
pip install -e "./Quasar[mongo]"
pip install -e "./Quasar[object-recognition]"
```

本地完整开发环境可使用 `pip install -e "./Quasar[all,dev]"`。生产环境不应为了方便安装 `all`，应只选择实际启用的 extra。

## 3. 纯内存运行

```python
from Quasar.cai import (
    Capability,
    RuntimeBuilder,
    RuntimeConfig,
    SessionChange,
    SessionSnapshot,
)

runtime = (
    RuntimeBuilder(RuntimeConfig(max_concurrency=4))
    .use_core_defaults()
    .build()
)

sessions = runtime.require_capability(Capability.SESSION_STORE)
sessions.create(SessionSnapshot("request-1", values={"step": 1}))
sessions.update(SessionChange("request-1", state="running"))
assert sessions.get("request-1").state == "running"

runtime.start()
runtime.close(timeout=10)
```

默认包含 `ConversationStore`、`SessionStore`、`ArtifactStore` 和 `EventBus` 的内存实现。数据仅存在于当前进程；多进程部署时各进程互不共享。

## 4. 工具依赖与 ToolContext

工具必须声明依赖，并且只能从传入的 `ToolContext` 读取能力：

```python
from Quasar.cai import Capability, RuntimeBuilder, ToolSpec

def create_summary_tool(context):
    model = context.require(Capability.MODEL)
    return SummaryTool(model=model)

runtime = (
    RuntimeBuilder()
    .use_model(model)
    .add_tool(ToolSpec(
        name="summarize",
        factory=create_summary_tool,
        requires=frozenset({Capability.MODEL}),
        required=True,
    ))
    .build()
)
```

可选工具缺少依赖时不会执行 factory，注册状态为 `unavailable`；必需工具缺少依赖时 `build()` 直接抛出 `ConfigurationError`。同名注册项只能使用 `replace=True` 显式替换。

## 5. 本地文件产物

文件适配器在构造时不访问磁盘，只有 `runtime.start()` 才创建目录：

```python
from pathlib import Path
from Quasar.adapters.artifacts import (
    LocalFileArtifactStore,
    LocalFileArtifactStoreConfig,
)
from Quasar.cai import ArtifactInput, Capability, RuntimeBuilder

store = LocalFileArtifactStore(
    LocalFileArtifactStoreConfig(Path("./outputs/artifacts"))
)
runtime = RuntimeBuilder().use_artifact_store(store).build()
runtime.start()

ref = store.put(ArtifactInput(b"hello", "result.txt", "text/plain"))
with store.open(ref) as stream:
    assert stream.read() == b"hello"

runtime.close()
```

`ArtifactRef` 是后端无关引用，不承诺暴露真实文件路径。HTTP 下载 URL 应由宿主根据自己的鉴权和路由规则生成。

## 6. Redis 与 MongoDB 会话存储

Redis：

```python
from Quasar.adapters.sessions import RedisSessionStore, RedisSessionStoreConfig
from Quasar.cai import RuntimeBuilder

store = RedisSessionStore(RedisSessionStoreConfig(
    url="redis://redis.internal:6379/0",
    key_prefix="my-service:session:",
))
runtime = RuntimeBuilder().use_session_store(store).build()
runtime.start()  # 此时才导入 redis 驱动并连接
```

MongoDB：

```python
from Quasar.adapters.sessions import MongoSessionStore, MongoSessionStoreConfig

store = MongoSessionStore(MongoSessionStoreConfig(
    uri="mongodb://mongo.internal:27017",
    database="my_service",
    collection="agent_sessions",
))
runtime = RuntimeBuilder().use_session_store(store).build()
runtime.start()
```

连接信息必须由宿主传入。Quasar 不读取 `REDIS_URL`、`MONGO_URL` 或 `.env`，也不会因为驱动已安装就自动启用持久化。

## 7. SQLite 向量检索与物体识别

```python
from Quasar.adapters.vector import SQLiteVectorStore, SQLiteVectorStoreConfig
from Quasar.cai import RuntimeBuilder, VectorQuery, VectorRecord

vectors = SQLiteVectorStore(SQLiteVectorStoreConfig(
    "./runtime/vectors.db",
    vector_dim=3,
))
runtime = RuntimeBuilder().use_vector_store(vectors).build()
runtime.start()

vectors.upsert([VectorRecord("sample", (1.0, 0.0, 0.0), {"label": "样例"})])
matches = vectors.search(VectorQuery((1.0, 0.0, 0.0), limit=5))
runtime.close()
```

物体识别依赖位于 `object-recognition` extra，`RecognitionConfig.enable` 和自动扫描默认均为 `False`。宿主应先显式装配模型、`VectorStore` 和 `ArtifactStore`，再安装自己的识别工具或插件；Core 不会在导入时扫描图片、加载 Torch 或创建 SQLite 文件。

## 8. 生命周期与健康检查

Runtime 按依赖顺序执行 `start()`，关闭时逆序执行 `flush()` 和 `close()`。`close()` 幂等；关闭后的 Runtime 拒绝新请求。

```python
runtime.start()
health = runtime.health()
print(health.status.value)
for name, component in health.components.items():
    print(name, component.status.value, component.message)

try:
    serve(runtime)
finally:
    runtime.close(timeout=30)
```

必需组件为 `unavailable` 时整体为 `unavailable`；可选组件不可用时整体为 `degraded`。适配器连接、持久化和 flush 失败分别使用 `AdapterConnectionError`、`PersistenceError` 和 `BufferFlushError` 等结构化异常。

## 9. 旧版迁移

旧会话 facade 暂时位于 `Quasar.compat.v1`，调用时会产生 `DeprecationWarning`：

```python
from Quasar.compat.v1 import LegacyConfigAdapter, get_session_cache_manager

config = LegacyConfigAdapter.from_dict({
    "chat": {"request_timeout": 120},
    "runtime": {"max_workers": 4},
})
manager = get_session_cache_manager(runtime)
```

新代码应改为 `RuntimeConfig`、`RuntimeBuilder` 和 `runtime.require_capability()`。`CAIRuntime` 不再自动加载旧 AI entrance 或旧 registry；需要旧 entrance 时使用 `CAIApp.from_legacy_entrance()` 显式传入。

## 10. JSUT Agent 的装配方式

JSUT Agent 只保留健康检查、Agent SSE 和文件下载接口。它使用 `RuntimeBuilder` 注册八个 message-only 工具；每次请求的 `message` 是唯一上下文。课程、资源、任务、批改和查重结果都不写数据库，文件结果写入服务器 `OUTPUT_FOLDER` 并返回下载路径。

生产 Compose 只有 Web 服务和输出文件卷，没有 PostgreSQL、Redis、Celery、迁移步骤或数据库备份。Quasar 提供 Redis/Mongo 适配器不表示 JSUT Agent 会启用它们；只有宿主显式构造并注入时才会连接。

## 11. 发布前检查

```bash
python -m pytest Quasar/tests -q
python -m pytest tests -q
```

应同时确认：基础依赖没有数据库/GPU/媒体/云 SDK；`import Quasar.cai` 未导入可选驱动；默认 Runtime 构造与关闭没有文件、网络或线程副作用；宿主代码没有导入旧全局工具 registry。
