"""
Agent 执行模块
负责 Agent 的创建、运行和备用完成逻辑

并发安全说明：
- Agent 本身是无状态的（状态在 messages 中传递）
- 使用 RLock 保护 Agent 创建过程
- 支持多用户并发调用

工具获取：
- 优先从 ToolRegistry 获取已注册工具（需先执行 discover）
- 如果 ToolRegistry 为空，回退到 load_tools() 直接加载

LLM 获取：
- 使用 pool.get_chat_model() 统一入口
- 池模式：从 InnerAgentWorkflow.ai_pool 的 CHAT 池中获取
- 降级模式：使用 AIConfig 配置的单例客户端
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, SystemMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call

from ai_config.ai_config import AIConfig, get_ai_config
from ai_models.pool import get_chat_model
from ai_tools.registry import get_tool_registry


logger = logging.getLogger(__name__)

_CACHED_AGENT: Any = None
_AGENT_LOCK = threading.RLock()  # 保护 Agent 创建


def _should_retry(error: Exception) -> bool:
    """
    判断是否应该重试的错误类型

    处理以下情况：
    - TypeError: API 返回空 choices (Received response with null value for `choices`)
    - 网络超时和临时错误
    """
    # TypeError: 通常是 API 返回 null choices
    if isinstance(error, TypeError):
        error_msg = str(error).lower()
        if "choices" in error_msg or "null" in error_msg:
            return True

    # 检查常见的可重试错误
    error_msg = str(error).lower()
    retryable_keywords = [
        "timeout",
        "rate limit",
        "429",
        "503",
        "502",
        "connection",
        "temporary",
    ]
    return any(keyword in error_msg for keyword in retryable_keywords)


@wrap_model_call
def _retry_middleware(request, handler):
    """
    自定义重试中间件，处理 API 临时失败（如空 choices）

    最多重试 2 次，使用指数退避策略
    """
    max_retries = 2
    initial_delay = 1.0
    backoff_factor = 2.0

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return handler(request)
        except Exception as e:
            last_error = e
            if attempt < max_retries and _should_retry(e):
                delay = initial_delay * (backoff_factor**attempt)
                logger.warning(
                    f"模型调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}，"
                    f"{delay:.1f}秒后重试..."
                )
                time.sleep(delay)
            else:
                raise

    # 不应到达这里，但为了类型安全
    raise last_error  # type: ignore


def _build_agent(config: AIConfig) -> Any:
    """构建 Agent 实例（内部函数）"""
    chat_cfg = config.chat

    # 使用 pool 系统获取 LLM（自动检测池模式或降级模式）
    llm = get_chat_model(
        provider_name=chat_cfg.provider,
        model_name=chat_cfg.model,
        temperature=chat_cfg.temperature,
        request_timeout=chat_cfg.request_timeout,
    )

    # 从 ToolRegistry 获取工具
    # 如果已执行过 discover()，直接获取已注册工具
    # 否则触发 discover() 加载工具
    registry = get_tool_registry()
    if not registry.list_tools():
        # 尚未发现工具，执行发现
        registry.discover(config)

    tools = registry.list_tools()
    logger.debug(f"Agent 使用 {len(tools)} 个工具: {[t.name for t in tools]}")

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=chat_cfg.system_prompt,
        middleware=[_retry_middleware],
    )


def create_default_agent(force_reload: bool = False) -> Any:
    """
    创建或获取缓存的默认 Agent（线程安全）

    LangChain 的 ReAct Agent 本身是无状态的：
    - 状态（messages）在每次调用时传入
    - Agent 只负责推理和工具调用
    - 因此可以安全地在多线程间共享
    """
    global _CACHED_AGENT

    # 双重检查锁定模式
    if _CACHED_AGENT is None or force_reload:
        with _AGENT_LOCK:
            if _CACHED_AGENT is None or force_reload:
                _CACHED_AGENT = _build_agent(get_ai_config())

    return _CACHED_AGENT


def _is_connection_error(error: Exception) -> bool:
    """判断是否为连接错误（需要切换账号重试）"""
    error_msg = str(error).lower()
    connection_keywords = ["connection", "timeout", "unreachable", "refused"]
    return any(keyword in error_msg for keyword in connection_keywords)


def run_agent(messages: List[BaseMessage]) -> Dict[str, Any]:
    """
    运行 agent，接受标准的 LangChain BaseMessage 列表

    当遇到连接错误时，会强制重建 Agent（切换账号）并重试。

    Args:
        messages: LangChain BaseMessage 列表（HumanMessage, AIMessage 等）

    Returns:
        Agent 执行结果 {"messages": [...]}
    """
    max_account_switches = 2  # 最多切换账号次数
    last_error = None

    for switch_attempt in range(max_account_switches + 1):
        try:
            # 首次尝试或切换账号后重建 Agent
            agent = create_default_agent(force_reload=(switch_attempt > 0))
            return agent.invoke({"messages": messages})

        except Exception as e:
            last_error = e
            if switch_attempt < max_account_switches and _is_connection_error(e):
                logger.warning(
                    f"连接错误，切换账号重试 ({switch_attempt + 1}/{max_account_switches}): {e}"
                )
                time.sleep(1.0)  # 短暂等待后切换账号
            else:
                raise

    # 所有账号都失败
    if last_error:
        raise last_error
    return {"messages": []}


def stream_agent(messages: List[BaseMessage]):
    """
    流式运行 agent，逐步返回 AIMessage + 关联的 ToolMessage 组合

    当遇到连接错误时，会强制重建 Agent（切换账号）并重试。

    Args:
        messages: LangChain BaseMessage 列表

    Yields:
        每个 yield 包含：{"messages": [AIMessage, ToolMessage, ...]}
        - 每次 yield 一个完整的推理步骤（AI思考 + 工具调用结果）
    """
    max_account_switches = 2  # 最多切换账号次数
    last_error = None

    for switch_attempt in range(max_account_switches + 1):
        try:
            # 首次尝试或切换账号后重建 Agent
            agent = create_default_agent(force_reload=(switch_attempt > 0))

            # 使用 stream_mode="updates" 获取每个节点的更新
            for chunk in agent.stream({"messages": messages}, stream_mode="updates"):
                # chunk 格式: {"node_name": {"messages": [new_messages]}}
                yield chunk

            # 成功完成，退出循环
            return

        except Exception as e:
            last_error = e
            if switch_attempt < max_account_switches and _is_connection_error(e):
                logger.warning(
                    f"连接错误，切换账号重试 ({switch_attempt + 1}/{max_account_switches}): {e}"
                )
                time.sleep(1.0)  # 短暂等待后切换账号
            else:
                raise

    # 所有账号都失败
    if last_error:
        raise last_error


def fallback_completion(history: List[BaseMessage]) -> str:
    """
    备用完成方法：直接使用 LLM 而不经过 agent

    Args:
        history: 对话历史

    Returns:
        LLM 生成的文本内容
    """
    cfg = get_ai_config()
    chat_cfg = cfg.chat

    # 使用 pool 系统获取 LLM（自动检测池模式或降级模式）
    llm = get_chat_model(
        provider_name=chat_cfg.provider,
        model_name=chat_cfg.model,
        temperature=chat_cfg.temperature,
        request_timeout=chat_cfg.request_timeout,
    )

    # 添加系统提示
    prompt_messages: List[BaseMessage] = [
        SystemMessage(content=chat_cfg.system_prompt),
        *history,
    ]

    ai_message = llm.invoke(prompt_messages)
    content = ai_message.content or ""

    # content 为数组时提取 text
    if isinstance(content, list):
        content = "\n".join([b["text"] for b in content if b.get("type") == "text"])

    print(f"[Fallback] {content}")
    return content


__all__ = [
    "create_default_agent",
    "run_agent",
    "stream_agent",
    "fallback_completion",
]
