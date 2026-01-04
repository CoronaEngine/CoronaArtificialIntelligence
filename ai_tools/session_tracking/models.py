"""
会话追踪数据模型

定义会话进度追踪和账户使用记录的数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepRetryInfo:
    """单步重试信息"""

    attempt_count: int  # 当前尝试次数（从1开始）
    max_attempts: int  # 最大尝试次数
    last_error: Optional[str] = None  # 最后一次失败原因
    retry_history: List[Dict[str, Any]] = field(default_factory=list)  # 重试历史记录

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "last_error": self.last_error,
            "retry_history": self.retry_history,
        }


@dataclass
class StepInfo:
    """步骤执行信息"""

    step: int
    name: str
    status: str  # pending/running/completed/failed/retrying
    started_at: Optional[str] = None  # ISO格式时间字符串
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = None
    retry_info: Optional[StepRetryInfo] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "retry_info": self.retry_info.to_dict() if self.retry_info else None,
            "metadata": self.metadata,
        }


@dataclass
class SessionProgress:
    """会话进度信息"""

    current_step: int = 0
    total_steps: int = 0
    step_name: str = ""
    step_message: str = ""
    progress_percent: float = 0.0
    is_retrying: bool = False
    current_attempt: int = 1
    max_attempts: int = 3
    steps_history: List[StepInfo] = field(default_factory=list)
    total_retries: int = 0
    estimated_remaining_seconds: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "step_name": self.step_name,
            "step_message": self.step_message,
            "progress_percent": self.progress_percent,
            "is_retrying": self.is_retrying,
            "current_attempt": self.current_attempt,
            "max_attempts": self.max_attempts,
            "steps_history": [step.to_dict() for step in self.steps_history],
            "total_retries": self.total_retries,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
        }


@dataclass
class AccountUsageRecord:
    """账户使用记录"""

    account_id: str
    account_name: str  # adapter_type
    model: Optional[str]
    timestamp: float
    price: float
    latency_ms: float
    success: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_id": self.account_id,
            "account_name": self.account_name,
            "model": self.model,
            "timestamp": self.timestamp,
            "price": self.price,
            "latency_ms": round(self.latency_ms, 2),
            "success": self.success,
        }


@dataclass
class DeadlineInfo:
    """Deadline 超时管理信息"""

    deadline: float  # 绝对截止时间（时间戳）
    start_time: float  # 开始时间（时间戳）
    deadline_seconds: float  # 总超时时间（秒）
    is_timeout: bool = False  # 是否超时
    elapsed_ms: Optional[float] = None  # 实际耗时（毫秒）

    # 阶段耗时记录
    stage_timings: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deadline": self.deadline,
            "start_time": self.start_time,
            "deadline_seconds": self.deadline_seconds,
            "is_timeout": self.is_timeout,
            "elapsed_ms": round(self.elapsed_ms, 2) if self.elapsed_ms else None,
            "stage_timings": self.stage_timings,
        }

    @property
    def stage_summary(self) -> Dict[str, Any]:
        """获取阶段耗时摘要"""
        if not self.stage_timings:
            return {}

        total_ms = sum(
            st.get("duration_ms", 0)
            for st in self.stage_timings
            if st.get("duration_ms")
        )
        by_stage: Dict[str, float] = {}
        for st in self.stage_timings:
            name = st.get("name", "unknown")
            duration = st.get("duration_ms", 0)
            if duration:
                by_stage[name] = by_stage.get(name, 0) + duration

        return {
            "total_ms": round(total_ms, 2),
            "stage_count": len(self.stage_timings),
            "by_stage": {k: round(v, 2) for k, v in by_stage.items()},
        }


@dataclass
class SessionCache:
    """扩展的会话缓存结构"""

    session_id: str
    state: str = "idle"  # idle/running/completed/failed/cancelled
    created_at: float = 0.0  # Unix 时间戳 (float)
    updated_at: float = 0.0  # Unix 时间戳 (float)

    # 输入信息
    input_type: str = ""  # chat/workflow/single
    input_parameters: Dict[str, Any] = field(default_factory=dict)

    # 工作流初始状态字段（从 WorkflowState 提取）
    function_id: Optional[int] = None
    prompt: str = ""
    images: List[str] = field(default_factory=list)
    additional_type: Optional[List[str]] = None
    bounding_box: Optional[List[List[Dict[str, Any]]]] = None
    resolution: str = "1:1"
    image_size: str = "2K"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 进度信息
    progress: SessionProgress = field(default_factory=SessionProgress)

    # 输出信息
    outputs: List[Dict[str, Any]] = field(default_factory=list)

    # 错误信息
    error_message: Optional[str] = None

    # 账户使用记录
    account_usages: List[AccountUsageRecord] = field(default_factory=list)

    # Deadline 超时管理信息
    deadline_info: Optional[DeadlineInfo] = None

    @property
    def created_at_iso(self) -> str:
        """获取 ISO 格式的创建时间"""
        if not self.created_at:
            return ""
        from datetime import datetime

        return datetime.fromtimestamp(self.created_at).isoformat()

    @property
    def updated_at_iso(self) -> str:
        """获取 ISO 格式的更新时间"""
        if not self.updated_at:
            return ""
        from datetime import datetime

        return datetime.fromtimestamp(self.updated_at).isoformat()

    @property
    def total_cost(self) -> float:
        """计算总费用（仅成功的调用）"""
        return sum(r.price for r in self.account_usages if r.success)

    @property
    def usage_summary(self) -> Dict[str, Any]:
        """获取使用摘要统计"""
        successful = [r for r in self.account_usages if r.success]
        by_account: Dict[str, Dict[str, Any]] = {}

        for record in successful:
            key = record.account_id
            if key not in by_account:
                by_account[key] = {
                    "account_id": record.account_id,
                    "account_name": record.account_name,
                    "model": record.model,
                    "count": 0,
                    "total_cost": 0.0,
                    "avg_latency": 0.0,
                    "total_latency": 0.0,
                }
            by_account[key]["count"] += 1
            by_account[key]["total_cost"] += record.price
            by_account[key]["total_latency"] += record.latency_ms

        # 计算平均延迟
        for stats in by_account.values():
            if stats["count"] > 0:
                stats["avg_latency"] = round(stats["total_latency"] / stats["count"], 2)
            del stats["total_latency"]  # 移除中间计算字段

        return {
            "total_calls": len(self.account_usages),
            "successful_calls": len(successful),
            "failed_calls": len(self.account_usages) - len(successful),
            "total_cost": round(self.total_cost, 4),
            "by_account": list(by_account.values()),
        }

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "session_id": self.session_id,
            "state": self.state,
            "created_at": self.created_at,
            "created_at_iso": self.created_at_iso,
            "updated_at": self.updated_at,
            "updated_at_iso": self.updated_at_iso,
            "input_type": self.input_type,
            "input_parameters": self.input_parameters,
            "progress": self.progress.to_dict(),
            "outputs": self.outputs,
            "error_message": self.error_message,
            "account_usages": [r.to_dict() for r in self.account_usages],
            "usage_summary": self.usage_summary,
            "deadline_info": (
                self.deadline_info.to_dict() if self.deadline_info else None
            ),
        }

        # 添加工作流初始状态字段（如果有值）
        if self.function_id is not None:
            result["function_id"] = self.function_id
        if self.prompt:
            result["prompt"] = self.prompt
        if self.images:
            result["images"] = self.images
        if self.additional_type:
            result["additional_type"] = self.additional_type
        if self.bounding_box:
            result["bounding_box"] = self.bounding_box
        if self.resolution:
            result["resolution"] = self.resolution
        if self.image_size:
            result["image_size"] = self.image_size
        if self.metadata:
            result["metadata"] = self.metadata

        return result


__all__ = [
    "StepRetryInfo",
    "StepInfo",
    "SessionProgress",
    "AccountUsageRecord",
    "DeadlineInfo",
    "SessionCache",
]
