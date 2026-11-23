"""/runtime 路由的请求与响应模型（请求格式与 /plan 保持一致）。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class WaitIn(BaseModel):
    waiter: str = Field(..., description="等待方算子 id（该算子需等 waitee 完成）")
    waitee: str = Field(..., description="被等待方算子 id")


class TensorIn(BaseModel):
    name: str = Field(..., description="张量 id")
    producer: str = Field(..., description="生产算子 id")
    consumers: list[str] = Field(..., min_length=1, description="全部消费算子 id")
    size_bytes: int = Field(..., gt=0, description="张量字节数")
    alignment: int = Field(..., gt=0, description="偏移对齐要求，必须为 2 的幂")


class WorkspaceRequest(BaseModel):
    streams: dict[str, list[str]] = Field(
        ...,
        description="两条 CUDA 流，key 为流 id，value 为按提交顺序排列的算子 id",
    )
    waits: list[WaitIn] = Field(default_factory=list, description="已有跨流等待")
    tensors: list[TensorIn] = Field(..., min_length=1, description="待规划张量，最多 8 个")
    budget_bytes: int = Field(..., gt=0, description="显存预算（字节）")


class AddedWaitOut(BaseModel):
    waiter: str
    waitee: str


class PlanOut(BaseModel):
    feasible: bool
    total_bytes: int
    budget_bytes: int
    offsets: dict[str, int]
    added_wait: AddedWaitOut | None
    note: str
    best_effort_total_bytes: int | None = None
    bins: list[list[str]]


class WorkspaceResponse(BaseModel):
    accepted: bool
    inference_id: str | None
    base: int | None = Field(None, description="工作区在共享显存池中的基址（字节）")
    reason: str | None = Field(None, description="未受理原因：PLAN_INFEASIBLE / POOL_FULL")
    plan: PlanOut


class OpEventRequest(BaseModel):
    op_id: str = Field(..., description="规划中的算子 id")


class OpEventResponse(BaseModel):
    inference_id: str
    op_id: str
    status: str
    intervals: list[list[int]] | None = None
    inference_state: str | None = None
    workspace_freed: bool | None = None


class CancelResponse(BaseModel):
    inference_id: str
    status: str
    already_terminal: bool
    retained_intervals: list[list[int]]
    inflight_ops: list[str] | None = None
    freed_bytes: int | None = None


class SegmentOut(BaseModel):
    start: int
    end: int
    owner: str | None
    refs: int


class PoolResponse(BaseModel):
    pool_size: int
    free_bytes: int
    largest_free_run: int
    free_runs: list[list[int]]
    segments: list[SegmentOut]


class OpOut(BaseModel):
    op_id: str
    submitted: bool
    completed: bool
    intervals: list[list[int]]


class InferenceResponse(BaseModel):
    inference_id: str
    state: str
    base: int
    total_bytes: int
    retained_intervals: list[list[int]]
    ops: list[OpOut]
