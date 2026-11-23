"""FastAPI 离线显存规划 + 运行时共享显存池演练接口。"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .planner.models import PlanningError, PlanningInput, TensorSpec, Wait
from .planner.service import plan
from .runtime.router import router as runtime_router
from .runtime.service import RuntimeService

# 共享显存池容量（字节），可通过环境变量 POOL_SIZE_BYTES 配置
DEFAULT_POOL_SIZE = 16 * 1024 * 1024


def _load_pool_size() -> int:
    raw = os.environ.get("POOL_SIZE_BYTES")
    if raw is None or raw.strip() == "":
        return DEFAULT_POOL_SIZE
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"POOL_SIZE_BYTES={raw!r} 不是正整数"
        ) from exc
    if value <= 0:
        raise RuntimeError("POOL_SIZE_BYTES 必须为正整数")
    return value

app = FastAPI(
    title="CUDA 双流显存规划与单进程共享显存池演练接口",
    version="1.1.0",
    description=(
        "给定两条 CUDA 流的算子顺序、跨流等待与张量生命周期，"
        "在显存预算内规划各张量的字节偏移；必要时新增一条不成环的跨流等待。"
        "/runtime 路由按规划结果在单进程共享显存池中受理推理工作区，"
        "登记算子提交/完成并支持取消，模拟取消后在途算子仍占用显存的行为。"
    ),
)

app.state.runtime_service = RuntimeService(_load_pool_size())
app.include_router(runtime_router)


class WaitIn(BaseModel):
    waiter: str = Field(..., description="等待方算子 id（该算子需等 waitee 完成）")
    waitee: str = Field(..., description="被等待方算子 id")


class TensorIn(BaseModel):
    name: str = Field(..., description="张量 id")
    producer: str = Field(..., description="生产算子 id")
    consumers: list[str] = Field(..., min_length=1, description="全部消费算子 id")
    size_bytes: int = Field(..., gt=0, description="张量字节数")
    alignment: int = Field(..., gt=0, description="偏移对齐要求，必须为 2 的幂")


class PlanRequest(BaseModel):
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


class PlanResponse(BaseModel):
    feasible: bool
    total_bytes: int
    budget_bytes: int
    offsets: dict[str, int]
    added_wait: AddedWaitOut | None
    note: str
    best_effort_total_bytes: int | None = None
    bins: list[list[str]]


def _to_internal(req: PlanRequest) -> PlanningInput:
    return PlanningInput(
        streams={sid: list(ops) for sid, ops in req.streams.items()},
        waits=[Wait(w.waiter, w.waitee) for w in req.waits],
        tensors=[
            TensorSpec(
                name=t.name,
                producer=t.producer,
                consumers=tuple(t.consumers),
                size=t.size_bytes,
                alignment=t.alignment,
            )
            for t in req.tensors
        ],
        budget_bytes=req.budget_bytes,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/plan", response_model=PlanResponse)
def post_plan(req: PlanRequest) -> PlanResponse | JSONResponse:
    data = _to_internal(req)
    try:
        outcome = plan(data)
    except PlanningError as exc:
        # 输入不合法（结构问题由 pydantic 拦截，这里是语义校验失败）
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    return PlanResponse(
        feasible=outcome.feasible,
        total_bytes=outcome.total_bytes,
        budget_bytes=outcome.budget_bytes,
        offsets=outcome.offsets,
        added_wait=(
            AddedWaitOut(
                waiter=outcome.added_wait.waiter,
                waitee=outcome.added_wait.waitee,
            )
            if outcome.added_wait is not None
            else None
        ),
        note=outcome.note,
        best_effort_total_bytes=outcome.best_effort_total_bytes,
        bins=outcome.bins,
    )
