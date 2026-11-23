"""运行时显存池 HTTP 路由（挂在 /runtime 前缀下）。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..planner.models import PlanningError
from .schemas import (
    CancelResponse,
    InferenceResponse,
    OpEventRequest,
    OpEventResponse,
    PlanOut,
    PoolResponse,
    WorkspaceRequest,
    WorkspaceResponse,
)
from .service import WorkspaceError

router = APIRouter(prefix="/runtime", tags=["runtime"])


def _service(request: Request):
    return request.app.state.runtime_service


def _plan_to_out(outcome) -> PlanOut:
    return PlanOut(
        feasible=outcome.feasible,
        total_bytes=outcome.total_bytes,
        budget_bytes=outcome.budget_bytes,
        offsets=outcome.offsets,
        added_wait=(
            {"waiter": outcome.added_wait.waiter, "waitee": outcome.added_wait.waitee}
            if outcome.added_wait is not None
            else None
        ),
        note=outcome.note,
        best_effort_total_bytes=outcome.best_effort_total_bytes,
        bins=outcome.bins,
    )


@router.post("/workspaces", response_model=WorkspaceResponse)
def open_workspace(req: WorkspaceRequest, request: Request):
    """按 /plan 请求格式申请工作区；失败不改变任何现有占用。"""
    payload = req.model_dump()
    service = _service(request)
    try:
        record, outcome = service.open_workspace(payload)
    except PlanningError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    if record is None:
        reason = (
            "POOL_FULL"
            if outcome.feasible
            else "PLAN_INFEASIBLE"
        )
        return JSONResponse(
            status_code=409,
            content=WorkspaceResponse(
                accepted=False,
                inference_id=None,
                reason=reason,
                plan=_plan_to_out(outcome),
            ).model_dump(),
        )

    return WorkspaceResponse(
        accepted=True,
        inference_id=record.inference_id,
        base=record.base,
        reason=None,
        plan=_plan_to_out(outcome),
    )


@router.post(
    "/workspaces/{inference_id}/ops/submit",
    response_model=OpEventResponse,
)
def submit_op(inference_id: str, req: OpEventRequest, request: Request):
    try:
        body = _service(request).submit_op(inference_id, req.op_id)
    except WorkspaceError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "detail": exc.message},
        )
    return OpEventResponse(**body)


@router.post(
    "/workspaces/{inference_id}/ops/complete",
    response_model=OpEventResponse,
)
def complete_op(inference_id: str, req: OpEventRequest, request: Request):
    try:
        body = _service(request).complete_op(inference_id, req.op_id)
    except WorkspaceError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "detail": exc.message},
        )
    return OpEventResponse(**body)


@router.post(
    "/workspaces/{inference_id}/cancel",
    response_model=CancelResponse,
)
def cancel(inference_id: str, request: Request):
    try:
        body = _service(request).cancel(inference_id)
    except WorkspaceError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "detail": exc.message},
        )
    return CancelResponse(**body)


@router.get("/pool", response_model=PoolResponse)
def pool_status(request: Request):
    return _service(request).describe_pool()


@router.get(
    "/workspaces/{inference_id}",
    response_model=InferenceResponse,
)
def inference_status(inference_id: str, request: Request):
    try:
        body = _service(request).describe_inference(inference_id)
    except WorkspaceError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "detail": exc.message},
        )
    return InferenceResponse(**body)


@router.post("/reset", status_code=204)
def reset(request: Request):
    """清空显存池与全部推理登记（仅供测试/演练复位）。"""
    _service(request).reset()
    return JSONResponse(status_code=204, content=None)
