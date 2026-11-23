"""推理工作区生命周期服务。

语义（与需求逐条对应）：

- 申请工作区沿用 ``/plan`` 的请求格式：先离线规划，布局可行时按
  ``total_bytes`` 在共享显存池里申请一块满足全部张量对齐要求的连续空间；
  只要还有合适的空段就受理，申请失败（排不下或池里没有合适空段）时
  池内原有占用完全不变。
- 凭推理号登记算子的“提交 / 完成”，也可取消整笔推理。
- 正常推理在全部算子完成后整体归还工作区。
- 取消生效后拒绝该笔推理的新算子提交；只保留“已提交且未完成”的算子
  仍在读写的字节区间，其余空间立即开放给新推理。
- 多个在途算子共用的字节按引用计数保留，等最后一个完成后才释放；
  迟到或重复的完成记录被拒绝，不能释放其他推理的占用。
- 规划/分配与完成/取消登记在同一把进程内锁下串行化：计算布局期间仍可
  登记完成和取消，分配以拿到锁那一刻的池占用为准。
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field

from ..planner.models import PlanningInput, TensorSpec, Wait
from ..planner.service import plan
from .pool import MemoryPool, _union


class WorkspaceError(Exception):
    """运行时操作被拒绝。``code`` 映射到 HTTP 状态码。"""

    def __init__(self, code: str, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message


# 推理状态
ACTIVE = "active"        # 正常受理中
CANCELLED = "cancelled"  # 已取消，拒绝新提交，等待在途算子收尾
RELEASED = "released"    # 工作区已归还，记录仅用于拒绝迟到事件


@dataclass
class OpRecord:
    op_id: str
    intervals: list[tuple[int, int]]
    submitted: bool = False
    completed: bool = False


@dataclass
class InferenceRecord:
    inference_id: str
    base: int
    total_bytes: int
    plan_outcome: object
    request: dict
    ops: dict[str, OpRecord]
    state: str = ACTIVE
    retained_intervals: list[tuple[int, int]] = field(default_factory=list)


class RuntimeService:
    """单进程显存池 + 全部推理工作区的登记处。"""

    def __init__(self, pool_size: int) -> None:
        self.pool = MemoryPool(pool_size)
        self.pool_size = pool_size
        self._inferences: dict[str, InferenceRecord] = {}
        self._id_counter = itertools.count(1)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def to_planning_input(payload: dict) -> PlanningInput:
        return PlanningInput(
            streams={sid: list(ops) for sid, ops in payload["streams"].items()},
            waits=[
                Wait(w["waiter"], w["waitee"]) for w in payload.get("waits", [])
            ],
            tensors=[
                TensorSpec(
                    name=t["name"],
                    producer=t["producer"],
                    consumers=tuple(t["consumers"]),
                    size=t["size_bytes"],
                    alignment=t["alignment"],
                )
                for t in payload["tensors"]
            ],
            budget_bytes=payload["budget_bytes"],
        )

    @staticmethod
    def _op_intervals(
        data: PlanningInput,
        base: int,
        offsets: dict[str, int],
    ) -> dict[str, list[tuple[int, int]]]:
        """每个算子在池中的绝对字节区间（其生产或消费的全部张量并集）。

        规划中出现但不接触任何张量的算子登记为空区间：仍可提交/完成，
        但不持有任何显存引用。
        """
        raw: dict[str, list[tuple[int, int]]] = {
            op_id: [] for op_id in data.all_operators()
        }
        for t in data.tensors:
            start = base + offsets[t.name]
            interval = (start, start + t.size)
            for op_id in (t.producer, *t.consumers):
                raw.setdefault(op_id, []).append(interval)
        return {op_id: _union(iv) for op_id, iv in raw.items()}

    def _get(self, inference_id: str) -> InferenceRecord:
        record = self._inferences.get(inference_id)
        if record is None:
            raise WorkspaceError(
                "NOT_FOUND", f"推理号 {inference_id!r} 不存在", 404
            )
        return record

    # ------------------------------------------------------------------ #
    # 1. 申请工作区
    # ------------------------------------------------------------------ #
    def open_workspace(self, payload: dict) -> tuple[InferenceRecord | None, object]:
        """规划并申请工作区。

        返回 ``(记录, 规划结果)``：规划不可行或池内没有合适空段时记录为
        ``None``，且池内占用保持不变。输入语义错误抛 :class:`PlanningError`。
        """
        # 规划本身是纯函数，放在锁外执行；计算布局期间别的线程仍可登记
        # 完成/取消。真正分配时再进锁、以当时的池占用为准。
        data = self.to_planning_input(payload)
        outcome = plan(data)
        if not outcome.feasible:
            return None, outcome

        required_align = max(t.alignment for t in data.tensors)
        inference_id = f"inf-{next(self._id_counter):06d}"
        with self._lock:
            base = self.pool.allocate(
                size=outcome.total_bytes,
                alignment=required_align,
                owner=inference_id,
            )
            if base is None:
                # 申请失败：分配未发生，池内原有占用保持不变；
                # 推理号作废，下一笔申请从新号开始
                return None, outcome

            op_intervals = self._op_intervals(
                data, base, outcome.offsets
            )
            record = InferenceRecord(
                inference_id=inference_id,
                base=base,
                total_bytes=outcome.total_bytes,
                plan_outcome=outcome,
                request=payload,
                ops={
                    op_id: OpRecord(op_id=op_id, intervals=intervals)
                    for op_id, intervals in op_intervals.items()
                },
            )
            self._inferences[inference_id] = record
        return record, outcome

    # ------------------------------------------------------------------ #
    # 2. 算子提交
    # ------------------------------------------------------------------ #
    def submit_op(self, inference_id: str, op_id: str) -> dict[str, object]:
        with self._lock:
            record = self._get(inference_id)
            if record.state == RELEASED:
                raise WorkspaceError(
                    "INFERENCE_RELEASED",
                    f"推理 {inference_id!r} 的工作区已归还，拒绝迟到提交",
                )
            if record.state == CANCELLED:
                raise WorkspaceError(
                    "INFERENCE_CANCELLED",
                    f"推理 {inference_id!r} 已取消，拒绝新算子提交",
                )
            op = record.ops.get(op_id)
            if op is None:
                raise WorkspaceError(
                    "UNKNOWN_OP",
                    f"算子 {op_id!r} 不属于推理 {inference_id!r} 的规划",
                    422,
                )
            if op.submitted:
                raise WorkspaceError(
                    "DUPLICATE_SUBMIT",
                    f"算子 {op_id!r} 已提交，重复提交被拒绝",
                )
            op.submitted = True
            for start, end in op.intervals:
                self.pool.add_refs(start, end)
            return {
                "inference_id": inference_id,
                "op_id": op_id,
                "status": "submitted",
                "intervals": [list(iv) for iv in op.intervals],
            }

    # ------------------------------------------------------------------ #
    # 3. 算子完成
    # ------------------------------------------------------------------ #
    def complete_op(self, inference_id: str, op_id: str) -> dict[str, object]:
        with self._lock:
            record = self._get(inference_id)
            if record.state == RELEASED:
                raise WorkspaceError(
                    "INFERENCE_RELEASED",
                    f"推理 {inference_id!r} 的工作区已归还，拒绝迟到完成记录",
                )
            op = record.ops.get(op_id)
            if op is None:
                raise WorkspaceError(
                    "UNKNOWN_OP",
                    f"算子 {op_id!r} 不属于推理 {inference_id!r} 的规划",
                    422,
                )
            if not op.submitted:
                # 未提交就完成（或取消后补报的“完成”）：不允许，绝不动用池
                raise WorkspaceError(
                    "OP_NOT_SUBMITTED",
                    f"算子 {op_id!r} 从未提交，完成记录被拒绝",
                )
            if op.completed:
                # 重复完成：拒绝，不能重复释放字节（尤其不能碰到别的推理）
                raise WorkspaceError(
                    "DUPLICATE_COMPLETE",
                    f"算子 {op_id!r} 已完成，重复完成记录被拒绝",
                )

            op.completed = True
            for start, end in op.intervals:
                self.pool.release_refs(start, end)

            workspace_freed = False
            if record.state == CANCELLED:
                # 只释放本推理名下已无在途引用的段；仍被其他在途算子
                # 共用的字节（refs > 0）继续保留
                self.pool.release_unreferenced(inference_id)
                if not self._has_inflight(record):
                    record.state = RELEASED
                    record.retained_intervals = []
                    workspace_freed = True
            elif self._all_done(record):
                # 正常推理：全部算子完成，整体归还工作区
                self.pool.release_owner(inference_id)
                record.state = RELEASED
                workspace_freed = True

            return {
                "inference_id": inference_id,
                "op_id": op_id,
                "status": "completed",
                "inference_state": record.state,
                "workspace_freed": workspace_freed,
            }

    # ------------------------------------------------------------------ #
    # 4. 取消整笔推理
    # ------------------------------------------------------------------ #
    def cancel(self, inference_id: str) -> dict[str, object]:
        with self._lock:
            record = self._get(inference_id)
            already = record.state != ACTIVE
            if record.state == RELEASED:
                return {
                    "inference_id": inference_id,
                    "status": "released",
                    "already_terminal": True,
                    "retained_intervals": [],
                    "freed_bytes": 0,
                }

            inflight = [
                op for op in record.ops.values()
                if op.submitted and not op.completed
            ]
            keep = _union(
                iv for op in inflight for iv in op.intervals
            )
            retained = self.pool.prune_owner(inference_id, keep)
            record.state = CANCELLED
            record.retained_intervals = retained
            if not inflight:
                # 没有任何在途算子：工作区即刻全部归还
                record.state = RELEASED
                record.retained_intervals = []

            freed = record.total_bytes - sum(
                e - s for s, e in retained
            )
            return {
                "inference_id": inference_id,
                "status": record.state,
                "already_terminal": already,
                "retained_intervals": [list(iv) for iv in retained],
                "inflight_ops": [op.op_id for op in inflight],
                "freed_bytes": freed,
            }

    # ------------------------------------------------------------------ #
    # 观测 / 测试辅助
    # ------------------------------------------------------------------ #
    def describe_pool(self) -> dict[str, object]:
        with self._lock:
            segments = self.pool.snapshot()
            return {
                "pool_size": self.pool_size,
                "free_bytes": self.pool.free_bytes(),
                "largest_free_run": self.pool.largest_free_run(),
                "free_runs": [list(run) for run in self.pool.free_runs()],
                "segments": [seg.to_dict() for seg in segments],
            }

    def describe_inference(self, inference_id: str) -> dict[str, object]:
        with self._lock:
            record = self._get(inference_id)
            return {
                "inference_id": record.inference_id,
                "state": record.state,
                "base": record.base,
                "total_bytes": record.total_bytes,
                "retained_intervals": [
                    list(iv) for iv in record.retained_intervals
                ],
                "ops": [
                    {
                        "op_id": op.op_id,
                        "submitted": op.submitted,
                        "completed": op.completed,
                        "intervals": [list(iv) for iv in op.intervals],
                    }
                    for op in record.ops.values()
                ],
            }

    def reset(self) -> None:
        """清空全部占用（仅供测试/演练复位使用）。"""
        with self._lock:
            self.pool = MemoryPool(self.pool_size)
            self._inferences.clear()
            self._id_counter = itertools.count(1)

    @staticmethod
    def _has_inflight(record: InferenceRecord) -> bool:
        return any(
            op.submitted and not op.completed for op in record.ops.values()
        )

    @staticmethod
    def _all_done(record: InferenceRecord) -> bool:
        # 规划里登记的全部算子都已提交且完成
        return all(
            op.submitted and op.completed for op in record.ops.values()
        )
