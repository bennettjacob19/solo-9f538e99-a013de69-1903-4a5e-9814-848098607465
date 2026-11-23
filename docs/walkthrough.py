#!/usr/bin/env python3
"""单进程共享显存池演练（可直接运行，不依赖 HTTP 服务）。

用法：
    python docs/walkthrough.py                # 用默认小池（4096 字节）
    POOL_SIZE_BYTES=8192 python docs/walkthrough.py

演练覆盖需求中的每个关键语义：
  1. 按 /plan 请求格式申请工作区，返回推理号、规划与满足全部张量对齐的基址；
  2. 凭推理号登记算子提交/完成，全部完成后正常归还工作区；
  3. 取消后拒绝新提交，只留在途算子仍在读写的字节，其余立即开放；
  4. 多个在途算子共用字节的引用计数：最后一个完成才释放；
  5. 迟到/重复完成记录被拒绝，不能释放其他推理的占用；
  6. 池满或规划无解时申请失败，原占用保持不变；
  7. 布局计算期间完成/取消仍可登记，分配以当时的池占用为准（并发演示）。
"""

from __future__ import annotations

import os
import sys
import threading
import time

# 允许从任意目录直接运行本脚本（含 python -I 模式）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.runtime.service import RuntimeService

POOL_SIZE = int(os.environ.get("POOL_SIZE_BYTES", "4096"))


def payload(budget: int = 4096) -> dict:
    """两条流、三个张量；T1/T2 生命周期不相交可复用同一 bin。"""
    return {
        "streams": {
            "stream_a": ["a1", "a2", "a3", "a4"],
            "stream_b": ["b1", "b2"],
        },
        "waits": [],
        "tensors": [
            {"name": "T1", "producer": "a1", "consumers": ["a2"],
             "size_bytes": 1024, "alignment": 256},
            {"name": "T2", "producer": "a2", "consumers": ["a3"],
             "size_bytes": 512, "alignment": 128},
            {"name": "T3", "producer": "b1", "consumers": ["b2"],
             "size_bytes": 512, "alignment": 512},
        ],
        "budget_bytes": budget,
    }


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def show_pool(svc: RuntimeService, note: str) -> None:
    info = svc.describe_pool()
    print(f"[{note}] 空闲 {info['free_bytes']}/{info['pool_size']} 字节，"
          f"空闲段 {info['free_runs']}")
    for seg in info["segments"]:
        tag = seg["owner"] or "空闲"
        extra = f"，在途引用 x{seg['refs']}" if seg["refs"] else ""
        print(f"    [{seg['start']:>5}, {seg['end']:>5}) {tag}{extra}")


def main() -> None:
    svc = RuntimeService(POOL_SIZE)
    print(f"共享显存池容量：{POOL_SIZE} 字节（单进程模拟一块连续显存）")

    # ---------------------------------------------------------------- #
    banner("1. 申请工作区：规划可行 -> 连续分配，返回推理号/规划/对齐基址")
    # ---------------------------------------------------------------- #
    record, outcome = svc.open_workspace(payload())
    assert record is not None
    inf_a = record.inference_id
    print(f"推理号：{inf_a}")
    print(f"规划占用 total_bytes={outcome.total_bytes}，"
          f"张量偏移={outcome.offsets}，工作区基址 base={record.base}")
    for t in payload()["tensors"]:
        absolute = record.base + outcome.offsets[t["name"]]
        assert absolute % t["alignment"] == 0
        print(f"    {t['name']}: 绝对地址 {absolute}，"
              f"对齐 {t['alignment']} OK，区间 "
              f"[{absolute}, {absolute + t['size_bytes']})")
    show_pool(svc, "A 已受理")

    # ---------------------------------------------------------------- #
    banner("2. 算子提交/完成：引用计数随在途算子增减，全完成后整体归还")
    # ---------------------------------------------------------------- #
    svc.submit_op(inf_a, "b1")
    show_pool(svc, "A 的 b1 在途")
    svc.complete_op(inf_a, "b1")
    show_pool(svc, "b1 完成（字节仍是 A 的预留，引用归零）")

    # ---------------------------------------------------------------- #
    banner("3. 取消推理 B：拒绝新提交，只留在途字节，其余立即开放")
    # ---------------------------------------------------------------- #
    record_b, _ = svc.open_workspace(payload())
    assert record_b is not None
    inf_b = record_b.inference_id
    print(f"第二笔推理：{inf_b}，基址 {record_b.base}")
    svc.submit_op(inf_b, "a1")   # 只提交 a1，覆盖 [base, base+1024)
    svc.submit_op(inf_b, "a2")   # a2 与 a1 共用同一 bin 的字节
    cancel = svc.cancel(inf_b)
    print(f"取消 {inf_b}：状态={cancel['status']}，"
          f"保留区间={cancel['retained_intervals']}，"
          f"立即开放 {cancel['freed_bytes']} 字节，"
          f"在途算子={cancel['inflight_ops']}")
    show_pool(svc, "B 已取消")

    # 取消生效后新提交被拒
    try:
        svc.submit_op(inf_b, "a3")
    except Exception as exc:  # noqa: BLE001
        print(f"取消后提交 a3 被拒绝：{exc}")

    # 开放出来的空间立刻能被新推理使用
    small = {
        "streams": {"s1": ["x1", "x2"], "s2": ["y1"]},
        "waits": [],
        "tensors": [
            {"name": "X", "producer": "x1", "consumers": ["x2"],
             "size_bytes": 512, "alignment": 256}
        ],
        "budget_bytes": 1024,
    }
    record_c, _ = svc.open_workspace(small)
    assert record_c is not None
    print(f"新推理 {record_c.inference_id} 落到刚开放的区间，"
          f"基址={record_c.base}")

    # ---------------------------------------------------------------- #
    banner("4. 共用字节等最后一个在途算子完成才释放")
    # ---------------------------------------------------------------- #
    svc.complete_op(inf_b, "a1")
    show_pool(svc, "B 的 a1 完成（a2 仍在引用共用字节）")
    svc.complete_op(inf_b, "a2")
    show_pool(svc, "B 的 a2 完成（最后一个，保留区间归还）")
    assert svc._inferences[inf_b].state == "released"

    # ---------------------------------------------------------------- #
    banner("5. 迟到/重复完成记录被拒绝，不能释放别人的占用")
    # ---------------------------------------------------------------- #
    try:
        svc.complete_op(inf_b, "a2")
    except Exception as exc:  # noqa: BLE001
        print(f"工作区归还后的迟到完成被拒绝：{exc}")
    try:
        svc.complete_op(inf_a, "a999")
    except Exception as exc:  # noqa: BLE001
        print(f"未知算子完成被拒绝：{exc}")
    show_pool(svc, "池占用未受非法事件影响")

    # 新推理 C 正常收尾归还
    svc.submit_op(record_c.inference_id, "x1")
    svc.submit_op(record_c.inference_id, "x2")
    svc.submit_op(record_c.inference_id, "y1")
    svc.complete_op(record_c.inference_id, "x1")
    svc.complete_op(record_c.inference_id, "x2")
    svc.complete_op(record_c.inference_id, "y1")

    # A 也走完正常流程（b1 此前已提交并完成）
    for op in ("a1", "a2", "a3", "a4", "b2"):
        svc.submit_op(inf_a, op)
    for op in ("a1", "a2", "a3", "a4", "b2"):
        svc.complete_op(inf_a, op)
    show_pool(svc, "A/C 全部完成，工作区归还")

    # ---------------------------------------------------------------- #
    banner("6. 申请失败保留原占用：池满 / 规划无解")
    # ---------------------------------------------------------------- #
    svc.reset()
    tiny = RuntimeService(2048)
    rec, _ = tiny.open_workspace(payload())
    assert rec is not None
    before = tiny.describe_pool()
    rejected, outcome2 = tiny.open_workspace(payload())
    assert rejected is None and outcome2.feasible
    print("池内没有合适空段：第二笔申请失败（POOL_FULL），原占用不变")
    assert tiny.describe_pool() == before
    rejected2, outcome3 = tiny.open_workspace(payload(budget=128))
    assert rejected2 is None and not outcome3.feasible
    print("规划本身无解（PLAN_INFEASIBLE）：不分配任何字节，原占用不变")
    assert tiny.describe_pool() == before
    show_pool(tiny, "失败后池状态")

    # ---------------------------------------------------------------- #
    banner("7. 布局计算期间仍可登记完成/取消（并发演示）")
    # ---------------------------------------------------------------- #
    svc.reset()
    rec, _ = svc.open_workspace(payload())
    inf = rec.inference_id
    for op in ("a1", "a2", "b1"):
        svc.submit_op(inf, op)

    def slow_apply() -> None:
        # 另一笔申请：规划计算在锁外进行，期间事件线程仍能拿到锁
        svc.open_workspace(small)

    t_apply = threading.Thread(target=slow_apply)
    t_apply.start()
    time.sleep(0.01)
    event = svc.complete_op(inf, "b1")   # 不被布局计算阻塞
    print(f"布局计算进行中完成登记成功：{event}")
    cancel_event = svc.cancel(inf)
    print(f"布局计算进行中取消登记成功：保留 {cancel_event['retained_intervals']}")
    t_apply.join()
    show_pool(svc, "并发演示结束")
    print("\n全部演练步骤通过 ✔")


if __name__ == "__main__":
    main()
