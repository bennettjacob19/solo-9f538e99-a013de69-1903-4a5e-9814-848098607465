"""/runtime 运行时接口测试：申请、提交/完成、取消语义与迟到/重复事件防护。"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_pool():
    client.post("/runtime/reset")
    yield
    client.post("/runtime/reset")


def _payload(budget: int = 4096) -> dict:
    # 两条流；T1/T2 在 A 流串行可复用同一 bin，T3 独立 bin。
    # 布局：bin1 偏移 0（对齐 256，大小 1024），bin2 偏移 1024（对齐 512，大小 512）
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


def _open(payload=None):
    r = client.post("/runtime/workspaces", json=payload or _payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] is True
    return body


# --------------------------------------------------------------------------- #
# 申请与布局
# --------------------------------------------------------------------------- #
def test_open_workspace_returns_id_plan_and_aligned_base():
    body = _open()
    inf = body["inference_id"]
    assert inf.startswith("inf-")
    assert body["base"] == 0
    plan = body["plan"]
    assert plan["feasible"] is True
    assert plan["total_bytes"] == 1536
    # 基址 + 每个张量偏移都满足自身对齐
    base = body["base"]
    for name, off in plan["offsets"].items():
        align = next(
            t["alignment"] for t in _payload()["tensors"] if t["name"] == name
        )
        assert (base + off) % align == 0

    pool = client.get("/runtime/pool").json()
    assert pool["segments"][0]["owner"] == inf
    assert pool["free_bytes"] == pool["pool_size"] - 1536


def test_open_infeasible_plan_keeps_occupancy():
    # 先占一笔
    first = _open()
    pool_before = client.get("/runtime/pool").json()
    # 预算小于布局最小占用 -> PLAN_INFEASIBLE，池不变
    r = client.post("/runtime/workspaces", json=_payload(budget=128))
    assert r.status_code == 409
    body = r.json()
    assert body["accepted"] is False
    assert body["reason"] == "PLAN_INFEASIBLE"
    assert body["plan"]["feasible"] is False
    assert client.get("/runtime/pool").json() == pool_before
    assert first["inference_id"] in {
        s["owner"] for s in client.get("/runtime/pool").json()["segments"]
    }


def test_open_pool_full_keeps_occupancy_and_reuses_freed_space():
    svc = app.state.runtime_service
    svc.reset()
    svc.pool.size  # noqa: B018 - 直接构造小池
    from app.runtime.pool import MemoryPool
    svc.pool = MemoryPool(2048)

    first = _open()  # 占 1536，最大空段 512
    assert client.get("/runtime/pool").json()["largest_free_run"] == 512

    before = client.get("/runtime/pool").json()
    r = client.post("/runtime/workspaces", json=_payload())
    assert r.status_code == 409
    assert r.json()["reason"] == "POOL_FULL"
    assert r.json()["plan"]["feasible"] is True
    # 申请失败，原占用保留
    assert client.get("/runtime/pool").json() == before

    # 第一笔全部完成归还后，同样的申请立刻成功
    for op in ["a1", "a2", "a3", "a4", "b1", "b2"]:
        client.post(
            f"/runtime/workspaces/{first['inference_id']}/ops/submit",
            json={"op_id": op},
        )
    for op in ["a1", "a2", "a3", "a4", "b1", "b2"]:
        client.post(
            f"/runtime/workspaces/{first['inference_id']}/ops/complete",
            json={"op_id": op},
        )
    again = _open()
    assert again["base"] == 0


def test_open_semantic_error_422():
    payload = _payload()
    payload["tensors"][0]["producer"] = "b2"  # 生产不先于消费 -> 数据竞争
    r = client.post("/runtime/workspaces", json=payload)
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# 正常生命周期
# --------------------------------------------------------------------------- #
def test_normal_lifecycle_frees_workspace_after_last_op():
    body = _open()
    inf = body["inference_id"]
    ops = ["a1", "a2", "a3", "a4", "b1", "b2"]
    for op in ops:
        r = client.post(
            f"/runtime/workspaces/{inf}/ops/submit", json={"op_id": op}
        )
        assert r.status_code == 200
        assert r.json()["status"] == "submitted"

    detail = client.get(f"/runtime/workspaces/{inf}").json()
    assert all(o["submitted"] for o in detail["ops"])

    for op in ops[:-1]:
        r = client.post(
            f"/runtime/workspaces/{inf}/ops/complete", json={"op_id": op}
        )
        assert r.json()["workspace_freed"] is False
    # 最后一个算子完成 -> 整体归还
    last = client.post(
        f"/runtime/workspaces/{inf}/ops/complete", json={"op_id": ops[-1]}
    ).json()
    assert last["workspace_freed"] is True
    assert last["inference_state"] == "released"

    pool = client.get("/runtime/pool").json()
    assert pool["free_bytes"] == pool["pool_size"]
    assert all(s["owner"] is None for s in pool["segments"])
    assert client.get(f"/runtime/workspaces/{inf}").json()["state"] == "released"


def test_op_intervals_cover_its_tensors():
    body = _open()
    inf = body["inference_id"]
    r = client.post(
        f"/runtime/workspaces/{inf}/ops/submit", json={"op_id": "a1"}
    )
    # a1 生产 T1 -> 绝对区间 [0,1024)
    assert r.json()["intervals"] == [[0, 1024]]
    r = client.post(
        f"/runtime/workspaces/{inf}/ops/submit", json={"op_id": "b1"}
    )
    # b1 生产 T3 -> bin2 绝对区间 [1024,1536)
    assert r.json()["intervals"] == [[1024, 1536]]


# --------------------------------------------------------------------------- #
# 取消语义
# --------------------------------------------------------------------------- #
def test_cancel_keeps_only_inflight_bytes_and_reopens_rest():
    body = _open()
    inf = body["inference_id"]
    # 只提交 a1（区间 [0,1024)），其余算子未提交
    client.post(f"/runtime/workspaces/{inf}/ops/submit", json={"op_id": "a1"})
    cancel = client.post(f"/runtime/workspaces/{inf}/cancel").json()
    assert cancel["status"] == "cancelled"
    assert cancel["retained_intervals"] == [[0, 1024]]
    assert cancel["freed_bytes"] == 512

    pool = client.get("/runtime/pool").json()
    retained = [s for s in pool["segments"] if s["owner"] == inf]
    assert [(s["start"], s["end"]) for s in retained] == [(0, 1024)]
    # 其余立即开放，并与尾部原有空闲段合并
    assert [1024, pool["pool_size"]] in pool["free_runs"]

    # 开放的空间立刻能被新推理申请（需要 512 的连续段）
    small = {
        "streams": {"s1": ["x1", "x2"], "s2": ["y1"]},
        "waits": [],
        "tensors": [
            {"name": "X", "producer": "x1", "consumers": ["x2"],
             "size_bytes": 512, "alignment": 256}
        ],
        "budget_bytes": 1024,
    }
    r = client.post("/runtime/workspaces", json=small)
    assert r.status_code == 200
    assert r.json()["base"] == 1024


def test_cancel_rejects_new_submits_but_completes_inflight():
    body = _open()
    inf = body["inference_id"]
    client.post(f"/runtime/workspaces/{inf}/ops/submit", json={"op_id": "a1"})
    client.post(f"/runtime/workspaces/{inf}/cancel")

    r = client.post(
        f"/runtime/workspaces/{inf}/ops/submit", json={"op_id": "a2"}
    )
    assert r.status_code == 409
    assert r.json()["code"] == "INFERENCE_CANCELLED"

    # 在途算子完成后，保留区间释放，推理进入 released
    done = client.post(
        f"/runtime/workspaces/{inf}/ops/complete", json={"op_id": "a1"}
    ).json()
    assert done["workspace_freed"] is True
    assert done["inference_state"] == "released"
    pool = client.get("/runtime/pool").json()
    assert all(s["owner"] is None for s in pool["segments"])


def test_cancel_with_no_inflight_frees_immediately_and_is_idempotent():
    body = _open()
    inf = body["inference_id"]
    cancel = client.post(f"/runtime/workspaces/{inf}/cancel").json()
    assert cancel["status"] == "released"
    assert cancel["retained_intervals"] == []
    assert cancel["freed_bytes"] == 1536

    again = client.post(f"/runtime/workspaces/{inf}/cancel").json()
    assert again["already_terminal"] is True
    assert again["status"] == "released"


def test_shared_bytes_wait_for_last_inflight_op_after_cancel():
    body = _open()
    inf = body["inference_id"]
    # a2 同时是 T1 的消费者和 T2 的生产者，区间为 [0,1024)（T1/T2 同 bin 同偏移）
    client.post(f"/runtime/workspaces/{inf}/ops/submit", json={"op_id": "a1"})
    client.post(f"/runtime/workspaces/{inf}/ops/submit", json={"op_id": "a2"})
    client.post(f"/runtime/workspaces/{inf}/cancel")

    # 两个在途算子都覆盖 [0,1024)：a1 完成后引用计数仍 > 0，字节保留
    done1 = client.post(
        f"/runtime/workspaces/{inf}/ops/complete", json={"op_id": "a1"}
    ).json()
    assert done1["workspace_freed"] is False
    pool = client.get("/runtime/pool").json()
    assert any(
        s["owner"] == inf and s["start"] == 0 and s["end"] == 1024
        for s in pool["segments"]
    )
    # 最后一个在途算子完成，空间才释放
    done2 = client.post(
        f"/runtime/workspaces/{inf}/ops/complete", json={"op_id": "a2"}
    ).json()
    assert done2["workspace_freed"] is True
    assert all(
        s["owner"] is None for s in client.get("/runtime/pool").json()["segments"]
    )


# --------------------------------------------------------------------------- #
# 迟到 / 重复 / 非法事件
# --------------------------------------------------------------------------- #
def test_duplicate_submit_and_complete_are_rejected():
    body = _open()
    inf = body["inference_id"]
    client.post(f"/runtime/workspaces/{inf}/ops/submit", json={"op_id": "a1"})
    r = client.post(
        f"/runtime/workspaces/{inf}/ops/submit", json={"op_id": "a1"}
    )
    assert r.status_code == 409 and r.json()["code"] == "DUPLICATE_SUBMIT"

    client.post(f"/runtime/workspaces/{inf}/ops/complete", json={"op_id": "a1"})
    r = client.post(
        f"/runtime/workspaces/{inf}/ops/complete", json={"op_id": "a1"}
    )
    assert r.status_code == 409 and r.json()["code"] == "DUPLICATE_COMPLETE"
    # 重复完成没有破坏池状态：工作区仍属该推理
    assert any(
        s["owner"] == inf
        for s in client.get("/runtime/pool").json()["segments"]
    )


def test_complete_without_submit_rejected_and_touches_nothing():
    body = _open()
    inf = body["inference_id"]
    before = client.get("/runtime/pool").json()
    r = client.post(
        f"/runtime/workspaces/{inf}/ops/complete", json={"op_id": "a1"}
    )
    assert r.status_code == 409 and r.json()["code"] == "OP_NOT_SUBMITTED"
    assert client.get("/runtime/pool").json() == before


def test_unknown_inference_and_op():
    r = client.post(
        "/runtime/workspaces/inf-999999/ops/submit", json={"op_id": "a1"}
    )
    assert r.status_code == 404 and r.json()["code"] == "NOT_FOUND"

    body = _open()
    inf = body["inference_id"]
    r = client.post(
        f"/runtime/workspaces/{inf}/ops/submit", json={"op_id": "ghost"}
    )
    assert r.status_code == 422 and r.json()["code"] == "UNKNOWN_OP"


def test_late_events_after_release_rejected():
    body = _open()
    inf = body["inference_id"]
    for op in ["a1", "a2", "a3", "a4", "b1", "b2"]:
        client.post(f"/runtime/workspaces/{inf}/ops/submit", json={"op_id": op})
    for op in ["a1", "a2", "a3", "a4", "b1", "b2"]:
        client.post(f"/runtime/workspaces/{inf}/ops/complete", json={"op_id": op})

    r = client.post(
        f"/runtime/workspaces/{inf}/ops/submit", json={"op_id": "a1"}
    )
    assert r.status_code == 409 and r.json()["code"] == "INFERENCE_RELEASED"
    r = client.post(
        f"/runtime/workspaces/{inf}/ops/complete", json={"op_id": "a1"}
    )
    assert r.status_code == 409 and r.json()["code"] == "INFERENCE_RELEASED"


def test_late_complete_cannot_free_another_inference_bytes():
    # A、B 两笔推理并存；A 的迟到完成不得影响 B 的占用
    a = _open()
    b = _open()
    inf_a, inf_b = a["inference_id"], b["inference_id"]
    client.post(f"/runtime/workspaces/{inf_a}/ops/submit", json={"op_id": "a1"})
    client.post(f"/runtime/workspaces/{inf_b}/ops/submit", json={"op_id": "a1"})
    client.post(f"/runtime/workspaces/{inf_a}/ops/complete", json={"op_id": "a1"})

    # A 的 a1 重复完成被拒，B 的段原封不动
    r = client.post(
        f"/runtime/workspaces/{inf_a}/ops/complete", json={"op_id": "a1"}
    )
    assert r.status_code == 409
    pool = client.get("/runtime/pool").json()
    assert any(s["owner"] == inf_b and s["refs"] == 1 for s in pool["segments"])


# --------------------------------------------------------------------------- #
# 并发：布局期间仍可登记完成/取消，所有操作串行化、计数不错乱
# --------------------------------------------------------------------------- #
def test_concurrent_events_keep_refcounts_consistent():
    body = _open()
    inf = body["inference_id"]
    ops = ["a1", "a2", "a3", "a4", "b1", "b2"]
    errors: list[Exception] = []

    def worker(op: str) -> None:
        try:
            assert client.post(
                f"/runtime/workspaces/{inf}/ops/submit", json={"op_id": op}
            ).status_code == 200
            assert client.post(
                f"/runtime/workspaces/{inf}/ops/complete", json={"op_id": op}
            ).status_code == 200
        except Exception as exc:  # pragma: no cover - 失败时留证
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(op,)) for op in ops]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    detail = client.get(f"/runtime/workspaces/{inf}").json()
    assert all(o["submitted"] and o["completed"] for o in detail["ops"])
    assert detail["state"] == "released"
    pool = client.get("/runtime/pool").json()
    assert all(s["owner"] is None and s["refs"] == 0 for s in pool["segments"])
