"""MemoryPool 段式显存池单元测试：分配、对齐、引用计数、取消裁剪与合并。"""

from __future__ import annotations

import pytest

from app.runtime.pool import MemoryPool, Segment


def _owners(pool: MemoryPool) -> list[tuple[int, int, str | None, int]]:
    return [(s.start, s.end, s.owner, s.refs) for s in pool.snapshot()]


def test_initial_pool_is_one_free_segment():
    pool = MemoryPool(1024)
    assert pool.snapshot() == [Segment(0, 1024, None, 0)]
    assert pool.free_bytes() == 1024
    assert pool.largest_free_run() == 1024


def test_allocate_aligned_and_keeps_padding_free():
    pool = MemoryPool(1024)
    base = pool.allocate(size=100, alignment=256, owner="A")
    assert base == 0
    # 再分配一个要求 128 对齐的块：首个空闲段从 100 开始，对齐后落到 128
    base_b = pool.allocate(size=64, alignment=128, owner="B")
    assert base_b == 128
    segments = _owners(pool)
    # [0,100) A，[100,128) 空闲，[128,192) B，[192,1024) 空闲
    assert segments == [
        (0, 100, "A", 0),
        (100, 128, None, 0),
        (128, 192, "B", 0),
        (192, 1024, None, 0),
    ]
    assert pool.free_bytes() == 860


def test_allocate_uses_later_run_when_first_too_small():
    pool = MemoryPool(1000)
    assert pool.allocate(100, 1, "A") == 0
    assert pool.allocate(100, 1, "B") == 100
    pool.release_owner("A")  # [0,100) 空，[100,200) B，[200,1000) 空
    # 需要 300 连续字节：第一个空段只有 100，应落到 200
    base = pool.allocate(300, 1, "C")
    assert base == 200
    assert pool.free_runs() == [(0, 100), (500, 1000)]


def test_allocate_failure_keeps_state_unchanged():
    pool = MemoryPool(256)
    pool.allocate(200, 1, "A")  # 仅剩 56
    before = _owners(pool)
    assert pool.allocate(57, 1, "B") is None
    assert _owners(pool) == before
    # 对齐后放不下也算失败且无副作用
    assert pool.allocate(48, 64, "B") is None
    assert _owners(pool) == before


def test_refcount_shared_bytes_released_after_last_op():
    pool = MemoryPool(1024)
    pool.allocate(100, 1, "A")  # [0,100) A
    # 两个在途算子，区间重叠 [40,60)
    pool.add_refs(0, 60)       # op1
    pool.add_refs(40, 100)     # op2
    refs = {(s.start, s.end): s.refs for s in pool.snapshot() if s.owner == "A"}
    assert refs == {(0, 40): 1, (40, 60): 2, (60, 100): 1}

    pool.release_refs(0, 60)   # op1 完成：重叠区仍被 op2 引用，[0,40) 引用归零但仍预留
    refs = {
        (s.start, s.end): s.refs
        for s in pool.snapshot() if s.owner == "A"
    }
    assert refs == {(0, 40): 0, (40, 100): 1}

    pool.release_refs(40, 100)  # op2 完成：全部引用归零，但仍是 A 的预留
    assert _owners(pool) == [(0, 100, "A", 0), (100, 1024, None, 0)]
    pool.release_owner("A")
    assert pool.free_runs() == [(0, 1024)]


def test_negative_refcount_rejected():
    pool = MemoryPool(1024)
    pool.allocate(100, 1, "A")
    with pytest.raises(RuntimeError):
        pool.release_refs(0, 10)


def test_prune_keeps_only_referenced_bytes_and_frees_rest():
    pool = MemoryPool(1024)
    pool.allocate(200, 1, "A")
    pool.add_refs(50, 80)      # 唯一在途算子只覆盖 [50,80)
    retained = pool.prune_owner("A", [(50, 80)])
    assert retained == [(50, 80)]
    assert pool.snapshot() == [
        Segment(0, 50, None, 0),
        Segment(50, 80, "A", 1),
        Segment(80, 1024, None, 0),
    ]
    # 开放出来的空间立刻能给新推理用
    assert pool.allocate(50, 1, "B") == 0
    assert pool.allocate(900, 1, "C") == 80


def test_prune_union_of_multiple_intervals():
    pool = MemoryPool(1024)
    pool.allocate(200, 1, "A")
    pool.add_refs(10, 30)
    pool.add_refs(60, 90)
    retained = pool.prune_owner("A", [(10, 30), (60, 90)])
    assert retained == [(10, 30), (60, 90)]
    assert pool.free_runs() == [(0, 10), (30, 60), (90, 1024)]


def test_release_unreferenced_preserves_shared_bytes():
    pool = MemoryPool(1024)
    pool.allocate(200, 1, "A")
    pool.add_refs(0, 100)
    pool.add_refs(50, 150)
    pool.release_refs(0, 100)   # 第一个算子完成 -> [0,50) refs 归零
    pool.release_unreferenced("A")
    assert pool.snapshot() == [
        Segment(0, 50, None, 0),
        Segment(50, 150, "A", 1),
        Segment(150, 1024, None, 0),
    ]


def test_release_owner_merges_coalescing_free_runs():
    pool = MemoryPool(1000)
    pool.allocate(100, 1, "A")
    pool.allocate(100, 1, "B")
    pool.allocate(100, 1, "C")
    pool.release_owner("B")       # 中间先空
    pool.release_owner("A")       # 相邻空闲应合并为 [0,200)
    assert pool.free_runs() == [(0, 200), (300, 1000)]
    pool.release_owner("C")
    assert pool.free_runs() == [(0, 1000)]
    assert pool.free_bytes() == 1000


def test_release_owner_with_inflight_refuses():
    pool = MemoryPool(1024)
    pool.allocate(100, 1, "A")
    pool.add_refs(0, 10)
    with pytest.raises(RuntimeError):
        pool.release_owner("A")


def test_other_owner_segments_never_touched_by_prune():
    pool = MemoryPool(1024)
    pool.allocate(100, 1, "A")
    pool.allocate(100, 1, "B")
    pool.prune_owner("A", [])  # A 全量释放
    segments = _owners(pool)
    assert segments == [(0, 100, None, 0), (100, 200, "B", 0), (200, 1024, None, 0)]
