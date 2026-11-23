"""单进程共享显存池：连续空间、段式管理。

一块固定大小的连续显存被切分成首尾相接的有序段（:class:`Segment`），每个段属于
以下三种状态之一：

1. ``owner is None``：空闲段，可用于新推理的工作区；
2. ``owner`` 非空且 ``refs == 0``：已预留给某笔推理、当前没有在途算子读写；
3. ``owner`` 非空且 ``refs > 0``：被该推理已提交未完成的算子引用，
   每多一个在途算子覆盖该字节，``refs`` 加 1。

不变量：

- 段列表始终完整覆盖 ``[0, size)`` 且按起点有序；
- 任意相邻两段的 ``(owner, refs)`` 不同（每次变更后立即合并同质相邻段）；
- ``refs`` 只对非空闲段有意义，释放后归零的段回归空闲。

所有方法都不是线程安全的；调用方（:mod:`app.runtime.service`）持锁串行化。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    # 占用该段的推理 id；None 表示空闲
    owner: str | None
    # 在途算子引用计数：该字节被多少个已提交未完成的算子覆盖
    refs: int

    @property
    def size(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "owner": self.owner,
            "refs": self.refs,
        }


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _union(intervals) -> list[tuple[int, int]]:
    """把可能重叠/相邻的区间合并为互不相交的有序并集（接受可迭代对象）。"""
    ordered = sorted(iv for iv in intervals if iv)
    if not ordered:
        return []
    merged: list[list[int]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


class MemoryPool:
    """固定容量的连续显存池。"""

    def __init__(self, size: int) -> None:
        if not isinstance(size, int) or size <= 0:
            raise ValueError("显存池容量必须为正整数")
        self.size = size
        self._segments: list[Segment] = [Segment(0, size, None, 0)]

    # ------------------------------------------------------------------ #
    # 观测
    # ------------------------------------------------------------------ #
    def snapshot(self) -> list[Segment]:
        return list(self._segments)

    def free_bytes(self) -> int:
        return sum(seg.size for seg in self._segments if seg.owner is None)

    def free_runs(self) -> list[tuple[int, int]]:
        return [
            (seg.start, seg.end) for seg in self._segments if seg.owner is None
        ]

    def largest_free_run(self) -> int:
        runs = self.free_runs()
        return max((e - s for s, e in runs), default=0)

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _merge(self) -> None:
        """合并 (owner, refs) 相同的相邻段，维持段不变量。"""
        merged: list[Segment] = []
        for seg in self._segments:
            if (
                merged
                and merged[-1].owner == seg.owner
                and merged[-1].refs == seg.refs
                and merged[-1].end == seg.start
            ):
                merged[-1] = Segment(
                    merged[-1].start, seg.end, seg.owner, seg.refs
                )
            else:
                merged.append(seg)
        self._segments = merged

    def _apply_delta(self, start: int, end: int, delta: int) -> None:
        """对与 [start, end) 相交的段调整引用计数，必要时切开边界。"""
        if start < 0 or end > self.size or start >= end:
            raise ValueError(f"非法字节区间 [{start}, {end})")
        out: list[Segment] = []
        for seg in self._segments:
            if seg.end <= start or seg.start >= end:
                out.append(seg)
                continue
            left = max(seg.start, start)
            right = min(seg.end, end)
            if seg.start < left:
                out.append(Segment(seg.start, left, seg.owner, seg.refs))
            new_refs = seg.refs + delta
            if new_refs < 0:
                raise RuntimeError("引用计数变成负数，完成与提交不配对")
            out.append(Segment(left, right, seg.owner, new_refs))
            if right < seg.end:
                out.append(Segment(right, seg.end, seg.owner, seg.refs))
        self._segments = out
        self._merge()

    # ------------------------------------------------------------------ #
    # 预留：工作区分配 / 归还
    # ------------------------------------------------------------------ #
    def allocate(
        self, size: int, alignment: int, owner: str
    ) -> int | None:
        """在首个容得下的空闲段里按 ``alignment`` 对齐分配连续空间。

        成功返回基址并把该区间记给 ``owner``（refs 为 0）；
        没有合适空段时返回 ``None`` 且池内状态不变。
        """
        if size <= 0 or alignment <= 0 or (alignment & (alignment - 1)) != 0:
            raise ValueError("分配大小必须为正，对齐必须为 2 的幂")
        for run_start, run_end in self.free_runs():
            base = _align_up(run_start, alignment)
            if base + size <= run_end:
                break
        else:
            return None

        out: list[Segment] = []
        placed = False
        for seg in self._segments:
            if placed or seg.owner is not None or seg.end <= run_start:
                out.append(seg)
                continue
            # 当前空闲段与目标空闲 run 同源（free_runs 已合并相邻空闲段）
            if seg.start < base:
                out.append(Segment(seg.start, base, None, 0))
            out.append(Segment(base, base + size, owner, 0))
            if base + size < seg.end:
                out.append(Segment(base + size, seg.end, None, 0))
            placed = True
        if not placed:
            raise RuntimeError("分配落点异常，请检查 free_runs 与段不变量")
        self._segments = out
        self._merge()
        return base

    def release_owner(self, owner: str) -> None:
        """整笔释放：owner 的全部预留（含在途区间）回归空闲。

        仅在推理正常结束（所有算子已完成，refs 必为 0）时调用。
        """
        out: list[Segment] = []
        for seg in self._segments:
            if seg.owner == owner:
                if seg.refs != 0:
                    raise RuntimeError(
                        f"推理 {owner} 仍有 {seg.refs} 个在途算子，不能整体释放"
                    )
                out.append(Segment(seg.start, seg.end, None, 0))
            else:
                out.append(seg)
        self._segments = out
        self._merge()

    def prune_owner(
        self,
        owner: str,
        keep_intervals: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """取消推理：只保留 ``keep_intervals`` 覆盖的字节，其余立即开放。

        被保留的字节必然仍被在途算子引用（refs > 0）；其余 owner 段（含
        refs 为 0 的纯预留）全部回归空闲并与相邻空闲段合并。
        返回实际保留下来的字节区间。
        """
        keep = _union(keep_intervals)
        out: list[Segment] = []
        retained: list[tuple[int, int]] = []
        for seg in self._segments:
            if seg.owner != owner:
                out.append(seg)
                continue
            cursor = seg.start
            for k_start, k_end in keep:
                overlap_s = max(seg.start, k_start)
                overlap_e = min(seg.end, k_end)
                if overlap_s >= overlap_e:
                    continue
                if cursor < overlap_s:
                    # 未被在途算子覆盖的预留：立即开放
                    out.append(Segment(cursor, overlap_s, None, 0))
                out.append(Segment(overlap_s, overlap_e, owner, seg.refs))
                retained.append((overlap_s, overlap_e))
                cursor = overlap_e
            if cursor < seg.end:
                out.append(Segment(cursor, seg.end, None, 0))
        self._segments = out
        self._merge()
        return _union(retained)

    def release_unreferenced(self, owner: str) -> None:
        """释放 owner 名下 refs 已归零的段（取消后在途算子陆续完成时用）。"""
        out: list[Segment] = []
        for seg in self._segments:
            if seg.owner == owner and seg.refs == 0:
                out.append(Segment(seg.start, seg.end, None, 0))
            else:
                out.append(seg)
        self._segments = out
        self._merge()

    # ------------------------------------------------------------------ #
    # 在途算子引用计数
    # ------------------------------------------------------------------ #
    def add_refs(self, start: int, end: int) -> None:
        """登记一个已提交算子：覆盖区间内每个字节的引用计数 +1。"""
        self._apply_delta(start, end, +1)

    def release_refs(self, start: int, end: int) -> None:
        """登记一个算子完成：覆盖区间内每个字节的引用计数 -1。"""
        self._apply_delta(start, end, -1)
