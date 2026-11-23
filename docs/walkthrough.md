# 单进程共享显存池演练

双流推理被取消后，**已经提交给 CUDA 的算子无法撤回**，仍可能继续读写显存。
本演练在**单个进程内**用一块按配置创建的连续“共享显存池”模拟该行为：
离线规划决定每个张量在工作区内的字节偏移，运行时服务按推理号管理工作区的
预留、在途算子引用与取消回收。

- 实现：`app/runtime/`（`pool.py` 段式显存池、`service.py` 生命周期、
  `router.py` HTTP 路由）
- 可运行脚本：[`walkthrough.py`](walkthrough.py)（不依赖 HTTP，直接演练全部语义）
- 接口试玩：服务启动后打开 `http://localhost:8000/docs`

## 1. 显存池模型

进程启动时按环境变量 `POOL_SIZE_BYTES`（默认 16 MiB）创建一块连续空间，
被切分成首尾相接的有序**段**，每段记录：

| 字段 | 含义 |
| --- | --- |
| `start` / `end` | 池内绝对字节区间 `[start, end)` |
| `owner` | 预留该段的推理号；`null` 表示空闲 |
| `refs` | 在途（已提交未完成）算子引用计数，每多一个覆盖该字节的算子加 1 |

两个正交机制：

1. **预留（owner）**：推理申请工作区时整体预留，不同推理的预留互不重叠；
2. **引用计数（refs）**：只统计在途算子实际读写的字节。一个字节只有在
   “无推理预留”且“无在途引用”时才真正空闲。相邻同质段随时合并。

## 2. HTTP 接口一览（前缀 `/runtime`）

| 方法与路径 | 作用 |
| --- | --- |
| `POST /runtime/workspaces` | 用 `/plan` 的请求体申请工作区 |
| `POST /runtime/workspaces/{id}/ops/submit` | 登记算子提交（body `{"op_id": "..."}`） |
| `POST /runtime/workspaces/{id}/ops/complete` | 登记算子完成 |
| `POST /runtime/workspaces/{id}/cancel` | 取消整笔推理 |
| `GET  /runtime/pool` | 查看显存池段布局、空闲段与空闲字节 |
| `GET  /runtime/workspaces/{id}` | 查看推理状态与每个算子的提交/完成情况 |
| `POST /runtime/reset` | 清空占用（测试/演练复位用） |

## 3. 端到端演练（HTTP）

### 3.1 申请工作区

请求体与 `POST /plan` 完全相同。布局可行时，服务按规划的 `total_bytes`
在首个容得下的空闲段里分配**连续空间**，基址按该规划中最严格的张量对齐
向上取整，因此 `base + offsets[tensor]` 满足每个张量自身的对齐要求。

```bash
curl -sS -X POST http://localhost:8000/runtime/workspaces \
  -H 'Content-Type: application/json' \
  -d '{
    "streams": {"stream_a": ["a1","a2","a3","a4"], "stream_b": ["b1","b2"]},
    "waits": [],
    "tensors": [
      {"name": "T1", "producer": "a1", "consumers": ["a2"],
       "size_bytes": 1024, "alignment": 256},
      {"name": "T2", "producer": "a2", "consumers": ["a3"],
       "size_bytes": 512, "alignment": 128},
      {"name": "T3", "producer": "b1", "consumers": ["b2"],
       "size_bytes": 512, "alignment": 512}
    ],
    "budget_bytes": 4096
  }'
```

响应（节选）：

```json
{
  "accepted": true,
  "inference_id": "inf-000001",
  "base": 0,
  "reason": null,
  "plan": {
    "feasible": true,
    "total_bytes": 1536,
    "offsets": {"T1": 0, "T2": 0, "T3": 1024}
  }
}
```

申请失败有两种，均为 HTTP 409 且**不改变任何现有占用**：

- `reason="PLAN_INFEASIBLE"`：规划本身在预算内无解（响应里带
  `best_effort_total_bytes` 说明缺口），不发生任何分配；
- `reason="POOL_FULL"`：规划可行，但池中没有足够大的合适空段
  （first-fit 扫描全部空闲段后失败），分配动作从未生效。

### 3.2 登记算子提交与完成

算子读写的字节区间 = 它生产或消费的全部张量区间的并集（绝对地址）。

```bash
curl -sS -X POST http://localhost:8000/runtime/workspaces/inf-000001/ops/submit \
  -H 'Content-Type: application/json' -d '{"op_id": "a1"}'
# {"inference_id":"inf-000001","op_id":"a1","status":"submitted",
#  "intervals":[[0,1024]], ...}
```

提交后对应字节 `refs +1`；完成后 `refs -1`。**正常推理**在规划中的
全部算子都完成时整体归还工作区（最后一次完成响应里
`"workspace_freed": true`，推理状态变为 `released`）。

### 3.3 取消整笔推理

```bash
curl -sS -X POST http://localhost:8000/runtime/workspaces/inf-000001/cancel
```

取消立即生效，规则如下：

1. 此后该推理的任何新算子提交都被拒绝（409，`INFERENCE_CANCELLED`）；
2. 已提交且未完成的算子仍在 GPU 上运行，它们读写区间的并集
   （`retained_intervals`）继续保留，引用计数照旧；
3. 工作区内其余字节（含从未提交算子的预留）**立即开放**，可被新推理申请；
4. 在途算子陆续完成时，只释放本推理名下引用归零的段；被多个在途算子
   共用的字节计数大于 1，等**最后一个完成**后才释放；
5. 没有任何在途算子时，取消即整体归还，状态直接为 `released`；
   对已取消/已归还的推理重复取消是幂等的。

### 3.4 迟到与重复事件

| 情形 | 结果 |
| --- | --- |
| 重复提交同一算子 | 409 `DUPLICATE_SUBMIT`，计数不重复增加 |
| 未提交就报完成 | 409 `OP_NOT_SUBMITTED`，池状态不变 |
| 重复报完成 | 409 `DUPLICATE_COMPLETE`，不会二次释放，更不会触碰其他推理的段 |
| 工作区归还后再提交/完成 | 409 `INFERENCE_RELEASED`（迟到事件） |
| 取消后报未提交算子的完成 | 409 `OP_NOT_SUBMITTED` |
| 推理号不存在 | 404 `NOT_FOUND`；算子不在规划中 | 422 `UNKNOWN_OP` |

所有变更只针对目标推理名下的段，因此任何被拒绝的事件都不可能释放
其他推理的占用。

### 3.5 查看池状态

```bash
curl -sS http://localhost:8000/runtime/pool
```

```json
{
  "pool_size": 4096,
  "free_bytes": 2560,
  "largest_free_run": 2560,
  "free_runs": [[1536, 4096]],
  "segments": [
    {"start": 0, "end": 1536, "owner": "inf-000001", "refs": 0},
    {"start": 1536, "end": 4096, "owner": null, "refs": 0}
  ]
}
```

## 4. 并发语义：布局期间仍可登记完成和取消

`POST /runtime/workspaces` 的规划计算是**纯函数、在进程锁外执行**；
只有真正向池里分配时才取锁。因此一笔推理在做布局计算时，其他线程仍可
即时登记算子完成或取消。进入分配临界区后看到的是**当时最新**的池占用：
若空段刚被别的推理占用导致放不下，本次申请以 `POOL_FULL` 失败且无副作用；
若取消刚释放出足够空间，则本次申请可以成功。所有提交/完成/取消/分配
在同一把可重入锁下串行化，引用计数不会错乱。

## 5. 直接运行脚本演练

脚本用与上面相同的请求结构直接驱动服务层，逐段打印池布局：

```bash
# 仓库根目录下
python docs/walkthrough.py
POOL_SIZE_BYTES=8192 python docs/walkthrough.py   # 自定义池容量
```

七个小节依次演示：对齐分配、正常生命周期、取消回收、共用字节的最后释放、
迟到/重复事件防护、失败保留原占用、布局与事件并发。

## 6. Docker Compose 启动

```bash
docker compose up --build
# 接口文档：http://localhost:8000/docs
# 健康检查：http://localhost:8000/health
```

通过环境变量配置显存池容量（字节）：

```bash
POOL_SIZE_BYTES=4096 docker compose up
```

## 7. 需求与实现对照

| 需求 | 实现位置 |
| --- | --- |
| 按配置模拟一块共享显存池 | `app/runtime/pool.py` 的 `MemoryPool`，容量取 `POOL_SIZE_BYTES` |
| 用 `/plan` 请求格式申请工作区 | `RuntimeService.open_workspace` + `app/runtime/schemas.py` |
| 可行时按 `total_bytes` 连续分配，返回推理号、规划、满足全部张量对齐的基址 | `MemoryPool.allocate`（first-fit + 对齐取整） |
| 还有合适空段就受理；失败保留原占用 | 分配失败返回 `None`，不产生任何段变更 |
| 凭推理号登记算子提交/完成、可取消整笔推理 | `submit_op` / `complete_op` / `cancel` |
| 正常推理全部算子完成后归还 | `complete_op` 中 `_all_done` 后 `release_owner` |
| 取消后拒绝新提交，只留在途算子字节，其余立即开放 | `cancel` + `MemoryPool.prune_owner` |
| 多在途算子共用字节等最后一个完成再释放 | 段级 `refs` 引用计数 + `release_unreferenced` |
| 迟到/重复完成不能释放其他推理占用 | 事件校验拒绝 + 所有变更限定在本推理 owner 段 |
| 布局期间可登记完成/取消，分配以当时占用为准 | 规划在锁外、分配在锁内，统一 `threading.RLock` 串行化 |
| docker compose 启动 | `docker-compose.yml`（支持 `POOL_SIZE_BYTES` 环境变量） |
