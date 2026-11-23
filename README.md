# CUDA 双流显存规划服务

在边缘推理部署前，对两条 CUDA 流的算子/张量做**离线显存复用规划**。
给定两条流的算子顺序、已有的跨流等待（依赖）、每个张量的生产算子与全部消费算子、
字节数、偏移对齐要求与显存预算，接口返回每个张量在同一块显存中的字节偏移；
当原依赖下任何合法时序都无法满足预算时，允许新增**一条不成环**的跨流等待后再规划，
仍然排不下则报告无解。

此外，`/runtime` 路由提供**单进程共享显存池演练**：按配置模拟一块连续显存，
受理推理工作区申请、登记算子提交/完成并支持取消，模拟“取消后已提交给 CUDA
的算子仍在读写显存”的回收语义。详见 [`docs/walkthrough.md`](docs/walkthrough.md)。

## 规划语义（核心规则）

1. 每个张量从其**生产算子开始**到**最后一个消费算子执行完成为止**都算在用（live）。
2. 在**任何符合依赖关系的时序**下，同时在用的两个张量所占字节区间不得重叠。
   实现上用偏序（happens-before）判定：张量 `a` 先于 `b` 死亡、`b` 才出生，即
   `last_consumer(a)` 可到达 `producer(b)`，二者才允许复用同一段显存；方向反过来同理。
   两个方向都不可达时，存在让二者同时在用的合法时序，必须错开。
3. 流内按提交顺序建立全序依赖；已有跨流等待是额外的有向边。整条依赖图必须是 DAG。
4. 仅当原依赖下最优布局仍超预算时，才允许增加一条跨流等待边；只考虑不成环的边。
   在所有可行方案中取占用最小的，占用相同取（等待方算子 id, 被等方算子 id）字典序最小的。
5. 每个张量的偏移必须满足自身对齐要求（对齐量必须为 2 的幂）。
6. 约束：算子总数 ≤ 12，张量总数 ≤ 8，两条流各自的算子 id 不允许重复。

## 目录

```
app/
  main.py          FastAPI 入口与请求模型
  planner/
    models.py      输入校验后的内部数据结构
    solver.py      依赖图、干扰图与最优显存布局（标准库实现）
    service.py     规划编排：原依赖优先，必要时加一条等待
  runtime/
    pool.py        单进程共享显存池：段式预留 + 在途算子引用计数
    service.py     推理工作区生命周期（申请/提交/完成/取消）
    schemas.py     /runtime 请求与响应模型
    router.py      /runtime HTTP 路由
docs/
  walkthrough.md   共享显存池单进程演练说明（语义、接口、并发）
  walkthrough.py   可直接运行的演练脚本（逐段打印池布局）
tests/             pytest 自测（含属性化随机校验）
Dockerfile
docker-compose.yml
requirements.txt
```

## 本地运行

```bash
docker compose up --build
# 打开 http://localhost:8000/docs 提交规划/运行时请求
# 显存池容量可用环境变量配置：POOL_SIZE_BYTES=4096 docker compose up
```

不用 Docker 时：

```bash
python3 -m venv --without-pip .venv && .venv/bin/python <(curl -sS https://bootstrap.pypa.io/get-pip.py)
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 自测

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

## 请求示例

`POST /plan`，请求体见 `examples/request.json`，响应字段：

- `feasible`: 是否在预算内排下
- `total_bytes`: 本次布局实际占用的连续字节数
- `budget_bytes`: 回显预算
- `offsets`: 张量 id -> 字节偏移
- `added_wait`: 新增的跨流等待，未新增为 `null`
- `note`: 说明信息
- 无解时 `feasible=false`，`offsets` 为空，并通过 `best_effort_total_bytes`
  给出（加至多一条等待后）能达到的最小占用，便于调用方判断预算缺口。

## 运行时共享显存池（/runtime）

按 `/plan` 的请求格式向 `POST /runtime/workspaces` 申请工作区：布局可行时
按 `total_bytes` 在共享显存池中分配满足全部张量对齐的连续空间，返回推理号、
规划与基址；池内没有合适空段或规划无解时申请失败（HTTP 409）且原占用不变。

凭推理号通过 `.../ops/submit`、`.../ops/complete` 登记算子提交与完成，
通过 `.../cancel` 取消整笔推理：

- 正常推理在全部算子完成后整体归还工作区；
- 取消后拒绝新提交，只保留已提交未完成算子仍在读写的字节区间，
  其余空间立即开放给新推理；
- 多个在途算子共用的字节按引用计数保留，最后一个算子完成才释放；
- 迟到/重复的提交与完成记录一律拒绝，不能释放其他推理的占用；
- 规划计算在锁外进行，期间完成/取消仍可登记，分配以当时的池占用为准。

完整接口示例与逐语义演练见 [`docs/walkthrough.md`](docs/walkthrough.md)，
也可直接运行 `python docs/walkthrough.py` 观看池布局变化。
