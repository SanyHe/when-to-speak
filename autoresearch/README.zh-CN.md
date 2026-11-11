# Auto Research — Qwen3-0.6B 的中断策略

[English](README.md) | **简体中文**

一套无人值守、单指标的研究循环，服务于本仓库的 "when to speak" 任务。模型读取一条消息事件，输出严格 JSON：

```json
{"action": "notify|digest|silence", "channel": "...", "urgency": 0.0, "reason": "...", "message": "..."}
```

循环只改变**训练配方**，始终从同一基座训练，用同一份留出 dev 集打分，并在预算耗尽或候选命中硬门禁时停止。

## 协议

| 约束 | 规则 |
|---|---|
| 可编辑面 | 只有 [`recipe.py`](recipe.py) 允许定义试验差异 |
| 冻结 oracle | `training/src/*.py`、`data/synthetic/*`、`data/templates/*`、`autoresearch/data/dev.jsonl`、`autoresearch/data/sealed_test.jsonl`、`eval_one.py`、`gate.py` |
| 完整性 | [`frozen_hashes.json`](frozen_hashes.json) 记录每个 oracle 文件的 SHA-256，不匹配则循环拒绝启动 |
| 指标 | 仅 dev `macro_f1` |
| 安全约束 | `json_valid_rate >= 0.99`、`noise_rate <= 0.02`、`miss_rate <= 0.02` |
| 基座 | `Qwen/Qwen3-0.6B`，全参微调，不使用 adapter |
| sealed 集 | 循环停止后只开一次，仅用于最终评测 |

完整研究纲领见 [`program.md`](program.md)。

### 门禁

1. `json_valid_rate >= 0.99`
2. `noise_rate <= 0.02`
3. `miss_rate <= 0.02`
4. 相对 incumbent 不倒退（noise 与 miss 允许 `+0.005` 容差）
5. 显著提升：`macro_f1 >= best + 0.005` 且 paired-bootstrap
   `P(better) >= 0.80`（2000 次重采样），或 `notify_recall >= best + 0.02`
   且 noise 不退化

第 1–4 条快速拒绝。通过第 5 条的候选会用 `seed=43` 重训一次，两种子均值仍胜过 incumbent 才接受。

### 停止条件

`STOP` 文件、墙钟、GPU 分钟、trial 预算、连续 3 次失败、`--patience` 次无新最优，
或达成 goal（`macro_f1 >= 0.97`、`noise <= 0.005`、`miss <= 0.005`、
`recall >= 0.99`、`json_valid >= 0.999`）。

## 目录结构

```
autoresearch/
  program.md        研究纲领（循环只读）
  recipe.py         唯一可编辑面：BASELINE + CANDIDATES + SEARCH_SPACE
  build_splits.py   确定性切分复现 + 泄漏检查
  data_ops.py       按配方物化 SFT / GRPO 数据
  rewards_ext.py    在冻结 reward 之上构造加权 reward
  train_grpo.py     GRPO 入口（覆写 reward 函数，复用冻结 trainer）
  eval_one.py       dev 评测 -> predictions.jsonl / metrics.json
  gate.py           5 道门禁 + paired bootstrap
  runner.py         单次 trial：物化 -> SFT -> GRPO -> eval，子进程隔离
  loop.py           选配方、过门禁、记账、写状态、停止、收尾
  launch.sh         脱离终端启动；status.sh 查看；supervise.sh 保活
  freeze.py         重新生成 frozen_hashes.json
```

每个 trial 写入 `runs/<trial_id>/`；循环每完成一个 trial 向 `ledger.jsonl`
追加一行，并原子重写 `state.json`。停止时对 incumbent 做一次 sealed 评测，
生成 `FINAL_REPORT.md` 与 `accepted/<trial_id>.json`。

## 快速开始

需要一张 CUDA GPU（≥24 GB）、系统 Python 中的 CUDA 版 PyTorch，以及约 20 GB
磁盘（两个全参 checkpoint）。

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/pip install pandas pyarrow datasets
.venv/bin/pip install --no-deps trl==1.17.1
.venv/bin/pip install "huggingface-hub>=1.5,<2.0"

export HF_ENDPOINT=https://hf-mirror.com
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B', local_dir='autoresearch/models/Qwen3-0.6B')
"

.venv/bin/python autoresearch/build_splits.py          # dev 450 + sealed 200，含泄漏检查
.venv/bin/python autoresearch/loop.py --smoke --max-trials 1
bash autoresearch/launch.sh --max-hours 9 --max-gpu-minutes 480 --max-trials 12 --patience 3
bash autoresearch/status.sh
```

`--smoke` 只跑一条极小的端到端链路（16 条 SFT / 8 条 eval），此时分数很低属正常，不写 ledger 与终报。

## 数据

`data/synthetic/` 与 `data/templates/` 是冻结的数据源。`build_splits.py`
按原始 pipeline 的方式精确复现切分（seed 42，连续 RNG 流），并对冻结的
SFT / GRPO / eval 产物做逐字节断言。

`**/data/*.jsonl` 中的 dev 与 sealed 切分**不随仓库分发**：公开 sealed 会直接
让留出集失效。请运行 `build_splits.py` 在本地重建。切分规模为
800 / 300 / 200（训练 / GRPO prompts / eval），另有 250 条 dev-extra 与 200 条 sealed。

## 结果

单卡 96 GB，一轮，7 个 trial，141 GPU 分钟，停止原因 `patience`。

| trial | 改动 | dev macro-F1 | 结果 |
|---|---|---|---|
| `r00_baseline` | — | 0.9264 | 初始 incumbent |
| `r01_sft_1ep` | SFT 1 epoch | 0.8225 | 拒：noise 4.0% |
| `r02_digest_over2` | digest ×2 | 0.8971 | 拒：miss 31.9% |
| **`r03_digest_focus2`** | digest ×2 + `{deep_work, offline}` ×2 | **0.9943** | 接受（seed 43: 0.9923，均值 0.9933） |
| `r04_contrastive_300` | r03 + 300 条 boundary 副本 | 0.9904 | 拒：提升不显著 |
| `r05_grpo_hard2` | r03 + hard-prompt ×2 | 0.9904 | 拒：提升不显著 |
| `r06_grpo_g16_beta02` | r03 + G=16, beta 0.02 | 0.9904 | 拒：提升不显著 |

`r03` 在 sealed（200 条，只开一次）上：macro-F1 **0.9828**，precision/recall 1.0，
noise 0，miss 0，JSON 合法率 0.995。

完整记录见 [`FINAL_REPORT.md`](FINAL_REPORT.md) 与 [`ledger.jsonl`](ledger.jsonl)。
账本中的绝对路径已归一化为 `./`。

每个 trial 的产物归档在 [`results/`](results/)：解析后的 SFT/GRPO 配置、SFT loss 曲线、
GRPO reward 曲线、dev 指标与数据计数，不含任何评测样本、逐条预测或 checkpoint。

## 说明

- 若 Python 版本不同，`build_splits.py` 的 `user_focus` 抽样顺序可能变化；三个冻结产物
  有逐字节断言，真正的切分漂移会直接报错。
- runner 会剥离沙箱注入的 `LD_LIBRARY_PATH`，并以干净环境启动 detached 进程；
  部分 IDE 沙箱会预加载一个库导致 CUDA 初始化失败（`error 304` / SIGSEGV）。
