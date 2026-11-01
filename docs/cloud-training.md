# 国内云 GPU 训练运行手册（全参 SFT + GRPO）

> 训练**不在本地执行**，统一在**国内云 GPU** 上跑。本文档以 **AutoDL**（国内主流按小时租卡平台）为例，火山引擎 / 阿里云 PAI 等同理。

## 1. 选型

| 项 | 建议 |
|---|---|
| 平台 | AutoDL（autodl.com） |
| 显卡 | RTX 4090 / 3090（24GB）起步；GRPO 建议 A100 40GB 更稳 |
| 镜像 | PyTorch 2.x + CUDA 12.1（平台预置镜像） |
| 训练方式 | **全参微调**（无 LoRA / 无 4-bit 量化 / 无 Unsloth） |
| 显存预算 | 0.6B 全参 SFT ≈ 11–13GB；GRPO（G=8）≈ 18–22GB |
| 预计训练量 | SFT 800 条 / 2 epoch；GRPO 300 prompts × 8 rollouts / 1 epoch |

显存构成（bf16，AdamW fp32 状态）：

| 阶段 | 权重 | 梯度 | 优化器 | 参考模型 | 激活 / rollout | 合计（估） |
|---|---|---|---|---|---|---|
| Full SFT | 1.2GB | 1.2GB | 4.8GB | – | 4–6GB | ≈ 11–13GB |
| GRPO | 1.2GB | 1.2GB | 4.8GB | 1.2GB（beta>0） | 6–10GB | ≈ 18–22GB |

> 24GB 卡可跑，但要减小 `--batch_size` / `--num_generations`；OOM 时优先
> `--num_generations 4` 或 `--beta 0`（关闭参考模型）。

## 2. 实例创建后初始化

```bash
# 1. 确认 GPU
nvidia-smi

# 2. 拉取代码（仓库公开）
cd /root
git clone <your-repo-url> when-to-speak
cd when-to-speak

# 3. 安装训练依赖（国内源）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r training/requirements.txt

# 4. 验证工具链（不依赖 GPU）
python3 -m pytest training/tests/ -q

# 5. 重建全量数据（固定种子，可复现）
python3 training/src/build_dataset.py
```

预期 `dataset_stats.json`：

- SFT 800 条 / GRPO prompts 300 条 / Eval 200 条
- 三类 action 均有分布
- `near_duplicate_cross_split: 0`

> **一键执行**：§3–§6 已封装为 `./training/run_all.sh`（数据集 → SFT → GRPO → 评测 → 可选编译）。
> 超参用环境变量覆盖（如 `SFT_BATCH=4 GRPO_GENERATIONS=4 ./training/run_all.sh`），`RUN_COMPILE=1` 触发第 6 步编译。
> 下面各节是等价的逐条命令与排错说明，建议首次跑通仍按节执行、逐步确认。

## 3. 全参 SFT（F2.3）

```bash
python3 training/src/full_sft.py \
  --data data/synthetic/sft_full.jsonl \
  --output_dir output/full_sft \
  --epochs 2 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --learning_rate 2e-5 \
  --seed 42
```

产物：

- `output/full_sft/final/` — 完整 HF 权重（约 1.2GB，不入库）
- `output/full_sft/run_config.json` — 超参与种子
- `output/full_sft/train_log.json` — 损失曲线

实现要点：

- 只对 assistant（JSON decision）token 计算损失，prompt 掩码为 -100。
- `enable_thinking=False`，避免 Qwen3 生成思考块污染严格 JSON 目标。
- 全参训练用较低学习率（2e-5），prompt 掩码逻辑有本地单测（`test_full_sft.py`）。

## 4. GRPO 强化学习（F2.4）

```bash
python3 training/src/grpo.py \
  --model_path output/full_sft/final \
  --data data/synthetic/grpo_prompts.jsonl \
  --output_dir output/grpo \
  --epochs 1 \
  --batch_size 2 \
  --num_generations 8 \
  --beta 0.04 \
  --seed 42
```

奖励函数（`training/src/rewards.py`，确定性规则，无需奖励模型）：

| 函数 | 规则 | 分值 |
|---|---|---|
| `format_reward` | 严格 schema 通过 / 仅 JSON 可解析 / 不可解析 | +1.0 / +0.5 / 0 |
| `action_reward` | 动作正确 / 噪音 notify / 漏报 notify / digest↔silence | +1.0 / -0.5 / -1.0 / -0.15 |
| `channel_reward` | 动作与渠道都正确 | +0.25 |

> 非对称惩罚（漏报 -1.0 > 噪音 -0.5）是产品约束：宁可多一次汇总，不可漏掉 P0。

产物：

- `output/grpo/final/` — GRPO 后的完整 HF 权重
- `output/grpo/run_config.json` — 超参、种子、奖励函数名、样本数
- `output/grpo/train_log.json` — reward / loss / completion_length 曲线

实现要点与版本提示：

- 使用 TRL `GRPOTrainer`；`trl>=0.19`。不同版本 tokenizer 参数名可能是
  `tokenizer` 或 `processing_class`，脚本已按签名自动选择。
- `beta` 是相对 SFT 参考模型的 KL 系数；`beta 0` 关闭参考模型以省显存。
- GRPO 依赖 full-SFT 权重，必须在上一步完成后执行。
- 若 `GRPOConfig` 字段与当前版本不符，以
  `python -c "import trl; help(trl.GRPOConfig)"` 的实际签名为准调整，并把偏差记录到
  `private/claude-progress.md`。

**GPU 执行前的本地证据**：奖励函数与 prompt 掩码均有确定性单测
（`training/tests/test_rewards.py`、`test_full_sft.py`），`pytest` 全绿即证明打分规则与
数据契约一致；训练指标本身在 GPU 跑完后才回填，当前下游 mock 数字均明确标注 MOCK。

## 5. 真实模型评测（F2.5）

```bash
python3 training/src/evaluate.py --mode hf \
  --eval data/synthetic/eval_set.jsonl \
  --models base=Qwen/Qwen3-0.6B,sft=output/full_sft/final,sft_grpo=output/grpo/final \
  --out data/eval/eval_report_hf.json \
  --md-out data/eval/eval_report_hf.md \
  --errors-out data/eval/errors_hf.jsonl \
  --web-out web/data/traces.json
```

验收方向：

- SFT JSON 合法率 ≥ 95%
- 噪音率：base → SFT → SFT+GRPO 单调下降
- 漏报率：SFT+GRPO 相对 SFT 不上升

当前 mock 参考值（仅用于链路验证，GPU 后必须替换）：

| Model | JSON 合法率 | 噪音率 | 漏报率 |
|---|---|---|---|
| base | 85.50% | 34.50% | 10.53% |
| SFT | 98.50% | 7.00% | 7.89% |
| SFT + GRPO | 99.00% | 3.50% | 2.63% |

## 6. 编译 WebGPU 产物（F3.1）

```bash
# 依赖（在 GPU 机器上）
pip install mlc-llm

# 对 SFT / SFT+GRPO 各编译一次
./training/src/compile_webgpu.sh output/full_sft/final dist/wts-sft wts-sft
./training/src/compile_webgpu.sh output/grpo/final dist/wts-grpo wts-grpo
```

产物：`dist/<name>/params/`、`mlc-chat-config.json`、`<name>-webgpu.wasm`。

发布与接入：

1. 把 `params/` 与 `wasm` 上传到 GitHub Release 或 Hugging Face 仓库（单文件 < 2GB）。
2. 填写 `web/js/model-records.js` 中对应条目的 `model` / `modelLib`，`status` 改为 `"ready"`。
3. 本地核对 WebLLM 能加载 custom record（见官方文档的 custom model record 说明）。

## 7. 回传 evidence

训练/评测完成后，把以下内容回传并更新 `private/feature_list.json`：

- 训练终端输出（最终 loss / 最终 reward）
- `run_config.json` 与 `train_log.json`（SFT 与 GRPO 各一份）
- `eval_report_hf.json` / `eval_report_hf.md`
- 错误案例数量与类型分布
- 编译产物清单（文件名与体积）

同时把 `web/data/traces.json` 换成真实模型轨迹（`--mode hf --web-out ...`），并在网页上确认三模型可切换、统计一致。

## 8. 下载产物到本地

```bash
# 在 AutoDL 控制台或通过 scp 下载 output/*/final/ 与 dist/
# 权重不入代码仓库：GitHub Release / HF 是唯一分发渠道。
```

## 9. 常见问题

- **OOM（SFT）**：减小 `--batch_size`、增大 `--grad_accum_steps`；`max_seq_length` 默认 1024。
- **OOM（GRPO）**：先降 `--num_generations`（8 → 4），再设 `--beta 0` 关闭参考模型。
- **GRPO 生成不收敛 / reward 全 0**：检查是否开启 Qwen3 thinking（脚本已用 `enable_thinking=False`）；
  再检查 `grpo_prompts.jsonl` 的 gold 字段是否被 `remove_unused_columns` 裁掉（脚本已设 False）。
- **TRL API 不匹配**：`python -c "import trl; print(trl.__version__)"`，按实际签名为准调整并在
  `private/claude-progress.md` 记录偏差。
- **HuggingFace 下载慢**：设置 `HF_ENDPOINT=https://hf-mirror.com`，或预先下载模型到本地目录并用
  `--model_path` 指向。
- **mlc-llm 子命令报错**：CLI 随版本变化，先 `mlc_llm --help` 核对，再更新 `compile_webgpu.sh`。
