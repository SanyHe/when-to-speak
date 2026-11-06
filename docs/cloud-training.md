# 国内云 GPU 训练运行手册（全参 SFT + GRPO）

[English](en/cloud-training.md) | **简体中文**

> 训练**不在本地执行**，统一在**国内云 GPU** 上跑。本文档以 **AutoDL**（国内主流按小时租卡平台）为例，火山引擎 / 阿里云 PAI 等同理。

## 1. 选型

| 项 | 建议 |
|---|---|
| 平台 | AutoDL（autodl.com） |
| 显卡 | RTX 4090 / 3090（24GB）起步；GRPO 建议 A100 40GB 更稳 |
| 镜像 | PyTorch 2.x + CUDA 12.1（平台预置镜像） |
| 训练方式 | **全参微调**（无 LoRA / 无 4-bit 量化 / 无 Unsloth） |
| 推理服务 | **vLLM**（OpenAI 兼容 API），部署见 §6 |
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
pip install -r training/requirements.txt

# 4. 验证工具链（不依赖 GPU）
python3 -m pytest training/tests/ -q

# 5. 重建全量数据（固定种子，可复现）
python3 training/src/build_dataset.py
```

预期 `dataset_stats.json`：

- SFT 800 条 / GRPO prompts 300 条 / Eval 200 条
- 三类 action 均有分布
- `near_duplicate_cross_split: 0`

> **一键执行**：§3–§5 已封装为 `./training/run_all.sh`（数据集 → SFT → GRPO → 评测）。
> 超参用环境变量覆盖（如 `SFT_BATCH=4 GRPO_GENERATIONS=4 ./training/run_all.sh`）。
> 训练完需要在线服务时，按 §6 起 vLLM。下面各节是等价的逐条命令与排错说明，建议首次跑通仍按节执行、逐步确认。

## 3. 全参 SFT

```bash
python3 training/src/full_sft.py \
  --model_path Qwen/Qwen3-0.6B \
  --data data/synthetic/sft_full.jsonl \
  --output_dir output/full_sft \
  --epochs 2 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --learning_rate 2e-5 \
  --seed 42
```

`--model_path` 接受 Hugging Face repo id 或**本地目录**。国内网络建议先把权重下到实例上，
再指向本地路径，避免训练中途反复拉取：

```bash
HF_ENDPOINT=https://hf-mirror.com \
  hf download Qwen/Qwen3-0.6B --local-dir /root/models/Qwen3-0.6B

python3 training/src/full_sft.py --model_path /root/models/Qwen3-0.6B ...
```

用一键脚本时通过环境变量传入：`BASE_MODEL=/root/models/Qwen3-0.6B ./training/run_all.sh`。

产物：

- `output/full_sft/final/` — 完整 HF 权重（约 1.2GB，不入库）
- `output/full_sft/run_config.json` — 超参与种子
- `output/full_sft/train_log.json` — 损失曲线

实现要点：

- 只对 assistant（JSON decision）token 计算损失，prompt 掩码为 -100。
- `enable_thinking=False`，避免 Qwen3 生成思考块污染严格 JSON 目标。
- 全参训练用较低学习率（2e-5），prompt 掩码逻辑有本地单测（`test_full_sft.py`）。

## 4. GRPO 强化学习

```bash
python3 training/src/grpo.py \
  --model_path output/full_sft/final \
  --data data/synthetic/grpo_prompts.jsonl \
  --output_dir output/grpo \
  --epochs 1 \
  --batch_size 2 \
  --num_generations 16 \
  --temperature 1.1 \
  --beta 0.04 \
  --seed 42
```

奖励函数（`training/src/rewards.py`，确定性规则，无需奖励模型）：

| 函数 | 规则 | 分值 |
|---|---|---|
| `format_reward` | 严格 schema 通过 / 仅 JSON 可解析 / 不可解析 | +1.0 / +0.5 / 0 |
| `action_reward` | 动作正确 / 噪音 notify / 漏报 notify / digest↔silence | +1.0 / -0.5 / -1.0 / -0.15 |
| `channel_reward` | 动作与渠道都正确 | +0.25 |
| `urgency_reward` | 动作正确且 urgency 与 gold 一致 | +0.10 |

> 非对称惩罚（漏报 -1.0 > 噪音 -0.5）是产品约束：宁可多一次汇总，不可漏掉 P0。
>
> `num_generations 16` + `temperature 1.1` + `urgency_reward` 是针对上一轮 GRPO 的诊断结果：
> 8 代采样中有 66.7% 的组内奖励完全相同（无优势可用），格式与渠道奖励已接近满分，
> 因此需要更多采样、更高温度和一个更细的奖励项来制造组内差异。显存不够时先降回
> `GRPO_GENERATIONS=8`。
>
> TRL 要求 `generation_batch_size` 能被 `num_generations` 整除；脚本会按
> `batch_size × grad_accum_steps` 自动向上取整对齐（G=16 时从默认 8 提到 16），
> 无需手动改 `--grad_accum_steps`。

产物：

- `output/grpo/final/` — GRPO 后的完整 HF 权重（发布模型）
- `output/grpo/run_config.json` — 超参、种子、奖励函数名、样本数
- `output/grpo/train_log.json` — reward / loss / completion_length 曲线
- `output/grpo_v1/` — 第一轮 GRPO（G=8、temperature 1.0）的日志，保留作对照；
  它在 200 条评测上与 SFT 打平，第二轮（G=16、temperature 1.1、新增 urgency 奖励）才是发布模型

> 发布模型最初跑在 `output/grpo_v2/`，定稿后重命名为 `output/grpo/`，第一轮改为 `output/grpo_v1/`。
> 两个 `run_config.json` 里的 `output_dir` 字段仍记录运行时的原始路径，是训练事实，未回改。

实现要点与版本提示：

- 使用 TRL `GRPOTrainer`；`trl>=0.19`。不同版本 tokenizer 参数名可能是
  `tokenizer` 或 `processing_class`，脚本已按签名自动选择。
- `beta` 是相对 SFT 参考模型的 KL 系数；`beta 0` 关闭参考模型以省显存。
- GRPO 依赖 full-SFT 权重，必须在上一步完成后执行。
- 若 `GRPOConfig` 字段与当前版本不符，以
  `python -c "import trl; help(trl.GRPOConfig)"` 的实际签名为准调整，并把偏差记录到
  `private/trae-progress.md`。

**GPU 执行前的本地证据**：奖励函数与 prompt 掩码均有确定性单测
（`training/tests/test_rewards.py`、`test_full_sft.py`），`pytest` 全绿即证明打分规则与
数据契约一致；训练指标在 GPU 跑完后回填，已落在 `output/*/run_config.json` 与 `train_log.json`。

## 5. 真实模型评测

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

实测值（200 条留出集）：

| Model | JSON 合法率 | 噪音率 | 漏报率 | Macro-F1 |
|---|---|---|---|---|
| base | 0.00% | 81.00% | 0.00% | 0.106 |
| SFT | 99.50% | 3.50% | 18.42% | 0.899 |
| SFT + GRPO | 99.50% | 3.50% | 5.26% | 0.924 |

## 6. 部署推理服务（vLLM）

训练完成后，用 vLLM 起一个 OpenAI 兼容的推理服务，让本地网页直接调用真实模型。

### 6.1 安装

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vllm
# 0.6B 模型对 vLLM 版本要求不高，但 Qwen3 需要较新版本：
python3 -c "import vllm; print(vllm.__version__)"
```

### 6.2 启动服务

```bash
# 默认加载 output/grpo/final，服务名 wts，端口 6006
ALLOWED_ORIGINS="http://localhost:8765 http://127.0.0.1:8765" \
  ./training/src/serve_vllm.sh

# 也可以指定 SFT 权重做对比：
./training/src/serve_vllm.sh output/full_sft/final
```

常用环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | `6006` | AutoDL 自定义服务端口 |
| `SERVED_NAME` | `wts` | 网页里填写的模型名 |
| `API_KEY` | 空 | 设置后请求需带 `Authorization: Bearer <key>` |
| `ALLOWED_ORIGINS` | `*` | CORS 允许来源，空格分隔；本地调试可放宽，公网建议收紧 |
| `MAX_MODEL_LEN` | `2048` | 不小于训练时的 1024 |
| `GPU_UTIL` | `0.6` | 0.6B 模型占用很小，0.6 足够 |

服务起来后先在实例内自测：

```bash
curl -s http://127.0.0.1:6006/v1/models
```

### 6.3 暴露到公网（AutoDL）

1. AutoDL 控制台 → 实例 → **自定义服务**，映射端口 `6006`。
2. 得到形如 `https://<region>-6006.autodl.pro` 的地址。
3. 把这个地址（**不要带 `/v1`**）填进网页的「服务地址」。

> AutoDL 的自定义服务域名会变化（实例重启/迁移后可能换）。以控制台显示的为准。

### 6.4 本地网页接入

```bash
# 本地起静态站点（必须用 http://，file:// 的 Origin 为 null，会被 CORS 拒绝）
python3 -m http.server 8765 --directory web
```

浏览器打开 `http://localhost:8765`，在「远程推理（vLLM）」面板：

1. 填服务地址、模型名（默认 `wts`），需要时填 API Key。
2. 点 **测试连接**，应返回 `连接成功，可用模型：wts`。
3. 点 **启用**，模型切换栏会多出「vLLM Live」标签。
4. 切到该标签，点 **单步** —— 页面把当前事件实时发给 vLLM，并显示返回的 JSON 决策。

说明：

- 请求使用的是与训练完全一致的 system prompt（`evaluate.py` 把它写进
  `web/data/traces.json` 的 `_meta.system_prompt`），并显式关闭 Qwen3 thinking。
- Live 结果按事件缓存，只在切换标签/单步时请求，不会后台批量抓取。
- 「实时指标」只统计已请求的 Live 事件，未请求的记为待定。
- 浏览器直连意味着 API Key 对能打开页面的人可见；本地自用可以，公开部署请改用
  无 Key 的内网端口，或加一层后端代理。

### 6.5 排错

| 现象 | 原因 / 处理 |
|---|---|
| 测试连接报 `Failed to fetch` | CORS：`ALLOWED_ORIGINS` 没包含 `http://localhost:8765`；或服务地址带了 `/v1` |
| `HTTP 401` | 服务端设置了 `API_KEY`，网页里没填或填错 |
| `HTTP 404` | 模型名与服务端 `SERVED_NAME` 不一致 |
| 返回带 ` thinking` 前缀 | vLLM 版本过旧，不支持 `chat_template_kwargs`；升级 vLLM 或依赖前端解析兜底 |
| 显存不足 | 降低 `GPU_UTIL`，或加 `EAGER=1` 关掉 CUDA graph |

---

## 7. 回传 evidence

训练/评测完成后回传以下内容。

**提交进仓库（小、公开、可复核）**

- `output/full_sft/run_config.json`、`output/full_sft/train_log.json`
- `output/grpo/run_config.json`、`output/grpo/train_log.json`（reward 曲线是 GRPO 收敛的唯一证据）
- `data/eval/eval_report_hf.json` / `eval_report_hf.md`、`data/eval/errors_hf.jsonl`
- `web/data/traces.json`（用 `--mode hf --web-out ...` 生成的真实轨迹）

**不入库（体积大或不可公开）**

- `output/**/final/` 权重与 `output/**/checkpoints/` 优化器状态（发 GitHub Release / HuggingFace）
- 训练终端原始输出（只把最终 loss / reward 的几行贴进 `private/` 记录）

同时把 `web/data/traces.json` 里的三模型轨迹与网页统计对一遍，确认切换与统计一致。

## 8. 下载产物到本地

```bash
# 权重：走 AutoDL 控制台或 scp，不入代码仓库

# 可入库的训练配置与日志
scp -P <端口> root@<host>:~/experiment/when-to-speak/output/full_sft/run_config.json output/full_sft/
scp -P <端口> root@<host>:~/experiment/when-to-speak/output/full_sft/train_log.json  output/full_sft/
scp -P <端口> root@<host>:~/experiment/when-to-speak/output/grpo/run_config.json  output/grpo/
scp -P <端口> root@<host>:~/experiment/when-to-speak/output/grpo/train_log.json  output/grpo/
```

## 9. 常见问题

- **OOM（SFT）**：减小 `--batch_size`、增大 `--grad_accum_steps`；`max_seq_length` 默认 1024。
- **OOM（GRPO）**：先降 `--num_generations`（8 → 4），再设 `--beta 0` 关闭参考模型。
- **GRPO 生成不收敛 / reward 全 0**：检查是否开启 Qwen3 thinking（脚本已用 `enable_thinking=False`）；
  再检查 `grpo_prompts.jsonl` 的 gold 字段是否被 `remove_unused_columns` 裁掉（脚本已设 False）。
- **TRL API 不匹配**：`python -c "import trl; print(trl.__version__)"`，按实际签名为准调整并在
  `private/trae-progress.md` 记录偏差。
- **HuggingFace 下载慢**：设置 `HF_ENDPOINT=https://hf-mirror.com`，或预先下载模型到本地目录并用
  `--model_path` 指向。
