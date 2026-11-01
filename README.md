# When to Speak

一个可复现的**后训练（post-training）最小 Demo**：训练一个小模型，在 always-on agent 的事件流中判断"**何时该说话、何时保持静默**"，并在浏览器里亲自验证效果。

- 基座：Qwen3-0.6B
- 后训练：**全参 SFT + GRPO 强化学习**（TRL，无 LoRA / 无量化）
- 浏览器推理：WebLLM + WebGPU（mlc-llm 编译）
- 交付：GitHub Pages 纯静态站点，**零后端、零 API Key、数据不出端**
- 界面：中英文双语（右上角按钮切换，或 `?lang=zh` / `?lang=en`）

## 快速开始

### 本地预览网页（无需 GPU）

```bash
# 1. 生成全量数据 + 评测轨迹（确定性 mock，无需 GPU）
python3 training/src/build_dataset.py
python3 training/src/evaluate.py \
  --eval data/synthetic/eval_set.jsonl \
  --models base,sft,sft_grpo \
  --out data/eval/eval_report.json \
  --md-out data/eval/eval_report.md \
  --errors-out data/eval/errors.jsonl \
  --web-out web/data/traces.json

# 2. 启动静态站点
python3 -m http.server 8765 --directory web
# 打开 http://localhost:8765
```

> 当前网页展示的是确定性 **mock 决策轨迹**（base / SFT / SFT+GRPO 行为模型，UI 中明确标注），
> 用于验证完整链路；真实权重在云端全参 SFT + GRPO 训练并经 MLC 编译后接入，
> 替换 `web/data/traces.json` 即可。

### 云端训练（国内 GPU，如 AutoDL）

训练不在本地执行，统一在国内云 GPU 上跑：

```bash
pip install -r training/requirements.txt
./training/run_all.sh   # 数据集 → 全参 SFT → GRPO → 真实评测 → 可选 MLC 编译
```

逐条命令、显存估算与排错见 [docs/cloud-training.md](docs/cloud-training.md)。

## 文档

- [docs/01-why-when-to-speak.md](docs/01-why-when-to-speak.md) — 选题：为什么"何时说话"值得单独训练
- [docs/02-method.md](docs/02-method.md) — 方法：数据契约、规则专家、SFT + GRPO
- [docs/03-eval.md](docs/03-eval.md) — 评测：JSON 合法率 / 噪音率 / 漏报率与验收标准
- [docs/cloud-training.md](docs/cloud-training.md) — 云 GPU 训练运行手册
- [AGENTS.md](AGENTS.md) — 本仓库的 Harness 工作方式（agent 协作规则与 Definition of Done）

## 项目状态

| 里程碑 | 状态 | 说明 |
|---|---|---|
| M1 最小闭环 | 代码完成（mock 验收通过） | 数据契约 / 种子集 / 规则专家就绪；SFT 训练待云 GPU |
| M2 数据 + 训练 + 评测 | 数据与评测链路完成 | 800/300/200 合成数据集、GRPO prompts、评测报告均已生成；训练待 GPU |
| M3 WebGPU 网页 | 网页完成 | 播放器 / 模型切换 / 实时统计 / 降级提示已实现；真实权重编译待 GPU |

## 范围边界

- 本期仅使用**合成数据**，不接入真实邮件/IM；真实数据需脱敏且仅限本地。
- 后训练链路为**全参 SFT + GRPO**；更复杂的 RL（PPO / 多轮环境）为后续阶段。

## 开发方式

本仓库按 **Harness Engineering** 实践维护：`AGENTS.md` 定义工作规则，`./init.sh` 一键安装依赖并跑验证，
一次只推进一个 feature、没有可复核 evidence 不算完成。功能清单与进度记录等内部状态保存在
`private/`（gitignored，不随仓库发布）。

```bash
./init.sh   # 安装本地依赖 + 跑验证（77 项测试）
```

## 目录结构

```
training/   数据生成、规则专家、全参 SFT、GRPO 奖励与训练、一键脚本
data/       Schema、模板、合成数据（可公开）
web/        静态站点与 model record（权重指针）
docs/       方案与方法文档
private/    内部方案与进度（.gitignore 排除，不入库）
```

## License

代码与文档以 [Apache-2.0](./LICENSE) 发布。微调产物基于 Qwen3-0.6B
（其本身为 Apache-2.0），分发权重时请保留 Qwen 的许可声明。
