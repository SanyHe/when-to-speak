# 方法：能力 = 模型 × Harness × Eval

> 本文说明「When to Speak」如何用 Harness Engineering 的方式把后训练拆成可验收的闭环。

## 1. 为什么需要 Harness

后训练最容易失败的方式不是"模型不行"，而是流程失控：数据不可复现、训练结果无法验证、网页与评测口径不一致、上一轮结论无法交接。

因此仓库先搭 Harness，再写模型代码：

- [`AGENTS.md`](../AGENTS.md) —— 工作规则与 Definition of Done
- [`harness.yaml`](../harness.yaml) —— 集中式 guardrails
- `init.sh` —— 一键安装依赖并跑验证

> 功能清单与进度等 harness 状态文件保存在 `private/`（gitignored，不随仓库发布）。

规则很朴素：**一次只做一个 feature；没有可复核 evidence 不得标记 passing；严格按 M1 → M2 → M3。**

## 2. 数据契约先行

`training/src/schema.py` 定义 `event` / `decision` 两套 JSON Schema：

- 字段固定、`additionalProperties: false`，不允许多余键。
- 时间戳必须是带时区的 ISO-8601。
- 用 `if/then/else` 把跨字段业务规则写进 schema：`silence` 时 channel 必须为 `none`、message 必须为空。

这让"输出合法 JSON"成为可单测的硬契约，而不是训练后的期望。

## 3. 合成数据与规则专家

本期不使用真实数据。数据来自 38 个事件模板（`data/templates/`），覆盖 5 种来源 × 3 类动作：

1. **确定性规则专家**（`expert_rules.py`）生成 gold label：重复事件 / 营销 / 例行自动化 → silence；时效 + 定向 → notify；其余 → digest；deep_work 期间把非紧急 notify 降级为 digest。
2. **模板组合**枚举 `(sender, subject, snippet)` 组合，作为近重复（near-duplicate）的划分单位，保证 Eval 与训练集不共享同一组合。
3. **边界样本**导出为 `boundary_review.jsonl`，人工复核后通过 `--reviews` 回写为 gold，形成数据迭代闭环。
4. **固定随机种子**，整套数据可一键重建：`python3 training/src/build_dataset.py`。

规模（可调）：

| 数据 | 规模 | 用途 |
|---|---|---|
| SFT | 800 | 专家示范：格式 + 基础决策 |
| GRPO prompts | 300 | 强化学习：每条事件带 gold decision 供奖励函数打分 |
| Eval | 200 | 留出评测，不参与训练 |

## 4. 训练

- 基座：Qwen3-0.6B；**全参微调**（不使用 LoRA / 4-bit 量化 / Unsloth），SFT 与 RL 共用同一套参数布局。
- **SFT**：输入 event + 指令，输出 gold decision；只对 assistant 部分计算损失（prompt 掩码为 -100），先教会严格 JSON 与三档动作。
- **GRPO**：在 full-SFT checkpoint 上做强化学习。每条 prompt 采样 G 个决策，用确定性规则奖励打分：
  - 格式奖励：严格 schema 通过 +1.0，仅 JSON 可解析 +0.5，其余 0；
  - 动作奖励：动作完全正确 +1.0；不该 notify 却 notify -0.5；该 notify 却漏报 -1.0；digest↔silence 混淆 -0.15；
  - 渠道奖励：动作与渠道都正确 +0.25。
  漏报惩罚是噪音惩罚的两倍，保证"降噪不靠牺牲召回"。规则实现见 `training/src/rewards.py`，可在本地单测。
- 训练只在**国内云 GPU**（AutoDL 单卡）执行，不在本地跑；超参、种子、日志、checkpoint 全部落盘。

运行手册见 [`cloud-training.md`](cloud-training.md)。

## 5. 评测

同一批留出事件上对比 base / SFT / SFT+GRPO：

- Precision / Recall（以 notify 为正类）
- 噪音率（不应通知却 notify / 总事件数）
- 漏报率（应通知却未 notify / 应通知事件数）
- 三分类 Macro-F1、JSON 合法率、混淆矩阵
- 每条错误样本归档为 `errors.jsonl`，作为下一轮数据与训练的输入

指标定义与当前结果见 [`03-eval.md`](03-eval.md)。

## 6. 网页交付

- `web/` 是纯静态站点，读 `web/data/traces.json`（事件 + gold + 各模型决策 + 指标），逐条回放并实时累计统计。
- 三模型切换、单步 / 自动播放 / 重置、噪音率与漏报率面板、WebGPU 能力检测与降级提示。
- 真实权重经 MLC 编译为 `q4f16_1` WebGPU 产物，通过 WebLLM custom model record 接入；权重大文件放 GitHub Release / HF，仓库只存指针。
- 当前阶段用**确定性 mock 轨迹**验证全链路，所有 mock 结果都在 UI 与文档中明确标注，云端训练完成后替换。

## 7. 已知偏差与约定

Harness 的默认规则是"M1 验收通过后才进入 M2"。本项目在训练算力到位前，先完成了与 GPU 无关的部分（数据 pipeline、评测器、网页），并用 mock 决策验证链路；训练类 feature 保持 `blocked`，不伪造 `passing`。该偏差记录在 `private/claude-progress.md` 的会话记录中。
