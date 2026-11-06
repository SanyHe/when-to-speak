# 评测：把「懂事」变成数字

[English](en/03-eval.md) | **简体中文**

> 所有指标都可以由 `python3 training/src/evaluate.py` 一键重算；网页端用完全相同的公式实时累计。

## 1. 指标定义

以 `notify` 为正类：

| 指标 | 定义 | 直观含义 |
|---|---|---|
| Precision | 预测 notify 中真正应通知的比例 | 打扰的"准确度" |
| Recall | 应通知事件中被正确 notify 的比例 | 重要事件不漏掉 |
| 噪音率 | 不应通知却 notify 的事件数 / 总事件数 | 用户被无谓打扰的频率 |
| 漏报率 | 应通知却未 notify 的事件数 / 应通知事件数 | 真正误事的频率 |
| Macro-F1 | 三分类 F1 的算术平均 | 类别不均衡下的整体质量 |
| JSON 合法率 | 输出严格可解析且符合 schema 的比例 | 工具调用格式的可靠性 |

核心验收不是"准确率更高"，而是：

> **噪音率显著下降，且漏报率不上升。**

## 2. 真实结果

Eval 集：200 条留出事件（`data/synthetic/eval_set.jsonl`），用 `--mode hf` 在微调后的真实权重上测得
（见 `data/eval/eval_report_hf.json`）：

| Model | JSON 合法率 | Precision | Recall | 噪音率 | 漏报率 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| base | 0.00% | 0.190 | 1.000 | 81.00% | 0.00% | 0.106 | 19.00% |
| SFT | 99.50% | 0.816 | 0.816 | 3.50% | 18.42% | 0.899 | 91.50% |
| SFT + GRPO | 99.50% | 0.837 | 0.947 | 3.50% | 5.26% | 0.924 | 93.00% |

读法：

1. **SFT 先解决格式**：JSON 合法率从 0%（base 完全打不出可解析 JSON）提升到 99.5%，过了 M1 的 ≥95% 验收线。
2. **SFT 已经足够安静**：噪音率从 81.0% 降到 3.5%，但召回停在 0.816——它把"少打扰"做得过头，
   漏掉了 18.4% 本该通知的事件。
3. **GRPO 校准边界**：噪音率维持 3.5%，召回从 0.816 提到 0.947，漏报率从 18.42% 减半到 5.26%，
   macro-F1 由 0.899 升到 0.924。这正是非对称惩罚（漏报 -1.0 / 噪音 -0.5）想要的结果。

M1 的 base vs SFT 小集对比（100 条种子事件）见
[`data/eval/m1_base_vs_sft.md`](../data/eval/m1_base_vs_sft.md)。

### 历史 mock 结果（仅验证链路）

训练算力到位前，用确定性 mock 策略产出下表，只用于验证「数据 → 评测 → 网页」链路，
**不是模型结果**，已由上表取代：

| Model | JSON 合法率 | Precision | Recall | 噪音率 | 漏报率 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| Base | 85.50% | 0.330 | 0.895 | 34.50% | 10.53% | 0.633 | 63.50% |
| SFT | 98.50% | 0.714 | 0.921 | 7.00% | 7.89% | 0.898 | 91.50% |
| SFT + GRPO | 99.00% | 0.841 | 0.974 | 3.50% | 2.63% | 0.950 | 96.00% |

## 3. 错误案例归档

`evaluate.py` 会把每条非完全一致的预测写入 `data/eval/errors.jsonl`，字段包括：

- `model` / `event_id` / `error_type`
- `gold` / `predicted`（完整 decision）
- `json_valid` / `raw`（模型原始输出）

错误类型：

| error_type | 含义 | 对应用户体验 |
|---|---|---|
| `over_notify` | 不该通知却通知 | 无谓打扰 |
| `missed_notify` | 应通知却静默/汇总 | 漏掉重要事件 |
| `invalid_json` | 输出无法解析（按通知回退） | 格式不可靠 |
| `wrong_action` | digest / silence 判反 | 汇总策略错误 |
| `wrong_channel` | 动作对、渠道错 | 通知走错地方 |

这份归档就是下一轮数据与训练的直接输入：每一个出错的事件都应该变成新的 eval 用例。

## 4. 网页一致性

网页（`web/`）读取 `web/data/traces.json`，用与 `compute_metrics()` 相同的公式在前端累计统计。

- 单步回放到最后一条时，面板数值应与对应评测报告完全一致：mock 轨迹对 `data/eval/eval_report.json`，真实模型轨迹对 `data/eval/eval_report_hf.json`。
- `web/data/traces.json` 是生成物。CI 读取其 `_meta.mode`：mock 模式下重新生成并逐字段比对；真实模型（hf）模式下无法在 CPU runner 上复现，改为校验网页指标公式与 `eval_report_hf.json` 一致。

## 5. 真实模型评测协议（AutoDL）

训练完成后，在 GPU 机器上执行：

```bash
python3 training/src/evaluate.py --mode hf \
  --eval data/synthetic/eval_set.jsonl \
  --models base=Qwen/Qwen3-0.6B,\
sft=output/full_sft/final,\
sft_grpo=output/grpo/final \
  --out data/eval/eval_report_hf.json \
  --md-out data/eval/eval_report_hf.md \
  --errors-out data/eval/errors_hf.jsonl \
  --web-out web/data/traces.json
```

真实权重（200 条留出集，见 `data/eval/eval_report_hf.json`）：

| 模型 | JSON 合法率 | 精确率 | 召回率 | 噪音率 | 漏报率 | macro-F1 | 准确率 |
|---|---|---|---|---|---|---|---|
| base | 0.00% | 0.190 | 1.000 | 81.00% | 0.00% | 0.106 | 19.00% |
| SFT | 99.50% | 0.816 | 0.816 | 3.50% | 18.42% | 0.899 | 91.50% |
| SFT + GRPO | 99.50% | 0.837 | 0.947 | 3.50% | 5.26% | 0.924 | 93.00% |

`output/grpo/` 是发布模型（第二轮 GRPO：G=16、temperature 1.1、新增 urgency 奖励）；
第一轮（G=8）的日志保留在 `output/grpo_v1/` 作对照，它在同一批 200 条上与 SFT 打平。

然后把以下内容回填到 `private/feature_list.json` 的 evidence：

- `eval_report_hf.json`（三模型指标）
- `errors_hf.jsonl`（错误案例数量与类型分布）
- 训练日志与超参（`run_config.json`、`train_log.json`）

## 6. 局限

- mock 策略只验证"链路正确"，不能证明真实模型的行为；所有 mock 数字都在 UI 与文档中显式标注。
- GRPO 的增益来自第二轮调参，不是首轮：首轮 `frac_reward_zero_std` 均值 0.667
  （三分之二 prompt 的 8 个采样奖励完全相同，没有优势可用），macro-F1 与 SFT 打平；
  第二轮把 G 提到 16、temperature 提到 1.1 并新增 urgency 奖励后，该指标降到 0.573，
  macro-F1 0.924、召回 0.947、漏报 5.26%，噪音率保持 3.50%。单 seed、200 条样本，
  +0.025 的 macro-F1 仍可能含抽样波动。
- Eval 规模 200 条，单类样本量有限，指标存在抽样波动；扩充模板后可提高置信度。
- gold label 来自确定性规则，规则本身对"非直接但时效性强"等边界场景偏保守，已通过 `boundary_review.jsonl` 保留人工复核入口。
- 真实模型的 JSON 合法率可能低于 mock 假设，harness 的回退策略（解析失败 → 通知）会抬高噪音率，这正是 SFT 需要优先解决的问题。
