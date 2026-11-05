# 评测：把「懂事」变成数字

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

## 2. 当前结果（MOCK，待云端训练替换）

Eval 集：200 条留出事件（`data/synthetic/eval_set.jsonl`）。
当前模型为确定性 mock 策略，用于验证数据、评测与网页的完整链路，**不是真实训练产物**。

| Model | JSON 合法率 | Precision | Recall | 噪音率 | 漏报率 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| Base | 85.50% | 0.330 | 0.895 | 34.50% | 10.53% | 0.633 | 63.50% |
| SFT | 98.50% | 0.714 | 0.921 | 7.00% | 7.89% | 0.898 | 91.50% |
| SFT + GRPO | 99.00% | 0.841 | 0.974 | 3.50% | 2.63% | 0.950 | 96.00% |

读法：

1. **SFT 先解决格式**：JSON 合法率从 85.5% 提升到 98.5%，达到 M1 的 ≥95% 验收线。
2. **SFT 已明显更安静**：噪音率从 34.5% 降到 7.0%，且漏报率不升（10.53% → 7.89%）。
3. **GRPO 校准边界**：噪音率继续降到 3.5%，漏报率同时降到 2.63%；说明奖励函数的非对称惩罚（漏报 -1.0 / 噪音 -0.5）在降噪的同时保住了召回。

M1 的 base vs SFT 小集对比（100 条种子事件）见
[`data/eval/m1_base_vs_sft.md`](../data/eval/m1_base_vs_sft.md)。

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

然后把以下内容回填到 `private/feature_list.json` 的 evidence：

- `eval_report_hf.json`（三模型指标）
- `errors_hf.jsonl`（错误案例数量与类型分布）
- 训练日志与超参（`run_config.json`、`train_log.json`）

## 6. 局限

- mock 策略只验证"链路正确"，不能证明真实模型的行为；所有 mock 数字都在 UI 与文档中显式标注。
- GRPO 与 SFT 在 200 条评测上打平（macro-F1 0.8975 vs 0.8995，差值在抽样误差内）。训练日志显示
  `frac_reward_zero_std` 均值 0.667——三分之二的 prompt 上 8 个采样奖励完全相同，没有优势可用；
  且 format / channel 奖励已接近上限。本轮不构成 RL 增益的证据，见 `output/grpo/train_log.json`。
- Eval 规模 200 条，单类样本量有限，指标存在抽样波动；扩充模板后可提高置信度。
- gold label 来自确定性规则，规则本身对"非直接但时效性强"等边界场景偏保守，已通过 `boundary_review.jsonl` 保留人工复核入口。
- 真实模型的 JSON 合法率可能低于 mock 假设，harness 的回退策略（解析失败 → 通知）会抬高噪音率，这正是 SFT 需要优先解决的问题。
