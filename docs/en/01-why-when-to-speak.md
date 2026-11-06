# Why "When to Speak"

**English** | [简体中文](../01-why-when-to-speak.md)

> The hard part of an always-on agent is not "what can it do", but **"when should it speak"**.

## 1. The problem

Picture an always-on personal assistant: it reads email, watches IM, runs background jobs, and polls on a schedule. It can produce dozens of candidate events every hour, while the user may genuinely need to be interrupted only two or three times a day.

So the question becomes:

> Given an event and its context, **should the user be interrupted right now? Through which channel? Saying what?** If there is nothing worth saying, stay silent.

This is not a "conversational ability" problem; it is an **attention allocation** problem with three traits:

- **Asymmetric cost**: missing a P0 incident is expensive, but so is interrupting twenty times a day — the latter makes users turn notifications off and abandon the product.
- **Context dependence**: the same message deserves a notification when the user is free, maybe only a digest during deep work, and always silence when it is bulk marketing.
- **Measurable**: precision/recall, noise rate and miss rate let you judge whether an agent is considerate with numbers instead of vibes.

## 2. Why it deserves a demo

Most demos are "style-cloned chatbots": swap the tone, and it is hard to see engineering depth. This topic is the opposite:

| Dimension | Why |
|---|---|
| Research relevance | It maps to one of the open problems named by Lee Robinson in Stanford CS146S (When to Speak), not an invented need. |
| Learnability | The output is a structured three-way classification, learnable by a 0.6B model with low, reproducible training cost. |
| Measurability | Gold labels and explicit metrics, not a subjective "feels good". |
| Demonstrability | base / SFT / SFT+GRPO behave visibly differently on the same event stream. |
| Storyline | Notification cost, false-interruption cost and the compute cost of idle agents connect naturally to inference FinOps. |

## 3. Task definition

- **Input**: one `event` (source, sender, subject, snippet, whether the user is mentioned, whether it is a direct message, whether it is time-sensitive, the user's current state, recent notification count, whether it is a repeat).
- **Output**: a strict JSON `decision`:

```json
{
  "action": "notify | digest | silence",
  "channel": "im | email | push | none",
  "urgency": "urgent | normal | low",
  "reason": "one-line justification",
  "message": "notification copy; empty string when silence"
}
```

The three actions:

- `notify`: interrupt now, reserved for highly time-sensitive, important, directly user-related events.
- `digest`: do not interrupt; roll into a periodic summary.
- `silence`: stay completely quiet — bulk marketing, duplicates, routine automation.

## 4. Success criteria

The goal of post-training is not "talking better", but:

> **A significant drop in the noise rate without a rise in the miss rate**, plus output that is always strictly parseable JSON.

This criterion runs through data, training, evaluation and web delivery, and is the acceptance basis for every milestone in this project.

## 5. Scope

- This stage uses **synthetic data only** and ingests no real email or IM, so it can be published with no privacy risk.
- The post-training pipeline is **full-parameter SFT + GRPO**; heavier RL (PPO, multi-turn environments) is left for later.
- The web page replays traces offline by default; when enabled it calls a vLLM service on AutoDL directly — zero backend, no local model.
