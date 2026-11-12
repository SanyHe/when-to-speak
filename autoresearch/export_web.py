"""Export the AutoResearch dev replay for the static web player.

Builds web/data/autoresearch.json from the frozen dev split and the per-trial
predictions, using exactly the same schema as web/data/traces.json:

    {_meta, models, events: [{event, gold, pred: {<model id>: ...}}], metrics}

The sealed split is deliberately *not* exported: it stays a one-shot holdout.
The replay compares the baseline recipe (r00) with the accepted one (r03) on
the 450-event dev split, so visitors can step through what the search bought.

Run: .venv/bin/python autoresearch/export_web.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "web", "data", "autoresearch.json")

MODELS = [
    {
        "id": "ar_base",
        "trial": "r00_baseline",
        "label": "0.6B SFT + GRPO",
        "tagline": "AutoResearch r00 · 基线配方",
        "tagline_en": "AutoResearch r00 · baseline recipe",
    },
    {
        "id": "ar_best",
        "trial": "r03_digest_focus2",
        "label": "0.6B AutoResearch Best",
        "tagline": "r03 · digest×2 + focus×2",
        "tagline_en": "r03 · digest + focus oversampling",
    },
]


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    with open(os.path.join(ROOT, "web", "data", "traces.json"), encoding="utf-8") as fh:
        prompt = json.load(fh)["_meta"]["system_prompt"]

    dev = read_jsonl(os.path.join(HERE, "data", "dev.jsonl"))
    events_by_id = {row["event_id"]: row for row in dev}

    predictions = {}
    metrics = {}
    for spec in MODELS:
        trial = spec["trial"]
        preds = read_jsonl(os.path.join(HERE, "runs", trial, "eval", "predictions.jsonl"))
        predictions[spec["id"]] = {row["event_id"]: row for row in preds}
        with open(os.path.join(HERE, "results", trial, "metrics.json"), encoding="utf-8") as fh:
            metrics[spec["id"]] = json.load(fh)["metrics"]

    events = []
    for row in dev:
        event_id = row["event_id"]
        event = {key: value for key, value in row.items() if key not in ("decision", "boundary")}
        event["boundary"] = row["boundary"]
        pred = {}
        for spec in MODELS:
            record = predictions[spec["id"]][event_id]
            assert record["gold"] == row["decision"], f"gold mismatch on {event_id}"
            pred[spec["id"]] = {
                "decision": record["pred"],
                "json_valid": record["json_valid"],
                "raw": record["raw"],
            }
        events.append({"event": event, "gold": row["decision"], "pred": pred})

    assert len(events) == len(events_by_id) == 450

    payload = {
        "_meta": {
            "mode": "autoresearch",
            "total": len(events),
            "note": (
                "AutoResearch dev replay (450 events, seed 42): r00 baseline vs the "
                "accepted r03 recipe (digest x2 + deep_work/offline x2 oversampling). "
                "Sealed holdout predictions are not published."
            ),
            "system_prompt": prompt,
        },
        "models": [dict(spec) for spec in MODELS],
        "events": events,
        "metrics": metrics,
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {OUT}: {len(events)} events, {len(MODELS)} models")
    for spec in MODELS:
        m = metrics[spec["id"]]
        print(f"  {spec['id']}: macro_f1={m['macro_f1']} noise={m['noise_rate']} miss={m['miss_rate']}")


if __name__ == "__main__":
    main()
