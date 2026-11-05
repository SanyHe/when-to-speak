// Verifies that the browser metric formulas and the committed web traces
// match the offline Python evaluation report exactly.
//
// Run: node training/tests/verify_web_metrics.js
// Exits non-zero on any drift (used in the Pages workflow).

"use strict";

const path = require("path");
const root = path.resolve(__dirname, "..", "..");

const metricsLib = require(path.join(root, "web", "js", "metrics.js"));
const traces = require(path.join(root, "web", "data", "traces.json"));

// The committed traces may come from deterministic mocks or from real
// checkpoints (mode "hf"); each mode has its own report to match against.
const mode = (traces._meta && traces._meta.mode) || "mock";
const reportFile = mode === "hf" ? "eval_report_hf.json" : "eval_report.json";
const report = require(path.join(root, "data", "eval", reportFile));

let failed = false;

function fail(message) {
  failed = true;
  console.error("FAIL: " + message);
}

for (const model of traces.models) {
  const records = traces.events.map(function (entry) {
    return {
      gold: entry.gold,
      pred: entry.pred[model.id].decision,
      json_valid: entry.pred[model.id].json_valid,
    };
  });

  const jsMetrics = metricsLib.computeMetrics(records);
  const pyMetrics = report.metrics[model.id];

  if (JSON.stringify(jsMetrics) !== JSON.stringify(pyMetrics)) {
    fail("JS metrics differ from eval_report.json for model " + model.id);
  }
  if (JSON.stringify(traces.metrics[model.id]) !== JSON.stringify(pyMetrics)) {
    fail("traces.json metrics are stale for model " + model.id);
  }
}

if (failed) {
  process.exit(1);
}
console.log("OK: web metrics match data/eval/" + reportFile + " for all models.");
