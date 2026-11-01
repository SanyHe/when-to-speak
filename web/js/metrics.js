// Shared metric formulas for the When-to-Speak player.
//
// This is the single JS source of truth for evaluation math. It mirrors
// compute_metrics() in training/src/evaluate.py exactly so that the live
// panel and the offline report can never drift apart.
//
// Usable from the browser (window.WTSMetrics) and from node (module.exports),
// which lets the build verify web/data/traces.json against the Python report.

(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.WTSMetrics = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  var ACTIONS = ["notify", "digest", "silence"];

  function r4(value) {
    return Math.round(value * 10000) / 10000;
  }

  function safeDiv(numerator, denominator) {
    return denominator ? numerator / denominator : 0;
  }

  function computeMetrics(records) {
    var total = records.length;
    var correct = 0;
    var valid = 0;
    var notifiable = 0;
    var predictedNotify = 0;
    var trueNotify = 0;
    var noisy = 0;
    var missed = 0;

    records.forEach(function (record) {
      var gold = record.gold.action;
      var pred = record.pred.action;
      if (gold === pred) correct += 1;
      if (record.json_valid) valid += 1;
      if (gold === "notify") notifiable += 1;
      if (pred === "notify") predictedNotify += 1;
      if (gold === "notify" && pred === "notify") trueNotify += 1;
      if (gold !== "notify" && pred === "notify") noisy += 1;
      if (gold === "notify" && pred !== "notify") missed += 1;
    });

    var perClass = {};
    ACTIONS.forEach(function (action) {
      var support = 0;
      var predicted = 0;
      var tp = 0;
      records.forEach(function (record) {
        if (record.gold.action === action) support += 1;
        if (record.pred.action === action) predicted += 1;
        if (record.gold.action === action && record.pred.action === action) tp += 1;
      });
      var precision = safeDiv(tp, predicted);
      var recall = safeDiv(tp, support);
      var f1 = safeDiv(2 * precision * recall, precision + recall);
      perClass[action] = {
        precision: r4(precision), recall: r4(recall), f1: r4(f1),
        support: support, predicted: predicted,
      };
    });

    var macroF1 = 0;
    ACTIONS.forEach(function (action) { macroF1 += perClass[action].f1; });
    macroF1 = macroF1 / ACTIONS.length;

    var confusion = {};
    ACTIONS.forEach(function (gold) {
      ACTIONS.forEach(function (pred) {
        var count = 0;
        records.forEach(function (record) {
          if (record.gold.action === gold && record.pred.action === pred) count += 1;
        });
        if (count) confusion[gold + "->" + pred] = count;
      });
    });

    return {
      total: total,
      json_valid_rate: r4(safeDiv(valid, total)),
      accuracy: r4(safeDiv(correct, total)),
      macro_f1: r4(macroF1),
      notify_precision: r4(safeDiv(trueNotify, predictedNotify)),
      notify_recall: r4(safeDiv(trueNotify, notifiable)),
      noise_rate: r4(safeDiv(noisy, total)),
      miss_rate: r4(safeDiv(missed, notifiable)),
      counts: {
        should_notify: notifiable,
        predicted_notify: predictedNotify,
        true_notify: trueNotify,
        noisy_notify: noisy,
        missed_notify: missed,
        invalid_json: total - valid,
      },
      per_class: perClass,
      confusion: confusion,
    };
  }

  return { computeMetrics: computeMetrics, ACTIONS: ACTIONS };
});
