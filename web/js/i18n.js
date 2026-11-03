// Bilingual UI strings for the When-to-Speak demo (Chinese / English).
//
// Exposes window.WTSI18n:
//   t(key, vars)        translate a key, replacing {name} placeholders
//   getLang()           current language code ("zh" | "en")
//   setLang(lang)       switch language, persist it, notify listeners
//   onChange(fn)        subscribe to language changes
//   applyDom(root)      refresh static text marked with data-i18n attributes
//
// Static markup uses data-i18n (textContent), data-i18n-html (innerHTML for
// strings that contain <code>), and data-i18n-title (title attribute).
// Dynamically rendered panels are re-rendered by app.js on every change.

(function (root) {
  "use strict";

  var DICT = {
    zh: {
      "app.title": "When to Speak — 何时该说话",
      "app.desc": "一个后训练 demo：让小模型学会在 always-on agent 的事件流中判断何时通知、何时静默。",
      "brand.sub": "post-training demo · 何时该说话",
      "badge.mock": "MOCK TRACES",
      "badge.real": "REAL TRACES",
      "badge.mockTitle": "当前展示的是确定性 mock 决策轨迹，真实 WebLLM 权重待编译",
      "lang.switch": "切换到英文",
      "lang.button": "EN",

      "hero.title": "什么时候该打断用户？",
      "hero.body": "把邮件、IM 提及、任务完成、定时检查这些事件流喂给小模型，让它输出严格 JSON 决策：立即通知 <code>notify</code>、进入汇总 <code>digest</code>、或保持静默 <code>silence</code>。切换模型，逐条回放，直观看到后训练如何降低噪音而不增加漏报。",
      "hero.switchAria": "模型切换",

      "stream.title": "事件流",
      "stream.sub": "Eval 集 · {total} 条留出事件",
      "stream.empty": "事件流待开始。点击「单步」逐条回放，或「自动播放」连续观察模型行为。",
      "stream.loadFail": "数据加载失败，页面结构仍可用，请检查 traces.json。",
      "controls.step": "单步",
      "controls.auto": "自动播放",
      "controls.pause": "暂停",
      "controls.reset": "重置",

      "decision.title": "当前决策",
      "decision.subIdle": "点击「单步」开始回放",
      "decision.subLive": "第 {n} 条 · {time}",
      "decision.empty": "事件尚未开始。按「单步」或「自动播放」，模型会为每条事件给出一个 JSON 决策。",
      "decision.goldLabel": "GOLD · 规则专家",
      "decision.modelLabel": "MODEL · {label}",
      "decision.rawLabel": "RAW MODEL OUTPUT",
      "decision.fallbackNote": "原始输出无法解析，展示的是 harness 回退决策。",
      "decision.silenceMessage": "（静默，无消息）",

      "verdict.exact": "与 gold 一致",
      "verdict.wrong_channel": "动作对，渠道错",
      "verdict.over_notify": "误打扰（过度通知）",
      "verdict.missed_notify": "漏报（应通知未通知）",
      "verdict.wrong_action": "动作判断错误",
      "verdict.invalid_json": "输出不是合法 JSON",
      "verdict.invalidSuffix": "（已按通知回退）",

      "facts.mention": "@你",
      "facts.direct": "直接发送",
      "facts.timeSensitive": "时效性强",
      "facts.duplicate": "重复事件",
      "facts.recent": "最近通知 {n} 条",
      "match.same": "与 gold 一致",
      "match.diff": "与 gold 不一致",

      "metrics.title": "实时指标",
      "metrics.subIdle": "对照 gold label · 随回放累计",
      "metrics.subLive": "对照 gold label · 已回放 {total} 条 · {model}",
      "metrics.confusion": "混淆矩阵",
      "metric.json": "JSON 合法率",
      "metric.precision": "Precision",
      "metric.recall": "Recall",
      "metric.noise": "噪音率",
      "metric.miss": "漏报率",
      "metric.macro": "Macro-F1",

      "compare.title": "三模型终局对比",
      "compare.sub": "全量 Eval 集 · mock 值，待云端训练后替换",
      "compare.noise": "噪音率",
      "compare.miss": "漏报率",
      "compare.json": "JSON 合法",
      "compare.liveCount": "已请求 {n} 条",

      "live.title": "远程推理（vLLM）",
      "live.sub": "连接 AutoDL 上的 vLLM 服务，实时对比真实模型决策",
      "live.badgeOn": "LIVE",
      "live.badgeOff": "OFFLINE",
      "live.baseLabel": "服务地址",
      "live.modelLabel": "模型名",
      "live.keyLabel": "API Key（可选）",
      "live.test": "测试连接",
      "live.enable": "启用",
      "live.disable": "停用",
      "live.hint": "未启用时页面只回放 mock 轨迹，不会发起任何网络请求。",
      "live.pending": "正在请求远程模型…",
      "live.testing": "正在测试连接…",
      "live.testOk": "连接成功，可用模型：{models}",
      "live.testFail": "连接失败：{error}",
      "live.needBase": "请先填写服务地址。",
      "live.onMsg": "已启用远程推理（模型 {model}）。切到「vLLM Live」标签并点「单步」发起请求。",

      "footer": "纯静态 · 零后端 · 零 API Key · 数据不出端。当前决策来自确定性 mock 轨迹（base / SFT / SFT+DPO 行为模型），用于验证完整链路；真实权重在 AutoDL 上训练并经 MLC 编译后接入。",

      "banner.noWebGPU": "<b>当前浏览器不支持 WebGPU。</b> 页面已自动切换到预置决策回放（mock traces），事件流、模型切换与实时统计均可正常使用。体验真实端侧推理请改用 Chrome / Edge 113+。",
      "banner.noAdapter": "<b>检测到 WebGPU 但未取得 GPU 适配器。</b> 已使用预置决策回放，功能不受影响。",
      "banner.loadFail": "<b>无法加载决策轨迹 data/traces.json：</b> {error}。请先生成数据（python3 training/src/evaluate.py --web-out web/data/traces.json）。",

      "gpu.checking": "WebGPU 检测中…",
      "gpu.ok": "WebGPU 可用",
      "gpu.noAdapter": "WebGPU 无适配器",
      "gpu.unavailable": "WebGPU 不可用",
      "gpu.error": "WebGPU 检测失败",

      "source.email": "邮件",
      "source.im": "IM",
      "source.task": "任务",
      "source.routine": "例行",
      "source.system": "系统",
      "focus.deep_work": "专注",
      "focus.available": "空闲",
      "focus.offline": "离线",
      "action.notify": "通知",
      "action.digest": "汇总",
      "action.silence": "静默"
    },

    en: {
      "app.title": "When to Speak — When should an agent speak?",
      "app.desc": "A post-training demo: teach a small model when to notify and when to stay silent in an always-on agent event stream.",
      "brand.sub": "post-training demo · when to speak",
      "badge.mock": "MOCK TRACES",
      "badge.real": "REAL TRACES",
      "badge.mockTitle": "Showing deterministic mock decision traces; real WebLLM weights are pending compilation.",
      "lang.switch": "Switch to Chinese",
      "lang.button": "中文",

      "hero.title": "When should an agent interrupt you?",
      "hero.body": "Feed an event stream — emails, IM mentions, task completions, scheduled checks — to a small model and make it emit strict JSON decisions: notify now <code>notify</code>, collect for later <code>digest</code>, or stay quiet <code>silence</code>. Switch models and replay events one by one to watch post-training cut noise without raising misses.",
      "hero.switchAria": "Model switch",

      "stream.title": "Event stream",
      "stream.sub": "Eval split · {total} held-out events",
      "stream.empty": "The stream has not started. Press “Step” to replay one event, or “Auto play” to watch continuously.",
      "stream.loadFail": "Failed to load decision traces. The page still works; check traces.json.",
      "controls.step": "Step",
      "controls.auto": "Auto play",
      "controls.pause": "Pause",
      "controls.reset": "Reset",

      "decision.title": "Current decision",
      "decision.subIdle": "Press “Step” to start the replay",
      "decision.subLive": "Event {n} · {time}",
      "decision.empty": "No event yet. Press “Step” or “Auto play” and the model emits one JSON decision per event.",
      "decision.goldLabel": "GOLD · rule expert",
      "decision.modelLabel": "MODEL · {label}",
      "decision.rawLabel": "RAW MODEL OUTPUT",
      "decision.fallbackNote": "Raw output could not be parsed; showing the harness fallback decision.",
      "decision.silenceMessage": "(silent, no message)",

      "verdict.exact": "Matches gold",
      "verdict.wrong_channel": "Right action, wrong channel",
      "verdict.over_notify": "Over-notify (false alert)",
      "verdict.missed_notify": "Missed notify",
      "verdict.wrong_action": "Wrong action",
      "verdict.invalid_json": "Output is not valid JSON",
      "verdict.invalidSuffix": " (fallback to notify applied)",

      "facts.mention": "@you",
      "facts.direct": "direct",
      "facts.timeSensitive": "time-sensitive",
      "facts.duplicate": "duplicate",
      "facts.recent": "{n} recent notifications",
      "match.same": "matches gold",
      "match.diff": "differs from gold",

      "metrics.title": "Live metrics",
      "metrics.subIdle": "Against gold labels · accumulated during replay",
      "metrics.subLive": "Against gold labels · {total} replayed · {model}",
      "metrics.confusion": "Confusion matrix",
      "metric.json": "JSON validity",
      "metric.precision": "Precision",
      "metric.recall": "Recall",
      "metric.noise": "Noise rate",
      "metric.miss": "Miss rate",
      "metric.macro": "Macro-F1",

      "compare.title": "Final comparison across models",
      "compare.sub": "Full eval split · mock values, replaced after cloud training",
      "compare.noise": "Noise rate",
      "compare.miss": "Miss rate",
      "compare.json": "Valid JSON",
      "compare.liveCount": "{n} requested",

      "live.title": "Remote inference (vLLM)",
      "live.sub": "Connect to a vLLM server on AutoDL and compare real model decisions",
      "live.badgeOn": "LIVE",
      "live.badgeOff": "OFFLINE",
      "live.baseLabel": "Endpoint",
      "live.modelLabel": "Model name",
      "live.keyLabel": "API key (optional)",
      "live.test": "Test connection",
      "live.enable": "Enable",
      "live.disable": "Disable",
      "live.hint": "While disabled the page only replays mock traces and makes no network requests.",
      "live.pending": "Requesting the remote model…",
      "live.testing": "Testing connection…",
      "live.testOk": "Connected. Models: {models}",
      "live.testFail": "Connection failed: {error}",
      "live.needBase": "Enter the endpoint first.",
      "live.onMsg": "Remote inference enabled (model {model}). Switch to the “vLLM Live” tab and press “Step” to send a request.",

      "footer": "Pure static · zero backend · zero API keys · data never leaves your device. Decisions shown here come from deterministic mock traces (base / SFT / SFT+DPO behaviour models) used to validate the full pipeline; real weights are trained on AutoDL and compiled with MLC.",

      "banner.noWebGPU": "<b>This browser does not support WebGPU.</b> The page fell back to the pre-recorded decision traces (mock), so the event stream, model switch, and live metrics keep working. Use Chrome / Edge 113+ for on-device inference.",
      "banner.noAdapter": "<b>WebGPU is present but no GPU adapter was found.</b> Pre-recorded traces are used instead; functionality is unaffected.",
      "banner.loadFail": "<b>Failed to load decision traces data/traces.json:</b> {error}. Generate them first with python3 training/src/evaluate.py --web-out web/data/traces.json.",

      "gpu.checking": "Checking WebGPU…",
      "gpu.ok": "WebGPU ready",
      "gpu.noAdapter": "WebGPU: no adapter",
      "gpu.unavailable": "WebGPU unavailable",
      "gpu.error": "WebGPU check failed",

      "source.email": "Email",
      "source.im": "IM",
      "source.task": "Task",
      "source.routine": "Routine",
      "source.system": "System",
      "focus.deep_work": "Deep work",
      "focus.available": "Available",
      "focus.offline": "Offline",
      "action.notify": "Notify",
      "action.digest": "Digest",
      "action.silence": "Silence"
    }
  };

  var STORAGE_KEY = "wts.lang";
  var listeners = [];
  var current = detect();

  function detect() {
    var param = null;
    try {
      param = new root.URLSearchParams(root.location.search).get("lang");
    } catch (e) {
      param = null;
    }
    if (param && DICT[param]) return param;
    try {
      var saved = root.localStorage.getItem(STORAGE_KEY);
      if (saved && DICT[saved]) return saved;
    } catch (e) {
      // localStorage can be unavailable in private mode; fall through.
    }
    return /^zh/i.test(root.navigator.language || "") ? "zh" : "en";
  }

  function t(key, vars) {
    var table = DICT[current] || DICT.zh;
    var text = table[key];
    if (text == null) text = DICT.zh[key];
    if (text == null) return key;
    if (!vars) return text;
    return text.replace(/\{(\w+)\}/g, function (match, name) {
      return vars[name] == null ? match : String(vars[name]);
    });
  }

  // Translate "prefix.value"; unknown values fall back to the raw value.
  function label(prefix, value) {
    var key = prefix + "." + value;
    var text = t(key);
    return text === key ? value : text;
  }

  function getLang() {
    return current;
  }

  function applyDom(rootNode) {
    var scope = rootNode || document;
    scope.querySelectorAll("[data-i18n]").forEach(function (node) {
      node.textContent = t(node.getAttribute("data-i18n"));
    });
    scope.querySelectorAll("[data-i18n-html]").forEach(function (node) {
      node.innerHTML = t(node.getAttribute("data-i18n-html"));
    });
    scope.querySelectorAll("[data-i18n-title]").forEach(function (node) {
      node.setAttribute("title", t(node.getAttribute("data-i18n-title")));
    });
    document.documentElement.lang = current === "zh" ? "zh-CN" : "en";
    document.title = t("app.title");
    var meta = document.querySelector('meta[name="description"]');
    if (meta) meta.setAttribute("content", t("app.desc"));
  }

  function setLang(lang) {
    if (!DICT[lang] || lang === current) return;
    current = lang;
    try {
      root.localStorage.setItem(STORAGE_KEY, lang);
    } catch (e) {
      // Persisting is best-effort only.
    }
    applyDom();
    listeners.forEach(function (fn) { fn(current); });
  }

  function onChange(fn) {
    listeners.push(fn);
  }

  root.WTSI18n = {
    t: t,
    label: label,
    getLang: getLang,
    setLang: setLang,
    onChange: onChange,
    applyDom: applyDom
  };
})(typeof window !== "undefined" ? window : this);
