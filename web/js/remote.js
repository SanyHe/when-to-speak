// Remote inference client for a vLLM OpenAI-compatible endpoint.
//
// The demo normally replays deterministic mock traces. This module adds an
// optional "live" channel: when enabled, the page sends the *same* system
// prompt used during training to a vLLM server on the cloud GPU and renders
// the returned decision next to the gold label.
//
// Exposes window.WTSRemote:
//   getConfig() / saveConfig(cfg)   read/persist settings (localStorage)
//   isEnabled()                     true when an endpoint is configured
//   decide(event, systemPrompt)     -> {decision, json_valid, raw}
//   ping()                          -> {ok, models} for the "test" button
//
// Nothing is sent anywhere until the user explicitly enables live mode, so
// the default static replay stays fully offline.
//
// Note: browsers call the vLLM server directly, so any API key typed here is
// visible to whoever can open this page. That is acceptable for a local demo;
// do not put a shared secret in a publicly hosted copy.

(function (root) {
  "use strict";

  var STORAGE_KEY = "wts.remote";
  var ACTIONS = ["notify", "digest", "silence"];
  var CHANNELS = ["im", "email", "push", "none"];
  var URGENCIES = ["urgent", "normal", "low"];

  // Used only when traces.json does not carry the prompt (older data files).
  // training/src/format_sft.py is the source of truth; evaluate.py copies it
  // into traces._meta.system_prompt.
  var FALLBACK_PROMPT =
    "You are an always-on agent that decides whether and how to notify the " +
    "user about an incoming event. Given a single event object, respond with " +
    "exactly one strict JSON object and nothing else. The JSON must contain " +
    "exactly these keys: action, channel, urgency, reason, message.\n" +
    "- action: one of notify | digest | silence\n" +
    "- channel: one of im | email | push | none (use none only for silence)\n" +
    "- urgency: one of urgent | normal | low\n" +
    "- reason: a short justification string\n" +
    "- message: the notification text (empty string when action is silence)\n" +
    "Do not add any extra keys or text outside the JSON object.";

  function defaults() {
    return { baseUrl: "", model: "wts", apiKey: "", enabled: false };
  }

  function getConfig() {
    var cfg = defaults();
    try {
      var raw = root.localStorage.getItem(STORAGE_KEY);
      if (!raw) return cfg;
      var parsed = JSON.parse(raw);
      if (typeof parsed.baseUrl === "string") cfg.baseUrl = parsed.baseUrl;
      if (typeof parsed.model === "string" && parsed.model) cfg.model = parsed.model;
      if (typeof parsed.apiKey === "string") cfg.apiKey = parsed.apiKey;
      cfg.enabled = !!parsed.enabled;
    } catch (err) {
      return defaults();
    }
    return cfg;
  }

  function saveConfig(cfg) {
    root.localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg));
  }

  function isEnabled() {
    var cfg = getConfig();
    return cfg.enabled && !!cfg.baseUrl;
  }

  function normalizeUrl(baseUrl) {
    var url = String(baseUrl || "").trim().replace(/\/+$/, "");
    if (!url) return "";
    if (!/^https?:\/\//i.test(url)) url = "https://" + url;
    // Accept the bare host:port form and append the OpenAI-compatible path.
    return url;
  }

  function headers(cfg) {
    var out = { "Content-Type": "application/json" };
    if (cfg.apiKey) out.Authorization = "Bearer " + cfg.apiKey;
    return out;
  }

  function errorText(status, body) {
    var detail = body;
    try {
      var parsed = JSON.parse(body);
      if (parsed && parsed.error && parsed.error.message) detail = parsed.error.message;
    } catch (err) { /* plain text body */ }
    return "HTTP " + status + (detail ? ": " + String(detail).slice(0, 160) : "");
  }

  function ping() {
    var cfg = getConfig();
    var base = normalizeUrl(cfg.baseUrl);
    if (!base) return Promise.reject(new Error("endpoint is empty"));
    return fetch(base + "/v1/models", { method: "GET", headers: headers(cfg) })
      .then(function (response) {
        return response.text().then(function (body) {
          if (!response.ok) throw new Error(errorText(response.status, body));
          var ids = [];
          try {
            var parsed = JSON.parse(body);
            ids = (parsed.data || []).map(function (m) { return m.id; });
          } catch (err) { /* tolerate an empty body */ }
          return { ok: true, models: ids };
        });
      });
  }

  // Strip markdown fences / thinking blocks and pull out the first JSON object.
  function extractJson(text) {
    var raw = String(text || "").trim();
    var fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
    if (fenced) raw = fenced[1].trim();
    // Qwen3 may emit a thinking block before the answer; drop it.
    raw = raw.replace(/^[\s\S]*?<\/think>/, "").trim();
    if (raw.charAt(0) !== "{") {
      var start = raw.indexOf("{");
      var end = raw.lastIndexOf("}");
      if (start >= 0 && end > start) raw = raw.slice(start, end + 1);
    }
    return raw;
  }

  function isString(value) {
    return typeof value === "string";
  }

  // Mirrors the validation rules in training/src/schema.py closely enough to
  // flag malformed model output in the UI.
  function parseDecision(text) {
    var jsonText = extractJson(text);
    var parsed;
    try {
      parsed = JSON.parse(jsonText);
    } catch (err) {
      return {
        decision: fallbackDecision("unparsable JSON"),
        json_valid: false,
        raw: String(text || ""),
      };
    }
    var ok = parsed && typeof parsed === "object" &&
      ACTIONS.indexOf(parsed.action) >= 0 &&
      CHANNELS.indexOf(parsed.channel) >= 0 &&
      URGENCIES.indexOf(parsed.urgency) >= 0 &&
      isString(parsed.reason) && isString(parsed.message);
    if (!ok) {
      return {
        decision: fallbackDecision("schema mismatch"),
        json_valid: false,
        raw: String(text || ""),
      };
    }
    return {
      decision: {
        action: parsed.action,
        channel: parsed.channel,
        urgency: parsed.urgency,
        reason: parsed.reason,
        message: parsed.message,
      },
      json_valid: true,
      raw: String(text || ""),
    };
  }

  function fallbackDecision(reason) {
    return {
      action: "notify",
      channel: "im",
      urgency: "normal",
      reason: "fallback: " + reason,
      message: "",
    };
  }

  function chat(systemPrompt, event, cfg) {
    var base = normalizeUrl(cfg.baseUrl);
    if (!base) return Promise.reject(new Error("endpoint is empty"));
    var payload = {
      model: cfg.model || "wts",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: JSON.stringify(event) },
      ],
      temperature: 0,
      max_tokens: 200,
      stream: false,
      // Keep generation in the same non-thinking format the model was
      // trained on (see format_sft.render_chat).
      chat_template_kwargs: { enable_thinking: false },
    };
    return fetch(base + "/v1/chat/completions", {
      method: "POST",
      headers: headers(cfg),
      body: JSON.stringify(payload),
    }).then(function (response) {
      return response.text().then(function (body) {
        if (!response.ok) throw new Error(errorText(response.status, body));
        var parsed = JSON.parse(body);
        var choice = (parsed.choices || [])[0] || {};
        var message = choice.message || {};
        return String(message.content || "");
      });
    });
  }

  function decide(event, systemPrompt) {
    var cfg = getConfig();
    var prompt = (typeof systemPrompt === "string" && systemPrompt.trim())
      ? systemPrompt
      : FALLBACK_PROMPT;
    return chat(prompt, event, cfg)
      .then(function (text) { return parseDecision(text); });
  }

  root.WTSRemote = {
    getConfig: getConfig,
    saveConfig: saveConfig,
    isEnabled: isEnabled,
    ping: ping,
    chat: chat,
    decide: decide,
    parseDecision: parseDecision,
    normalizeUrl: normalizeUrl,
    FALLBACK_PROMPT: FALLBACK_PROMPT,
  };
})(window);
