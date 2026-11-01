// WebLLM custom model records for the When-to-Speak demo.
//
// These are weight *pointers* only — no weights live in the repo. The
// `pending` entries are placeholders until F3.1 compiles the SFT / SFT+GRPO
// checkpoints with mlc-llm on the cloud GPU (see docs/cloud-training.md).
// Until then the site runs on deterministic decision traces and does not
// try to load a model, so visitors never see a blank page.
//
// After compiling, replace REPLACE_WITH_* with GitHub Release / HF URLs and
// set status to "ready". Then enable the real WebLLM path in js/app.js.
window.WTS_MODEL_RECORDS = {
  base: {
    id: "base",
    label: "Base",
    status: "pending", // verify the prebuilt id against the WebLLM model list
    mlcId: "Qwen3-0.6B-q4f16_1-MLC",
      model: "",
    modelLib: "",
  },
  sft: {
    id: "sft",
    label: "SFT",
    status: "pending",
    mlcId: "when-to-speak-sft-q4f16_1-MLC",
    model: "REPLACE_WITH_RELEASE_URL/sft-q4f16_1-MLC",
    modelLib: "REPLACE_WITH_RELEASE_URL/sft-q4f16_1-MLC-webgpu.wasm",
  },
  sft_grpo: {
    id: "sft_grpo",
    label: "SFT + GRPO",
    status: "pending",
    mlcId: "when-to-speak-sft-grpo-q4f16_1-MLC",
    model: "REPLACE_WITH_RELEASE_URL/sft-grpo-q4f16_1-MLC",
    modelLib: "REPLACE_WITH_RELEASE_URL/sft-grpo-q4f16_1-MLC-webgpu.wasm",
  },
};
