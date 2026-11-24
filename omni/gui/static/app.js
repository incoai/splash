/* splash-omni setup and demo page.
 *
 * The page's backend (gui/app.py) only sets up and runs `splash-omni serve`;
 * every speech or transcription request below is an API call to the
 * server's OpenAI-compatible endpoints, directly or through the backend's
 * transparent proxy (/proxy/...).
 */
"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

// ISO 639 codes the servers take, by language name (server/audio_frontend.py).
const ISO_CODES = {
  English: "en", Chinese: "zh", Cantonese: "yue", Arabic: "ar", German: "de",
  French: "fr", Spanish: "es", Portuguese: "pt", Indonesian: "id", Italian: "it",
  Korean: "ko", Russian: "ru", Thai: "th", Vietnamese: "vi", Japanese: "ja",
  Turkish: "tr", Hindi: "hi", Malay: "ms", Dutch: "nl", Swedish: "sv",
  Danish: "da", Finnish: "fi", Polish: "pl", Czech: "cs", Filipino: "fil",
  Persian: "fa", Greek: "el", Romanian: "ro", Hungarian: "hu", Macedonian: "mk",
};
const TTS_LANGUAGES = ["English", "Chinese", "French", "German", "Italian", "Japanese",
  "Korean", "Portuguese", "Russian", "Spanish"];
const DEFAULT_PCM_RATE = 24000;
// Silence between paragraphs, in seconds, in playback and in the WAV, and
// between the chunks a long text is spoken in (each ends a sentence).
const PARAGRAPH_GAP = 0.3;
const CHUNK_GAP = 0.15;
const SAMPLE_TEXT = "Splash Omni runs Qwen3 speech models right on your Mac. " +
  "This sentence was typed into a web page and spoken by the Mac itself, with nothing sent to the cloud.";

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */

const store = {
  get(key, fallback) {
    try {
      const value = localStorage.getItem("splash-ui." + key);
      return value === null ? fallback : JSON.parse(value);
    } catch (error) {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem("splash-ui." + key, JSON.stringify(value));
    } catch (error) {
      /* Storage is a convenience only. */
    }
  },
};

function element(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value === undefined || value === null || value === false) continue;
    if (name === "class") node.className = value;
    else if (name === "text") node.textContent = value;
    else if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
    else node.setAttribute(name, value === true ? "" : value);
  }
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function formatBytes(bytes) {
  if (bytes === null || bytes === undefined) return "?";
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + " GB";
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + " MB";
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(0) + " kB";
  return bytes + " B";
}

function formatMs(ms) {
  if (ms === null || ms === undefined || !isFinite(ms)) return "–";
  if (ms < 1000) return Math.round(ms) + " ms";
  return (ms / 1000).toFixed(ms < 10000 ? 2 : 1) + " s";
}

function formatSeconds(seconds) {
  if (seconds === null || seconds === undefined || !isFinite(seconds)) return "–";
  if (seconds < 60) return seconds.toFixed(seconds < 10 ? 2 : 1) + " s";
  const minutes = Math.floor(seconds / 60);
  return minutes + ":" + String(Math.round(seconds % 60)).padStart(2, "0") + " min";
}

function numberField(id, integer) {
  const text = $(id).value.trim();
  if (text === "") return null;
  const value = Number(text);
  if (!isFinite(value) || (integer && !Number.isInteger(value))) {
    throw new Error(`${$(id).closest("label").querySelector(".field-label").textContent} must be ${integer ? "a whole number" : "a number"}`);
  }
  return value;
}

function timestamp() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-` +
    `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
}

function autogrow(textarea) {
  const resize = () => {
    textarea.style.height = "auto";
    const limit = Math.max(window.innerHeight * 0.6, 220);
    textarea.style.height = Math.min(textarea.scrollHeight + 2, limit) + "px";
  };
  textarea.addEventListener("input", resize);
  window.addEventListener("resize", resize);
  return resize;
}

let audioContext = null;

/** The page's AudioContext, resumed when called from a user gesture
 * (Safari plays nothing from a context created or resumed elsewhere). */
function getAudioContext() {
  if (!audioContext) {
    const Context = window.AudioContext || window.webkitAudioContext;
    audioContext = new Context();
  }
  if (audioContext.state === "suspended") audioContext.resume().catch(() => {});
  return audioContext;
}

function decodeAudio(arrayBuffer) {
  const context = getAudioContext();
  return new Promise((resolve, reject) => {
    const promise = context.decodeAudioData(arrayBuffer, resolve, reject);
    if (promise && promise.then) promise.then(resolve, reject);
  });
}

/* ------------------------------------------------------------------ */
/* The page's backend: setup and lifecycle                             */

async function backend(path, body) {
  const init = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const response = await fetch(path, init);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

// The phases of a server that is starting or serving.
const RUNNING_PHASES = ["preparing", "loading", "ready"];

const app = {
  catalog: null,
  session: null,
  view: null,
  tab: null,
  voiceRunning: false,
  logSeq: 0,
  stopRequested: false,
  backendDown: false,
  downloadRates: {},
};

function showView(name) {
  if (app.view === "run" && name !== "run") haltAudio("Stopped.");
  app.view = name;
  for (const view of ["setup", "progress", "run"]) {
    $(`#view-${view}`).hidden = view !== name;
  }
  if (name === "run") location.hash = "run";
  else if (location.hash) history.replaceState(null, "", location.pathname);
  updateTopbar();
  window.scrollTo(0, 0);
}

/* ------------------------------------------------------------------ */
/* Setup                                                               */

// The speech model types, as the setup offers them: in this order, with
// what each does in words that need no knowledge of the checkpoints.
const VARIANT_TEXT = {
  custom_voice: ["Preset speakers", "Nine voices to choose from, with an optional mood or style."],
  voice_design: ["Describe a voice", "Create a voice from a few words: age, timbre, pace, emotion."],
  base: ["Natural", "The model picks a natural voice for the text by itself."],
};
const VARIANT_ORDER = ["custom_voice", "voice_design", "base"];
const TASK_NAMES = { tts: "Text to speech", asr: "Transcription", voice: "Voice chat" };

/** "Downloads about 2.2 GB" or null, for what a start would fetch. */
function downloadText(gigabytes) {
  return gigabytes > 0 ? `Downloads about ${gigabytes.toFixed(1)} GB the first time` : null;
}

const setup = {
  form: () => $("#setup-form"),
  // /api/voice/catalog: the language models and the splash command.
  voiceCatalog: null,
  task() {
    return this.form().task.value;
  },
  /** The start request of text to speech or transcription. */
  values() {
    const form = this.form();
    const task = form.task.value;
    const model = (form.querySelector('input[name="model"]:checked') || {}).value;
    const lanes = form.lanes.value.trim();
    return {
      task,
      model,
      speculative: form.speculative.checked,
      port: Number(form.port.value),
      lanes: lanes === "" ? null : Number(lanes),
      max_memory: form.max_memory.value.trim() || null,
      disable_ane: form.disable_ane.checked,
      offline: form.offline.checked,
    };
  },
  /** The start request of voice chat's three servers; throws for a port
   * that is not one. */
  voiceValues() {
    const form = this.form();
    const port = (name, label) => {
      const value = Number(form[name].value);
      if (!Number.isInteger(value) || value < 1 || value > 65535) throw new Error(`The ${label} port must be a number from 1 to 65535.`);
      return value;
    };
    const values = {
      asr_model: form.asr_model.value,
      tts_model: form.tts_model.value,
      llm_model: form.llm_model.value,
      llm_max_memory: form.llm_max_memory.value.trim(),
      asr_port: port("asr_port", "speech recognition"),
      tts_port: port("tts_port", "speech"),
      llm_port: port("llm_port", "assistant"),
      speculative: form.voice_speculative.checked,
    };
    if (form.splash.value.trim()) values.splash = form.splash.value.trim();
    return values;
  },
  restore() {
    const saved = store.get("setup", {});
    const form = this.form();
    const task = location.hash === "#voice" ? "voice" : saved.task;
    if (task) form.task.value = task;
    this.models = saved.models || {};
    this.variant = saved.variant || "custom_voice";
    if (saved.speculative === false) form.speculative.checked = false;
    // A port typed under Advanced settings is kept; otherwise the page
    // finds a free one (checkPort).
    this.portChosen = Boolean(saved.port_chosen);
    if (saved.port && this.portChosen) form.port.value = saved.port;
    if (saved.lanes) form.lanes.value = saved.lanes;
    if (saved.max_memory) form.max_memory.value = saved.max_memory;
    form.disable_ane.checked = !!saved.disable_ane;
    form.offline.checked = !!saved.offline;
  },
  save() {
    const form = this.form();
    const task = form.task.value;
    if (task === "voice") {
      const saved = store.get("setup", {});
      store.set("setup", { ...saved, task });
      try {
        store.set("voice-setup", this.voiceValues());
      } catch (error) {
        /* An unfinished port: saved once it is one. */
      }
      return;
    }
    const values = this.values();
    this.models = this.models || {};
    if (values.model) this.models[values.task] = values.model;
    const model = app.catalog && app.catalog.models.find((entry) => entry.id === values.model);
    if (model && model.variant) this.variant = model.variant;
    store.set("setup", { ...values, port_chosen: Boolean(this.portChosen), models: this.models, variant: this.variant });
  },
  /** The task's sections: its few choices, and its advanced settings. */
  renderSections() {
    const task = this.task();
    for (const card of $$(".task-card")) card.classList.toggle("selected", card.dataset.task === task);
    $("#variant-section").hidden = task !== "tts" || !(app.catalog.variants || []).length;
    $("#asr-section").hidden = task !== "asr";
    $("#voice-section").hidden = task !== "voice";
    $("#advanced-speech").hidden = task === "voice";
    $("#advanced-voice").hidden = task !== "voice";
    $("#speculation-row").hidden = task !== "asr";
    $("#port-warning").hidden = true;
  },
  /** The speech model types, then the precisions of the chosen type (or of
   * the transcription model) under Advanced settings. */
  renderModels() {
    const form = this.form();
    const task = form.task.value;
    if (task === "voice") return;
    const variants = VARIANT_ORDER
      .map((id) => (app.catalog.variants || []).find((variant) => variant.id === id))
      .filter(Boolean);
    const speech = task === "tts" && variants.length > 0;
    const chosen = this.models && this.models[task];
    const chosenEntry = app.catalog.models.find((entry) => entry.id === chosen);
    if (speech) {
      if (!variants.some((variant) => variant.id === this.variant)) this.variant = variants[0].id;
      $("#variant-choices").replaceChildren(...variants.map((variant) => {
        const [name, summary] = VARIANT_TEXT[variant.id] || [variant.name, variant.summary];
        return element("label", { class: "choice" },
          element("input", { type: "radio", name: "variant", value: variant.id, checked: variant.id === this.variant }),
          element("span", { class: "choice-body" },
            element("strong", {}, name),
            element("small", { text: summary })));
      }));
    }
    const entries = app.catalog.models.filter((model) =>
      model.task === task && (!speech || model.variant === this.variant));
    // Keep the size and precision across model types (VoiceDesign has no
    // 0.6B: the precision then).
    const precision = chosenEntry ? chosenEntry.precision : "4-bit";
    const size = chosenEntry ? chosenEntry.size : "1.7B";
    const selected = entries.find((model) => model.id === chosen) ||
      entries.find((model) => model.size === size && model.precision === precision) ||
      entries.find((model) => model.precision === precision) || entries[0];
    $("#model-choices").replaceChildren(...entries.map((model) =>
      element("label", { title: model.id },
        element("input", { type: "radio", name: "model", value: model.id, checked: model === selected }),
        element("span", {}, `${model.size} ${model.precision} · ${model.gigabytes} GB`))));
    const draft = selected && selected.draft;
    if (draft) $("#draft-badge").textContent = draft.cached ? "The draft is on this Mac." : `The draft downloads about ${draft.gigabytes} GB once.`;
    const lanes = app.catalog.max_lanes[task];
    form.lanes.max = lanes;
    $("#lanes-hint").textContent = `Requests decoded together; at most ${lanes}.`;
  },
  /** Voice chat's choices: the language models Splash serves, and the
   * speech checkpoints for each server. */
  renderVoice() {
    const catalog = this.voiceCatalog;
    if (!catalog) return;
    const form = this.form();
    const saved = store.get("voice-setup", {});
    const option = (value, label, chosen) => element("option", { value, selected: value === chosen }, label);
    const here = (entry) => (entry.cached || entry.installed ? "" : ` · ${entry.gigabytes} GB download`);
    const speech = catalog.speech.models;
    const keep = (select, build) => {
      const current = select.value;
      build();
      if (current && $$("option", select).some((item) => item.value === current)) select.value = current;
    };
    keep(form.asr_model, () => form.asr_model.replaceChildren(...speech.filter((entry) => entry.task === "asr")
      .map((entry) => option(entry.id, entry.label + here(entry), saved.asr_model || catalog.defaults.asr))));
    keep(form.tts_model, () => form.tts_model.replaceChildren(...speech.filter((entry) => entry.task === "tts")
      .map((entry) => option(entry.id, entry.label + here(entry), saved.tts_model || catalog.defaults.tts))));
    keep(form.llm_model, () => form.llm_model.replaceChildren(...catalog.llm_models.map((entry) =>
      option(entry.id, SplashLogic.languageModelName(entry.id) + (entry.cached ? "" : " · download"), saved.llm_model || catalog.defaults.llm))));
    if (!form.dataset.voiceFilled) {
      // Fill the rest once: later renders keep what was typed.
      form.dataset.voiceFilled = "1";
      form.splash.value = saved.splash || "";
      form.llm_max_memory.value = saved.llm_max_memory ?? this.voiceLlmMemory();
      form.dataset.llmMemoryAuto = this.voiceLlmMemory();
      for (const role of ["asr", "tts", "llm"]) form[`${role}_port`].value = saved[`${role}_port`] || catalog.defaults.ports[role];
      form.voice_speculative.checked = saved.speculative !== false;
    }
    form.splash.placeholder = catalog.splash || "path to splash";
    $("#splash-hint").textContent = catalog.splash ? `Found: ${catalog.splash}` : "Not found on this Mac.";
  },
  /** The language model's --max-memory beside the chosen speech servers,
   * as the server computes it (voice.default_llm_memory); "" for Splash's
   * own budget. */
  voiceLlmMemory() {
    const form = this.form();
    const memory = this.voiceCatalog && this.voiceCatalog.memory;
    if (!memory || memory.budget_gib == null) return "";
    const speech = (id, draft) => {
      const entry = memory.speech_gib[id];
      return entry ? entry[draft ? "draft" : "plain"] : null;
    };
    const asr = speech(form.asr_model.value, form.voice_speculative.checked);
    const tts = speech(form.tts_model.value, false);
    if (asr == null || tts == null) return "";
    const gib = Math.floor(memory.budget_gib - asr - tts);
    return gib >= memory.minimum_llm_gib ? `${gib}G` : "";
  },
  /** Follow the speech choices with the language model's memory, unless it
   * was typed: a value other than the one last filled in. */
  followVoiceMemory() {
    const form = this.form();
    if (!this.voiceCatalog || form.llm_max_memory.value.trim() !== (form.dataset.llmMemoryAuto ?? "")) return;
    form.llm_max_memory.value = form.dataset.llmMemoryAuto = this.voiceLlmMemory();
  },
  renderSummary() {
    const task = this.task();
    const summary = $("#download-summary");
    const start = $("#btn-start");
    if (task === "voice") {
      const catalog = this.voiceCatalog;
      if (!catalog) {
        summary.replaceChildren(element("span", { class: "muted", text: "Looking for Splash…" }));
        start.textContent = "Start voice chat";
        return;
      }
      const values = { asr: this.form().asr_model.value, tts: this.form().tts_model.value };
      const speech = catalog.speech.models;
      let gigabytes = 0;
      for (const id of Object.values(values)) {
        const entry = speech.find((model) => model.id === id);
        if (entry && !entry.cached && !entry.installed) gigabytes += entry.gigabytes;
      }
      const asr = speech.find((model) => model.id === values.asr);
      if (this.form().voice_speculative.checked && asr && asr.draft && !asr.draft.cached) gigabytes += asr.draft.gigabytes;
      const llm = catalog.llm_models.find((entry) => entry.id === this.form().llm_model.value);
      const parts = [];
      if (gigabytes > 0) parts.push(`speech models: about ${gigabytes.toFixed(1)} GB`);
      if (llm && !llm.cached) parts.push("the assistant's model");
      summary.replaceChildren(
        element("strong", { text: parts.length ? "Downloads " + parts.join(" and ") + " the first time" : "Everything is on this Mac" }),
        element("small", { text: "Speech recognition, the assistant and speech start together." }));
      start.textContent = "Start voice chat";
      return;
    }
    const values = this.values();
    const model = app.catalog.models.find((entry) => entry.id === values.model);
    const draft = model && model.draft;
    const total = (model && !model.cached && !model.installed ? model.gigabytes : 0) +
      (task === "asr" && values.speculative && draft && !draft.cached ? draft.gigabytes : 0);
    summary.replaceChildren(
      element("strong", { text: downloadText(total) || "Ready on this Mac" }),
      element("small", { text: model ? model.label : "" }));
    $("#command-preview").textContent = commandLine(values);
    start.textContent = total > 0 ? "Download and start" : "Start";
  },
  /** Whether voice chat lacks Splash: none found, and no path given. */
  splashMissing() {
    return Boolean(this.voiceCatalog) && !this.voiceCatalog.splash && !this.form().splash.value.trim();
  },
  /** Voice chat without Splash: where to find it, in place of the
   * assistant's choice and the start button. */
  renderSplash() {
    const missing = this.task() === "voice" && this.splashMissing();
    $("#voice-ready").hidden = missing;
    $("#splash-install").hidden = !missing;
    $("#start-row").hidden = missing;
  },
  render() {
    this.renderSections();
    this.renderSplash();
    this.renderModels();
    this.renderVoice();
    this.renderSummary();
    this.renderBusy();
    if (this.task() !== "voice") this.checkPort();
  },
  /** Warn before a start that the port is taken (the launcher would refuse
   * it), unless the page's own server holds it. */
  checkPort() {
    clearTimeout(this.portTimer);
    this.portTimer = setTimeout(async () => {
      const port = Number(this.form().port.value);
      const warning = $("#port-warning");
      const session = app.session;
      const ours = session && session.selection && session.selection.port === port &&
        ["preparing", "loading", "ready", "stopping"].includes(session.phase);
      if (this.task() === "voice" || !Number.isInteger(port) || port < 1 || port > 65535 || ours) {
        warning.hidden = true;
        return;
      }
      try {
        const status = await backend(`/api/port?port=${port}`);
        if (!status.free && !this.portChosen) {
          // Nobody chose this port: take the next free one instead of
          // asking.
          for (let next = port + 1; next <= Math.min(port + 20, 65535); next++) {
            if ((await backend(`/api/port?port=${next}`)).free) {
              this.form().port.value = next;
              warning.hidden = true;
              this.renderSummary();
              return;
            }
          }
        }
        warning.hidden = status.free || this.task() === "voice";
        warning.textContent = status.free ? "" :
          `Port ${port} is in use (${status.reason}). Choose another port under Advanced settings, or use the server already running there.`;
      } catch (error) {
        warning.hidden = true;
      }
    }, 250);
  },
  renderBusy() {
    const phase = app.session && app.session.phase;
    const busy = phase === "stopping" || phase === "preparing" || phase === "loading" || phase === "ready";
    const voice = this.task() === "voice";
    const noSplash = voice && this.voiceCatalog && !this.voiceCatalog.splash && !this.form().splash.value.trim();
    $("#btn-start").disabled = busy || app.voiceRunning || noSplash || (voice && !this.voiceCatalog);
    if (phase === "stopping") showSetupError("Waiting for the current server to stop…", true);
    else if (busy) showSetupError("A server is running. Stop it from the top bar to start another.", true);
    else if (app.voiceRunning) showSetupError("Voice chat is running. End it on the voice chat page to start something else.", true);
    else if ($("#setup-error").dataset.busy) showSetupError(null);
  },
};

/** The command a start runs. A model this Mac has starts --offline: the
 * page never asks the Hub about it (gui/session.py, plan). */
function commandLine(values) {
  const model = app.catalog && app.catalog.models.find((entry) => entry.id === values.model);
  const present = model && (model.installed || model.cached);
  const parts = ["splash-omni", "serve", "--model", values.model || "?", "--port", String(values.port || "")];
  if (values.offline || present) parts.push("--offline");
  if (values.lanes) parts.push("--lanes", String(values.lanes));
  if (values.max_memory && values.max_memory.toLowerCase() !== "auto") parts.push("--max-memory", values.max_memory);
  if (values.disable_ane) parts.push("--disable-ane");
  if (values.task === "asr" && !values.speculative) parts.push("--no-draft-model");
  return parts.join(" ");
}

function showSetupError(message, busy = false) {
  const node = $("#setup-error");
  node.hidden = !message;
  node.textContent = message || "";
  if (busy) node.dataset.busy = "1";
  else delete node.dataset.busy;
}

async function startServer(event) {
  event.preventDefault();
  if (setup.task() === "voice") return startVoice();
  const values = setup.values();
  if (!values.model) return showSetupError("Choose a model.");
  if (!Number.isInteger(values.port) || values.port < 1 || values.port > 65535) {
    return showSetupError("The port must be a number from 1 to 65535.");
  }
  setup.save();
  showSetupError(null);
  $("#btn-start").disabled = true;
  try {
    app.stopRequested = false;
    app.logSeq = 0;
    $("#log").replaceChildren();
    app.downloadRates = {};
    const state = await backend("/api/start", values);
    receiveState(state);
    showView("progress");
    renderProgress();
  } catch (error) {
    showSetupError(error.message);
  } finally {
    $("#btn-start").disabled = false;
    setup.renderBusy();
  }
}

/** Start voice chat's three servers, then follow them on the voice chat
 * page, which shows their start and holds the conversation. */
async function startVoice() {
  showSetupError(null);
  let values;
  try {
    values = setup.voiceValues();
  } catch (error) {
    return showSetupError(error.message);
  }
  setup.save();
  $("#btn-start").disabled = true;
  try {
    await backend("/api/voice/start", values);
    location.href = "/voice";
  } catch (error) {
    showSetupError(error.message);
    $("#btn-start").disabled = false;
    setup.renderBusy();
  }
}

async function loadVoiceCatalog() {
  try {
    setup.voiceCatalog = await backend("/api/voice/catalog");
  } catch (error) {
    setup.voiceCatalog = null;
  }
  if (app.view === "setup") setup.render();
}

/* ------------------------------------------------------------------ */
/* Progress                                                            */

const STEPS = ["prepare", "download", "load", "ready"];

function currentStep(state) {
  if (state.phase === "ready") return "ready";
  if (state.phase === "loading") return "load";
  if (state.step === "download" || state.step === "install") return "download";
  if (state.phase === "failed" && state.step === "load") return "load";
  return "prepare";
}

function receiveState(state) {
  const previous = app.session;
  app.session = state;
  // The server stopped or exited (from this page, another tab or a crash).
  if (previous && previous.phase === "ready" && state.phase !== "ready") haltAudio();
  appendLog(state.log || [], state.log_seq);
  if (previous && previous.phase !== "ready" && state.phase === "ready" && app.view === "progress") {
    setTimeout(() => enterRun({ base: state.base_url, model: state.selection.model, task: state.selection.task, session: true }), 600);
  }
  if (app.view === "progress") {
    if (state.phase === "idle" && app.stopRequested) {
      app.stopRequested = false;
      showView("setup");
      refreshCatalog();
    } else {
      renderProgress();
    }
  }
  if (app.view === "setup") setup.renderBusy();
  if (app.view === "run") {
    updateRunBanner();
    renderTabLock();
  }
  updateTopbar();
}

function appendLog(lines, sequence) {
  const log = $("#log");
  if (sequence !== undefined && sequence < app.logSeq) {
    // A new start: its log begins again.
    log.replaceChildren();
    app.logSeq = 0;
  }
  const nearBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
  for (const line of lines) {
    if (line.seq <= app.logSeq) continue;
    log.append(document.createTextNode(line.text + "\n"));
    app.logSeq = line.seq;
  }
  while (log.childNodes.length > 2000) log.firstChild.remove();
  const live = $("#log-live") || element("span", { id: "log-live", class: "muted" });
  live.textContent = app.session && app.session.live ? app.session.live : "";
  log.append(live);
  if (nearBottom) log.scrollTop = log.scrollHeight;
}

function renderProgress() {
  const state = app.session;
  if (!state) return;
  const selection = state.selection || {};
  const task = (TASK_NAMES[selection.task] || "the server").toLowerCase();
  const titles = {
    preparing: `Getting ${task} ready`,
    loading: "Loading the model",
    ready: "Ready",
    stopping: "Stopping",
    failed: `${task[0].toUpperCase()}${task.slice(1)} did not start`,
    idle: "Stopped",
  };
  $("#progress-title").textContent = titles[state.phase] || "Starting";
  const entry = app.catalog && app.catalog.models.find((model) => model.id === selection.model);
  $("#progress-model").textContent = selection.model ? (entry ? entry.label : selection.model) : "";
  $("#progress-command").textContent = state.command
    ? [...(state.fetch || []).map((repo) => `fetch ${repo}`), state.command].join("  →  ")
    : "";
  const step = currentStep(state);
  const index = STEPS.indexOf(step);
  const downloadsDone = (state.downloads || []).every((entry) => entry.state === "cached" || entry.state === "complete");
  $$("#steps li").forEach((item, position) => {
    item.className = "";
    if (state.phase === "ready" || position < index) item.classList.add("done");
    else if (position === index) item.classList.add(state.phase === "failed" ? "failed" : "active");
    if (item.dataset.step === "download" && downloadsDone && index > 1) item.className = "done";
  });
  $("#progress-detail").textContent = state.phase === "failed" ? "" : (state.detail || "");
  renderDownloads(state);
  const failed = state.phase === "failed";
  const error = $("#progress-error");
  error.hidden = !failed;
  error.textContent = failed ? state.error || "The server exited." : "";
  $("#btn-cancel").hidden = !(state.phase === "preparing" || state.phase === "loading");
  $("#btn-back").hidden = !(failed || state.phase === "idle");
  // The launcher's refusal when another `splash-omni serve` holds the port.
  const portTaken = failed && /already serving/i.test(state.error || "");
  $("#btn-use-running").hidden = !portTaken;
  if (failed) $("#log-details").open = true;
}

function renderDownloads(state) {
  const container = $("#downloads");
  container.replaceChildren();
  const now = performance.now();
  for (const entry of state.downloads || []) {
    const rate = app.downloadRates[entry.repo] || {};
    let speed = "";
    if (entry.present !== null && rate.present !== undefined && now > rate.time) {
      const perSecond = (entry.present - rate.present) / ((now - rate.time) / 1000);
      rate.speed = perSecond > 0 ? perSecond : rate.speed;
    }
    if (entry.state === "downloading" && rate.speed) speed = ` · ${formatBytes(rate.speed)}/s`;
    if (entry.present !== null && (rate.present === undefined || entry.present !== rate.present)) {
      rate.present = entry.present;
      rate.time = now;
    }
    app.downloadRates[entry.repo] = rate;
    let text;
    let fraction = null;
    let barClass = "bar";
    switch (entry.state) {
      case "cached":
        text = "Already on this Mac";
        fraction = 1;
        barClass += " done";
        break;
      case "complete":
        text = entry.total ? `Downloaded · ${formatBytes(entry.total)}` : "Downloaded";
        fraction = 1;
        barClass += " done";
        break;
      case "downloading":
        if (entry.total) {
          fraction = Math.min(entry.present / entry.total, 1);
          text = `${formatBytes(entry.present)} of ${formatBytes(entry.total)} · ${Math.floor(fraction * 100)}%${speed}`;
        } else {
          text = `${formatBytes(entry.present)} downloaded${speed}`;
          barClass += " indeterminate";
        }
        break;
      case "pending":
        fraction = entry.total ? entry.present / entry.total : 0;
        text = entry.total ? `Waiting · ${formatBytes(entry.total)} to fetch` : "Waiting";
        break;
      case "listing":
        text = "Checking what to download…";
        barClass += " indeterminate";
        break;
      default:
        text = "Size unknown";
        barClass += state.phase === "preparing" ? " indeterminate" : "";
    }
    const bar = element("div", { class: barClass }, element("span"));
    if (fraction !== null) bar.firstChild.style.width = (fraction * 100).toFixed(1) + "%";
    container.append(element("div", { class: "download-row" },
      element("div", { class: "download-head" },
        element("span", { class: "repo", text: entry.repo }),
        element("span", { class: "muted", text })),
      bar));
  }
}

async function cancelStart() {
  app.stopRequested = true;
  try {
    receiveState(await backend("/api/stop", {}));
  } catch (error) {
    $("#progress-error").hidden = false;
    $("#progress-error").textContent = error.message;
  }
}

async function backToSetup() {
  try {
    if (app.session && app.session.phase === "failed") receiveState(await backend("/api/stop", {}));
  } catch (error) {
    /* The failed session is cleared on the next start anyway. */
  }
  showView("setup");
  refreshCatalog();
}

async function refreshCatalog() {
  try {
    app.catalog = await backend("/api/catalog");
    setup.render();
  } catch (error) {
    /* Keep the last catalog. */
  }
}

/* ------------------------------------------------------------------ */
/* Top bar                                                             */

function updateTopbar() {
  const pill = $("#server-pill");
  const text = $("#server-pill-text");
  const state = app.session;
  const running = state && ["preparing", "loading", "ready"].includes(state.phase);
  pill.className = "pill";
  if (app.view === "run" && endpoint.external) {
    pill.hidden = false;
    if (endpoint.reachable !== null) pill.classList.add(endpoint.reachable ? "ready" : "down");
    text.textContent = `Connected · ${endpoint.base}`;
  } else if (state && state.selection && state.phase !== "idle") {
    pill.hidden = false;
    const task = TASK_NAMES[state.selection.task] || "Server";
    if (state.phase === "ready") {
      pill.classList.add("ready");
      text.textContent = `${task} · ready`;
    } else if (state.phase === "failed") {
      pill.classList.add("down");
      text.textContent = `${task} · stopped`;
    } else {
      pill.classList.add("busy");
      text.textContent = `${task} · ${state.phase === "stopping" ? "stopping" : "starting"}`;
    }
    pill.title = state.selection.model;
  } else {
    pill.hidden = true;
  }
  // In the run view: stop the server (or leave a connected one) and choose
  // another task.
  const stop = $("#btn-stop");
  stop.hidden = !(app.view === "run" || (running && app.view !== "progress"));
  stop.textContent = app.view === "run" && endpoint.external ? "Disconnect" : "Stop";
}

/** Silence the page when its server goes away: stop generating and
 * transcribing, and pause playback (what arrived stays replayable). */
function haltAudio(message = "Server stopped.") {
  if (tts.run) tts.run.stop();
  if (tts.player && tts.player.playing) {
    tts.player.pause();
    renderPlayer();
  }
  if (asr.controller) asr.controller.abort();
  if (asr.recording) stopRecording();
  const preview = $("#asr-preview");
  if (preview && !preview.paused) preview.pause();
  if (live.session && live.session.running) stopLive(message);
}

async function stopServer(goToSetup) {
  const state = app.session;
  // A connected server is someone else's: leave it running.
  const connected = app.view === "run" && endpoint.external;
  if (!connected && state && ["preparing", "loading", "ready"].includes(state.phase)) {
    const task = TASK_NAMES[state.selection.task] || "the server";
    if (!confirm(`Stop ${task.toLowerCase()}? Its model is unloaded, and you can choose another task.`)) return;
    haltAudio();
    try {
      receiveState(await backend("/api/stop", {}));
    } catch (error) {
      alert(error.message);
      return;
    }
  }
  if (goToSetup || app.view === "run") {
    showView("setup");
    refreshCatalog();
  }
}

/* ------------------------------------------------------------------ */
/* Run: the API endpoint                                               */

const endpoint = {
  base: "http://127.0.0.1:8000",
  model: "",
  key: "",
  viaProxy: true,
  external: false,
  reachable: null,
  languages: null,
  task: null,
  // /v1/models' entries, and what the chosen one takes for a voice
  // (SplashLogic.voiceControls).
  models: [],
  controls: SplashLogic.voiceControls(null),
};

/* ------------------------------------------------------------------ */
/* Run: the voice a speech model takes                                 */

// Example system prompts: a VoiceDesign description (timbre, rhythm,
// style), or a CustomVoice style or emotion.
const VOICE_DESIGN_EXAMPLES = [
  ["Warm narrator", "A warm, low-pitched male narrator in his forties, speaking slowly and calmly with clear articulation, like a late-night radio host."],
  ["Bright and quick", "A bright, energetic young woman with a light, clear timbre, a quick rhythm and a cheerful, smiling tone."],
  ["Gravelly elder", "An elderly man with a deep, gravelly voice, an unhurried rhythm with long pauses, and a dry, wry sense of humour."],
  ["Soft whisper", "A soft, breathy female whisper, intimate and slow, as if telling a secret late at night."],
  ["新闻播音", "一位专业的女性新闻主播，声音清亮、吐字清晰，语速适中，语气沉稳而客观。"],
];
const CUSTOM_VOICE_EXAMPLES = [
  ["Happy", "Speak happily, with excitement in your voice."],
  ["Sad", "Say it sadly, in a low, slow voice."],
  ["Angry", "Speak angrily and fast."],
  ["Whisper", "Whisper it softly."],
  ["温柔", "用温柔、亲切的语气说。"],
];
const MODEL_TYPE_NAMES = {
  base: "Natural",
  voice_design: "Described in words",
  custom_voice: "Preset speakers",
};

/** Take the voice controls of the endpoint's chosen model: from its
 * /v1/models entry, and for a preset speaker list it lacks, from
 * /v1/audio/voices. A Base or VoiceDesign model has no speakers to ask for. */
async function applyVoiceControls(model) {
  const controls = SplashLogic.voiceControls(model);
  if (!controls.voices && (controls.type === null || controls.type === SplashLogic.CUSTOM_VOICE) &&
      (!model || model.task !== "transcription")) {
    try {
      const answer = await api("/v1/audio/voices");
      if (answer.ok) {
        const voices = SplashLogic.voiceList(await answer.json());
        // A lone "default" is no choice.
        if (voices.length && !(voices.length === 1 && voices[0] === "default")) {
          controls.voices = voices;
          controls.freeVoice = false;
        }
      }
    } catch (error) {
      /* Not every server lists voices. */
    }
  }
  endpoint.controls = controls;
  renderVoiceControls();
}

/** The controls a model type takes before the server answers: a model the
 * setup started has its type in the catalog. */
function presetVoiceControls(modelId) {
  const entry = app.catalog && app.catalog.models.find((model) => model.id === modelId);
  endpoint.controls = SplashLogic.voiceControls(entry && entry.variant ? { tts_model_type: entry.variant } : null);
  renderVoiceControls();
}

function renderVoiceControls() {
  const controls = endpoint.controls;
  const settings = store.get("tts", {});
  $("#tts-model-type").textContent = controls.type ? MODEL_TYPE_NAMES[controls.type] : "As the server takes it";
  $("#tts-model-type").dataset.type = controls.type || "unknown";
  // Instructions: the system prompt.
  const takesPrompt = controls.instructions !== "unsupported";
  $("#tts-instructions-row").hidden = !takesPrompt;
  $("#tts-base-note").hidden = controls.type !== SplashLogic.BASE;
  const tag = $("#tts-instructions-tag");
  tag.textContent = controls.instructions === "required" ? "required" : "optional";
  tag.className = "tag" + (controls.instructions === "required" ? " required" : "");
  const prompt = $("#tts-instructions");
  const key = "instructions_" + (controls.type || "other");
  if (prompt.dataset.key !== key) {
    // Each model type keeps its own prompt.
    if (prompt.dataset.key) saveTtsSettings();
    prompt.dataset.key = key;
    const saved = store.get("tts", {})[key];
    // A first VoiceDesign run starts from an example description.
    prompt.value = typeof saved === "string" ? saved
      : controls.type === SplashLogic.VOICE_DESIGN ? VOICE_DESIGN_EXAMPLES[0][1] : "";
  }
  let examples = [];
  if (controls.type === SplashLogic.VOICE_DESIGN) {
    prompt.placeholder = "Describe the voice: gender and age, timbre and pitch, rhythm and pace, style and emotion.";
    $("#tts-instructions-label").textContent = "Description";
    $("#tts-instructions-hint").textContent = "The model creates the voice from this description. Try an example:";
    examples = VOICE_DESIGN_EXAMPLES;
  } else if (controls.type === SplashLogic.CUSTOM_VOICE) {
    prompt.placeholder = "A style or emotion, such as “Speak happily, with excitement.”";
    $("#tts-instructions-label").textContent = "Style";
    $("#tts-instructions-hint").textContent = "The speaker sets the voice; this sets the mood. Try one:";
    examples = CUSTOM_VOICE_EXAMPLES;
  } else {
    prompt.placeholder = "Optional: sent as the request's instructions.";
    $("#tts-instructions-label").textContent = "Instructions";
    $("#tts-instructions-hint").textContent = "Sent as the request's instructions when filled; a server that does not take them may refuse the request.";
  }
  $("#tts-examples").replaceChildren(...examples.map(([name, text]) =>
    element("button", { type: "button", class: "chip", title: text, text: name, onclick: () => {
      prompt.value = text;
      updatePromptCount();
      saveTtsSettings();
    } })));
  updatePromptCount();
  // The speaker.
  const speakers = $("#tts-speaker");
  $("#tts-speaker-row").hidden = !controls.voices;
  if (controls.voices) {
    const current = speakers.value || settings.speaker || "";
    const options = controls.voiceRequired ? [] : [element("option", { value: "", text: "(server default)" })];
    speakers.replaceChildren(...options, ...controls.voices.map((voice) =>
      element("option", { value: voice, text: voice.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase()) })));
    if (controls.voices.includes(current)) speakers.value = current;
    // A list without an empty choice always has a speaker: no tag.
    $("#tts-speaker-tag").textContent = "optional";
    $("#tts-speaker-tag").hidden = controls.voiceRequired;
  }
  $("#tts-voice-row").hidden = !controls.freeVoice;
  updateTtsCounts();
}

function updatePromptCount() {
  const length = $("#tts-instructions").value.trim().length;
  $("#tts-instructions-count").textContent = length ? `${length} characters` : "";
}

function loadEndpointFields() {
  $("#ep-base").value = endpoint.base;
  $("#ep-model").value = endpoint.model;
  $("#ep-key").value = endpoint.key;
  $("#ep-proxy").checked = endpoint.viaProxy;
  $("#endpoint-summary").textContent = endpoint.base;
}

function readEndpointFields() {
  const base = $("#ep-base").value.trim().replace(/\/+$/, "");
  if (base !== endpoint.base) {
    endpoint.base = base;
    endpoint.reachable = null;
    const session = app.session;
    endpoint.external = !(session && session.base_url === base && session.phase === "ready");
  }
  const model = $("#ep-model").value.trim();
  if (model !== endpoint.model) {
    endpoint.model = model;
    // Another model of the same server: its own voice controls.
    const entry = endpoint.models.find((item) => item.id === model);
    if (entry) applyVoiceControls(entry);
  }
  endpoint.key = $("#ep-key").value;
  endpoint.viaProxy = $("#ep-proxy").checked;
  $("#endpoint-summary").textContent = endpoint.base;
  store.set("endpoint", { base: endpoint.base, model: endpoint.model, viaProxy: endpoint.viaProxy });
  updateTopbar();
}

/** An API request to the endpoint: through the page's proxy, or directly. */
function api(path, init = {}) {
  const headers = new Headers(init.headers || {});
  if (endpoint.key) headers.set("Authorization", "Bearer " + endpoint.key);
  let url;
  if (endpoint.viaProxy) {
    url = "/proxy" + path;
    headers.set("X-Splash-Upstream", endpoint.base);
  } else {
    url = endpoint.base + path;
  }
  return fetch(url, { ...init, headers, cache: "no-store" });
}

async function errorMessage(response) {
  const text = await response.text().catch(() => "");
  try {
    const data = JSON.parse(text);
    const error = data.error;
    if (typeof error === "string") return `${response.status}: ${error}`;
    if (error && error.message) return `${response.status}: ${error.message}`;
    if (data.detail) return `${response.status}: ${typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)}`;
  } catch (error) {
    /* Not JSON. */
  }
  return `${response.status} ${response.statusText}${text ? ": " + text.slice(0, 300) : ""}`;
}

async function checkEndpoint() {
  readEndpointFields();
  const status = $("#endpoint-status");
  status.textContent = "Checking…";
  try {
    const response = await api("/v1/models");
    if (!response.ok) throw new Error(await errorMessage(response));
    const data = await response.json();
    const models = Array.isArray(data.data) ? data.data : [];
    $("#ep-models").replaceChildren(...models.map((model) => element("option", { value: model.id })));
    const chosen = models.find((model) => model.id === endpoint.model) || models[0];
    if (chosen && !endpoint.model) {
      endpoint.model = chosen.id;
      $("#ep-model").value = chosen.id;
    }
    endpoint.models = models;
    endpoint.languages = chosen && Array.isArray(chosen.languages) ? chosen.languages : null;
    endpoint.task = chosen && chosen.task === "text-to-speech" ? "tts" : chosen && chosen.task === "transcription" ? "asr" : null;
    await applyVoiceControls(chosen || null);
    let ready = "";
    try {
      const answer = await api("/ready");
      ready = answer.ok ? " · ready" : answer.status === 404 ? "" : " · not ready";
    } catch (error) {
      /* Not every server has /ready. */
    }
    endpoint.reachable = true;
    status.textContent = `${models.length} model${models.length === 1 ? "" : "s"}${ready}`;
    fillLanguages();
  } catch (error) {
    endpoint.reachable = false;
    status.textContent = `Unreachable: ${error.message}`;
    if (!endpoint.viaProxy) status.textContent += " (try the proxy: the server may not allow this page's origin)";
  }
  updateTopbar();
}

function fillLanguages() {
  const ttsSelect = $("#tts-language");
  const asrSelect = $("#asr-language");
  const ttsSaved = ttsSelect.value || (store.get("tts", {}).language);
  const asrSaved = asrSelect.value || (store.get("asr", {}).language);
  const liveSelect = $("#live-language");
  const liveSaved = liveSelect.value || (store.get("live", {}).language);
  let ttsNames = TTS_LANGUAGES;
  let asrNames = Object.keys(ISO_CODES);
  if (endpoint.languages && endpoint.languages.length) {
    if (endpoint.task === "tts") ttsNames = endpoint.languages;
    if (endpoint.task === "asr") asrNames = endpoint.languages;
  }
  ttsSelect.replaceChildren(element("option", { value: "", text: "Automatic" }),
    ...ttsNames.map((name) => element("option", { value: name.toLowerCase(), text: name })));
  asrSelect.replaceChildren(element("option", { value: "", text: "Automatic (detect)" }),
    ...asrNames.map((name) => element("option", { value: ISO_CODES[name] || name, text: name })));
  if (ttsSaved && $$("option", ttsSelect).some((option) => option.value === ttsSaved)) ttsSelect.value = ttsSaved;
  if (asrSaved && $$("option", asrSelect).some((option) => option.value === asrSaved)) asrSelect.value = asrSaved;
  liveSelect.replaceChildren(...$$("option", asrSelect).map((option) => option.cloneNode(true)));
  liveSelect.value = liveSaved && $$("option", liveSelect).some((option) => option.value === liveSaved) ? liveSaved : "";
}

function enterRun({ base, model, task, session }) {
  endpoint.base = base.replace(/\/+$/, "");
  if (model !== undefined) endpoint.model = model;
  endpoint.external = !session;
  endpoint.reachable = null;
  loadEndpointFields();
  presetVoiceControls(session ? endpoint.model : null);
  $("#endpoint-card").open = !session;
  showView("run");
  if (task) selectTab(task);
  renderTabLock();
  checkEndpoint().then(() => {
    if (!task && endpoint.task) selectTab(endpoint.task);
    renderTabLock();
  });
  updateRunBanner();
}

/** The task of the server the run view sends to: the page's own server's,
 * or what a connected server reports (null while unknown). */
function runTask() {
  if (endpoint.external) return endpoint.task;
  const selection = app.session && app.session.selection;
  return (selection && selection.task) || endpoint.task || null;
}

/** Show only the tabs the server answers (none when it answers one), and
 * leave a tab it cannot; title the view after its task and model. */
function renderTabLock() {
  const task = runTask();
  const allowed = SplashLogic.runTabs(task);
  for (const tab of ["tts", "asr", "live"]) $(`#tab-${tab}`).hidden = !allowed.includes(tab);
  $("#run-tabs").hidden = allowed.length < 2;
  if (!allowed.includes(app.tab)) selectTab(allowed[0]);
  $("#run-title").textContent = task ? TASK_NAMES[task] : "Connected server";
  const entry = app.catalog && app.catalog.models.find((model) => model.id === endpoint.model);
  const name = entry ? entry.label : endpoint.model;
  $("#run-subtitle").textContent = [name, endpoint.base.replace(/^https?:\/\//, "")].filter(Boolean).join(" · ");
}

function selectTab(name) {
  const allowed = SplashLogic.runTabs(app.view === "run" ? runTask() : null);
  if (!allowed.includes(name)) name = allowed[0];
  app.tab = name;
  for (const tab of ["tts", "asr", "live"]) {
    $(`#tab-${tab}`).setAttribute("aria-selected", String(tab === name));
    $(`#panel-${tab}`).hidden = tab !== name;
  }
  store.set("tab", name);
  if (name === "tts") ttsResize();
}

function updateRunBanner() {
  const state = app.session;
  if (endpoint.external || !state) return;
  const status = $("#tts-status");
  if (state.phase === "failed" && !status.dataset.serverDown) {
    status.dataset.serverDown = "1";
    status.textContent = "The server stopped: " + (state.error || "it exited") + ". Use Stop, then start it again.";
    status.classList.add("error-text");
  }
}

/* ------------------------------------------------------------------ */
/* Speech: WAV/PCM stream decoding, ordered playback, metrics          */

/** Turns an audio response's bytes, as they arrive, into mono int16
 * samples: a WAV (of known or unknown length) or raw 16-bit PCM. */
class PcmDecoder {
  constructor(format, contentType) {
    const type = (contentType || "").toLowerCase();
    this.raw = format === "pcm" || /audio\/(pcm|l16|raw)/.test(type);
    this.opaque = !this.raw && type !== "" && !/wav|wave|octet-stream|audio\/x-wav/.test(type);
    const rate = /rate=(\d+)/.exec(type);
    this.rate = rate ? Number(rate[1]) : DEFAULT_PCM_RATE;
    this.channels = 1;
    this.bits = 16;
    this.float = false;
    this.header = this.raw ? null : new Uint8Array(0);
    this.carry = new Uint8Array(0);
    this.remaining = Infinity;
  }

  static concat(a, b) {
    if (!a.length) return b;
    const out = new Uint8Array(a.length + b.length);
    out.set(a);
    out.set(b, a.length);
    return out;
  }

  /** The samples `bytes` completes, or null while the header is incomplete.
   * Throws for a response that is not WAV. */
  push(bytes) {
    if (this.header) {
      this.header = PcmDecoder.concat(this.header, bytes);
      const start = this.parseHeader(this.header);
      if (start === null) return null;
      bytes = this.header.subarray(start);
      this.header = null;
    }
    if (this.remaining !== Infinity) {
      bytes = bytes.subarray(0, Math.max(0, Math.min(bytes.length, this.remaining)));
      this.remaining -= bytes.length;
    }
    const data = PcmDecoder.concat(this.carry, bytes);
    const frame = this.channels * (this.bits / 8);
    const usable = data.length - (data.length % frame);
    this.carry = data.slice(usable);
    return this.convert(data.subarray(0, usable), frame);
  }

  parseHeader(head) {
    if (head.length < 12) return null;
    const view = new DataView(head.buffer, head.byteOffset, head.byteLength);
    const tag = (offset) => String.fromCharCode(...head.subarray(offset, offset + 4));
    if (tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("the response is not a WAV file");
    let offset = 12;
    for (;;) {
      if (head.length < offset + 8) return null;
      const id = tag(offset);
      const size = view.getUint32(offset + 4, true);
      if (id === "data") {
        this.remaining = size === 0 || size === 0xffffffff ? Infinity : size;
        return offset + 8;
      }
      if (id === "fmt ") {
        if (head.length < offset + 24) return null;
        const code = view.getUint16(offset + 8, true);
        this.channels = view.getUint16(offset + 10, true) || 1;
        this.rate = view.getUint32(offset + 12, true) || DEFAULT_PCM_RATE;
        this.bits = view.getUint16(offset + 22, true) || 16;
        this.float = code === 3 || (code === 0xfffe && this.bits === 32 && size >= 40 && view.getUint16(offset + 32, true) === 3);
        if (![8, 16, 24, 32].includes(this.bits)) throw new Error(`unsupported WAV sample size ${this.bits}`);
      }
      offset += 8 + size + (size & 1);
      if (offset > 1 << 20) throw new Error("the WAV header has no data chunk");
    }
  }

  convert(bytes, frame) {
    const count = bytes.length / frame;
    if (this.channels === 1 && this.bits === 16 && !this.float) {
      const copy = new Uint8Array(bytes);
      return new Int16Array(copy.buffer, 0, count);
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const out = new Int16Array(count);
    const width = this.bits / 8;
    for (let i = 0; i < count; i++) {
      let sum = 0;
      for (let channel = 0; channel < this.channels; channel++) {
        const at = i * frame + channel * width;
        let value;
        if (this.float) value = view.getFloat32(at, true);
        else if (this.bits === 16) value = view.getInt16(at, true) / 32768;
        else if (this.bits === 8) value = (view.getUint8(at) - 128) / 128;
        else if (this.bits === 24) value = ((view.getUint8(at) | (view.getUint8(at + 1) << 8) | (view.getInt8(at + 2) << 16))) / 8388608;
        else value = view.getInt32(at, true) / 2147483648;
        sum += value;
      }
      out[i] = Math.max(-32768, Math.min(32767, Math.round((sum / this.channels) * 32767)));
    }
    return out;
  }
}

const int16FromFloat = SplashLogic.int16FromFloat;

function wavBlob(chunks, rate) {
  return new Blob([SplashLogic.wavBytes(chunks, rate)], { type: "audio/wav" });
}

function formatClock(seconds) {
  if (!isFinite(seconds) || seconds < 0) seconds = 0;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}:${rest.toFixed(1).padStart(4, "0")}`;
}

const tts = { run: null, url: null, lastWav: null, player: null };
let ttsResize = () => {};

/** The player's controls and progress, for the run whose player it is. */
function renderPlayer() {
  const player = tts.player;
  const run = tts.run;
  const root = $("#tts-player");
  if (!player) {
    root.dataset.state = "empty";
    $("#tts-play").disabled = true;
    $("#tts-time").textContent = "0:00.0 / 0:00.0";
    for (const id of ["#tts-received", "#tts-played", "#tts-generating"]) $(id).style.width = "0";
    $("#tts-head").style.left = "0";
    $("#tts-progress").textContent = "";
    return;
  }
  player.tick();
  const generating = run && run.finishedAt === null;
  const length = player.length;
  const position = player.position;
  // While audio is still coming, its received part fills most of the bar
  // and a moving stripe stands for the rest; before any, the stripe is all.
  const share = generating ? (length > 0 ? 0.88 : 0) : 1;
  const played = length > 0 ? (position / length) * share : 0;
  $("#tts-received").style.width = `${share * 100}%`;
  $("#tts-generating").style.left = `${share * 100}%`;
  $("#tts-generating").style.width = generating ? `${(1 - share) * 100}%` : "0";
  $("#tts-played").style.width = `${played * 100}%`;
  $("#tts-head").style.left = `${played * 100}%`;
  root.dataset.state = player.playing ? "playing" : generating ? "generating" : length > 0 ? "paused" : "empty";
  const button = $("#tts-play");
  button.disabled = length <= 0 && !generating;
  button.setAttribute("aria-label", player.playing ? "Pause" : "Play");
  button.classList.toggle("pause", player.playing);
  $("#tts-time").textContent = `${formatClock(position)} / ${formatClock(length)}`;
  const timeline = $("#tts-timeline");
  timeline.setAttribute("aria-valuemax", length.toFixed(1));
  timeline.setAttribute("aria-valuenow", position.toFixed(1));
  timeline.setAttribute("aria-valuetext", `${formatClock(position)} of ${formatClock(length)}`);
  if (run) {
    const elapsed = (run.finishedAt !== null ? run.finishedAt : performance.now() - run.started) / 1000;
    const received = run.audioSeconds;
    let text;
    if (generating) {
      text = received > 0
        ? `Generating · ${received.toFixed(1)} s of audio received in ${elapsed.toFixed(1)} s`
        : `Generating · waiting for the first audio · ${elapsed.toFixed(1)} s` +
          ($("#tts-stream").checked ? "" : " (not streamed: the audio arrives when it is done)");
      if (player.playing && position >= length - 0.05 && length > 0) text += " · playback waiting for audio";
    } else {
      text = received > 0 ? `${received.toFixed(1)} s of audio in ${elapsed.toFixed(2)} s` : "";
    }
    $("#tts-progress").textContent = text;
  }
}

function togglePlayback() {
  const player = tts.player;
  if (!player) return;
  getAudioContext();
  if (player.playing) player.pause();
  else player.play();
  renderPlayer();
}

function seekFromPointer(event) {
  const player = tts.player;
  if (!player || player.length <= 0) return;
  const run = tts.run;
  const share = run && run.finishedAt === null ? 0.88 : 1;
  const box = $("#tts-timeline").getBoundingClientRect();
  const fraction = Math.min(Math.max((event.clientX - box.left) / box.width / share, 0), 1);
  player.seek(fraction * player.length);
  renderPlayer();
}

function seekByKey(event) {
  const player = tts.player;
  if (!player) return;
  const steps = { ArrowLeft: -2, ArrowRight: 2, Home: -Infinity, End: Infinity };
  if (event.key === " " || event.key === "Enter") {
    event.preventDefault();
    togglePlayback();
  } else if (event.key in steps) {
    event.preventDefault();
    const step = steps[event.key];
    player.seek(isFinite(step) ? player.position + step : step < 0 ? 0 : player.length);
    renderPlayer();
  }
}

function speechParameters() {
  const body = {};
  const language = $("#tts-language").value;
  if (language) body.language = language;
  for (const [name, integer] of [["temperature", false], ["top_k", true], ["top_p", false],
    ["repetition_penalty", false], ["max_frames", true], ["chunk_frames", true]]) {
    const value = numberField(`#tts-${name}`, integer);
    if (value !== null) body[name] = value;
  }
  body.response_format = $("#tts-format").value;
  body.stream = $("#tts-stream").checked;
  return body;
}

/** The voice the page holds for a request: the system prompt and the
 * speaker (a preset) or voice name (another server's). */
function voiceInputs() {
  const controls = endpoint.controls;
  return {
    instructions: controls.instructions === "unsupported" ? "" : $("#tts-instructions").value,
    voice: controls.voices ? $("#tts-speaker").value : controls.freeVoice ? $("#tts-voice").value : "",
  };
}

function saveTtsSettings() {
  const settings = store.get("tts", {});
  settings.paragraphs = $("#tts-paragraphs").checked;
  for (const id of ["language", "temperature", "top_k", "top_p", "repetition_penalty", "seed",
    "max_frames", "chunk_frames", "voice", "format", "speed"]) settings[id] = $(`#tts-${id}`).value;
  for (const id of ["stream", "autoplay", "shared-seed", "chunks"]) settings[id] = $(`#tts-${id}`).checked;
  settings.text = $("#tts-text").value;
  const prompt = $("#tts-instructions");
  if (prompt.dataset.key) settings[prompt.dataset.key] = prompt.value;
  if ($("#tts-speaker").value) settings.speaker = $("#tts-speaker").value;
  store.set("tts", settings);
}

function restoreTtsSettings() {
  const settings = store.get("tts", {});
  for (const id of ["temperature", "top_k", "top_p", "repetition_penalty", "seed",
    "max_frames", "chunk_frames", "voice", "format", "speed"]) {
    if (settings[id] !== undefined) $(`#tts-${id}`).value = settings[id];
  }
  for (const id of ["stream", "autoplay", "shared-seed", "chunks"]) {
    if (settings[id] !== undefined) $(`#tts-${id}`).checked = settings[id];
  }
  $("#tts-paragraphs").checked = !!settings.paragraphs;
  $("#tts-text").value = settings.text || SAMPLE_TEXT;
}

/** Whether this run splits the text: the switch, where the model offers it. */
function paragraphModeOn() {
  return endpoint.controls.paragraphs && $("#tts-paragraphs").checked;
}

function updateTtsCounts() {
  const text = $("#tts-text").value;
  $("#tts-count").textContent = `${text.trim().length} characters`;
  const allowed = endpoint.controls.paragraphs;
  const paragraphMode = paragraphModeOn();
  $("#tts-paragraph-row").hidden = !allowed;
  $("#paragraph-unavailable").hidden = allowed;
  $("#paragraph-note").hidden = !paragraphMode;
  const controls = endpoint.controls;
  $("#paragraph-note-controls").textContent = controls.voices || controls.freeVoice
    ? (controls.instructions === "unsupported" ? "speaker" : "system prompt and speaker")
    : "system prompt";
  $("#paragraph-note-anchor").hidden = !controls.references;
  $("#paragraph-note-plain").hidden = controls.references;
  $("#tts-shared-seed-row").hidden = !paragraphMode;
  const count = SplashLogic.paragraphs(text).length;
  const chunks = paragraphMode ? 1 : chunkedText(text.trim()).length;
  $("#tts-chunks-row").hidden = paragraphMode || !controls.references;
  $("#tts-paragraph-count").textContent = paragraphMode
    ? `${count} paragraph${count === 1 ? "" : "s"} → ${count} request${count === 1 ? "" : "s"}`
    : chunks > 1 ? `${chunks} chunks → ${chunks} requests` : "one request";
}

function setMetric(root, name, value, sub) {
  // The figures appear with the first result.
  if (value !== "–") root.hidden = false;
  $(`[data-metric="${name}"]`, root).textContent = value;
  if (sub !== undefined) $(`[data-metric="${name}-sub"]`, root).textContent = sub;
}

function ttsStatus(message, error = false) {
  const status = $("#tts-status");
  status.textContent = message;
  status.classList.toggle("error-text", error);
  delete status.dataset.serverDown;
}

class SpeechRun {
  /** `bodies`: one request body per segment, built alike
   * (SplashLogic.speechBody). `anchored`: the server continues an earlier
   * request (reference_speech_id), so paragraphs after the first are sent
   * once it is done, as its continuations, in its voice. */
  constructor(bodies, { play, anchored = false, chunked = false, speed = 1 }) {
    this.bodies = bodies;
    this.speed = speed;
    this.paragraphMode = bodies.length > 1;
    this.chunked = chunked && this.paragraphMode;
    this.anchored = anchored && this.paragraphMode;
    this.started = performance.now();
    this.controller = new AbortController();
    this.segments = bodies.map((body) => ({ text: body.input, chunks: [], samples: 0, rate: DEFAULT_PCM_RATE, state: "waiting" }));
    this.firstAudio = null;
    this.firstAudioSegment = null;
    this.finishedAt = null;
    this.player = new SplashLogic.TimelinePlayer(bodies.length, {
      context: getAudioContext,
      gap: this.chunked ? CHUNK_GAP : PARAGRAPH_GAP,
      onStart: (index) => {
        this.firstAudio = performance.now() - this.started;
        this.firstAudioSegment = index;
        this.render();
      },
    });
    if (play) this.player.play();
    this.timer = setInterval(() => this.render(), 100);
  }

  /** Seconds of audio the server generated (before the page's speed). */
  get audioSeconds() {
    return this.segments.reduce((sum, segment) => sum + (segment.modelSamples || 0) / segment.rate, 0);
  }

  /** Send segment `index`; `onSpeechId` learns its speech id as soon as
   * its answer starts (or null when it has none). */
  async request(index, onSpeechId = null) {
    const segment = this.segments[index];
    const body = this.bodies[index];
    segment.state = "sent";
    let told = false;
    const tell = (id) => {
      if (onSpeechId && !told) onSpeechId(id);
      told = true;
    };
    try {
      const send = () => api("/v1/audio/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: this.controller.signal,
      });
      let response = await send();
      if (!response.ok && body.reference_speech_id && response.status < 500) {
        // The first segment cannot be continued (it failed, or is longer
        // than a reference may be): speak this one on its own.
        delete body.reference_speech_id;
        response = await send();
      }
      if (!response.ok) throw new Error(await errorMessage(response));
      segment.speechId = response.headers.get("X-Splash-Speech-Id");
      tell(segment.speechId);
      segment.serverFirstAudio = response.headers.get("X-Splash-Time-To-First-Audio-Ms");
      segment.finishReason = response.headers.get("X-Splash-Finish-Reason");
      segment.frames = response.headers.get("X-Splash-Frames");
      segment.state = "streaming";
      const decoder = new PcmDecoder(body.response_format, response.headers.get("Content-Type"));
      // Played faster at the same pitch (SplashLogic.TimeStretcher).
      let stretcher = null;
      const receive = (raw, rate, last = false) => {
        if (raw && raw.length) segment.modelSamples = (segment.modelSamples || 0) + raw.length;
        if (!stretcher && raw && raw.length) stretcher = new SplashLogic.TimeStretcher(rate, this.speed);
        if (!stretcher) return;
        let samples = raw && raw.length ? stretcher.push(raw) : new Int16Array(0);
        if (last) {
          const rest = stretcher.flush();
          const joined = new Int16Array(samples.length + rest.length);
          joined.set(samples);
          joined.set(rest, samples.length);
          samples = joined;
        }
        if (!samples.length) return;
        if (segment.firstAudio === undefined) segment.firstAudio = performance.now() - this.started;
        segment.rate = rate;
        segment.chunks.push(samples);
        segment.samples += samples.length;
        this.player.push(index, samples, rate);
      };
      if (decoder.opaque) {
        // Compressed audio (mp3, ogg, ...) from another server: decode it whole.
        const decoded = await decodeAudio(await response.arrayBuffer());
        receive(int16FromFloat(decoded.getChannelData(0)), decoded.sampleRate, true);
      } else {
        const reader = response.body.getReader();
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          receive(decoder.push(value), decoder.rate);
        }
        receive(null, decoder.rate, true);
        if (decoder.header) throw new Error("the response ended inside its WAV header");
      }
      segment.state = "done";
    } catch (error) {
      segment.state = error.name === "AbortError" ? "stopped" : "failed";
      segment.error = error.name === "AbortError" ? "stopped" : error.message;
    }
    tell(null);
    segment.done = performance.now() - this.started;
    this.player.finish(index);
    this.render();
  }

  async start() {
    const count = this.bodies.length;
    const unit = this.chunked ? "chunks" : "paragraphs";
    ttsStatus(!this.paragraphMode
      ? "Generating…"
      : this.anchored
        ? `Generating ${count} ${unit}: the first, and the others as its continuations, in its voice…`
        : `Generating ${count} ${unit} as ${count} concurrent requests…`);
    this.render();
    if (this.anchored) {
      // The others are sent as soon as the first's answer starts: the
      // server holds a continuation until what it continues is done.
      let first;
      const anchor = await new Promise((resolve) => { first = this.request(0, resolve); });
      if (anchor) {
        for (const body of this.bodies.slice(1)) body.reference_speech_id = anchor;
      }
      await Promise.all([first, ...this.bodies.slice(1).map((_, index) => this.request(index + 1))]);
    } else {
      await Promise.all(this.bodies.map((_, index) => this.request(index)));
    }
    this.finishedAt = performance.now() - this.started;
    this.player.close();
    this.render();
    this.finish();
    // Keep the player's clock and bar moving while it plays.
    clearInterval(this.timer);
    this.timer = setInterval(() => {
      if (tts.run !== this) clearInterval(this.timer);
      else renderPlayer();
    }, 100);
  }

  /** Stop generating; what arrived stays playable. */
  stop() {
    this.controller.abort();
  }

  /** Stop everything: a new run replaces this one. */
  dispose() {
    this.controller.abort();
    clearInterval(this.timer);
    this.player.dispose();
  }

  finish() {
    const failed = this.segments.filter((segment) => segment.state === "failed");
    const stopped = this.segments.some((segment) => segment.state === "stopped");
    const rate = (this.segments.find((segment) => segment.samples) || {}).rate || DEFAULT_PCM_RATE;
    const chunks = [];
    this.segments.forEach((segment, index) => {
      if (index > 0 && segment.samples && chunks.length) chunks.push(new Int16Array(Math.round((this.chunked ? CHUNK_GAP : PARAGRAPH_GAP) * rate)));
      chunks.push(...segment.chunks);
    });
    if (this.audioSeconds > 0) {
      // The download; playback stays on the one player.
      const blob = wavBlob(chunks, rate);
      if (tts.url) URL.revokeObjectURL(tts.url);
      tts.url = URL.createObjectURL(blob);
      tts.lastWav = blob;
      const link = $("#tts-download");
      link.href = tts.url;
      link.download = `splash-omni-speech-${timestamp()}.wav`;
      link.hidden = false;
      $("#asr-use-tts").hidden = false;
      $("#live-use-tts").hidden = false;
    }
    if (failed.length) {
      ttsStatus(failed.length === this.segments.length
        ? `Failed: ${failed[0].error}`
        : `${failed.length} of ${this.segments.length} paragraphs failed: ${failed[0].error}`, true);
    } else if (stopped) {
      ttsStatus("Stopped.");
    } else {
      const reasons = this.segments.map((segment) => segment.finishReason).filter(Boolean);
      const reason = reasons.length && reasons.some((value) => value !== "stop") ? ` Finish reason: ${reasons.join(", ")}.` : "";
      ttsStatus(`Done.${reason}`);
    }
    setTtsRunning(false);
  }

  render() {
    if (tts.run !== this) return;
    renderPlayer();
    const root = $("#tts-metrics");
    const elapsed = (this.finishedAt !== null ? this.finishedAt : performance.now() - this.started);
    const audio = this.audioSeconds;
    const server = this.segments[0].serverFirstAudio;
    setMetric(root, "first", formatMs(this.firstAudio),
      this.paragraphMode
        ? (this.firstAudioSegment !== null ? `paragraph ${this.firstAudioSegment + 1}` : "")
        : (server && endpoint.viaProxy ? `server ${formatMs(Number(server))}` : ($("#tts-stream").checked ? "" : "whole file, not streamed")));
    setMetric(root, "total", formatMs(elapsed),
      this.paragraphMode ? `${this.bodies.length} requests` : (this.finishedAt === null ? "running" : "1 request"));
    setMetric(root, "audio", formatSeconds(audio),
      audio ? `${this.segments[0].rate / 1000} kHz${this.segments[0].frames ? ` · ${this.segments[0].frames} frames` : ""}` : "");
    if (audio > 0 && elapsed > 0) {
      setMetric(root, "speed", `${(audio / (elapsed / 1000)).toFixed(1)}×`, `RTF ${((elapsed / 1000) / audio).toFixed(3)}`);
    } else {
      setMetric(root, "speed", "–", "audio s per s");
    }
    const wrap = $("#tts-segments-wrap");
    wrap.hidden = !this.paragraphMode;
    if (!this.paragraphMode) return;
    const rows = this.segments.map((segment, index) => element("tr", { class: this.player.cursor === index && this.finishedAt === null ? "playing" : "" },
      element("td", { text: index + 1 }),
      element("td", { class: "text", title: segment.text, text: segment.text }),
      element("td", { text: formatMs(segment.firstAudio) }),
      element("td", { text: segment.done !== undefined ? formatMs(segment.done) : "…" }),
      element("td", { text: segment.samples ? formatSeconds(segment.samples / segment.rate) : "–" }),
      element("td", { text: segment.error || segment.state, title: segment.error || "" })));
    $("#tts-segments").replaceChildren(...rows);
  }
}

function setTtsRunning(running) {
  $("#tts-go").disabled = running;
  $("#tts-stop").disabled = !running;
}

/** The request bodies of a run: one per paragraph in paragraph mode, one
 * per chunk of a long text when the server continues speech (each chunk
 * after the first continues it, which keeps the level a long single
 * request loses), else one; every body carries the same voice,
 * instructions and seed. */
function speechBodies(text) {
  const paragraphMode = paragraphModeOn();
  const texts = paragraphMode ? SplashLogic.paragraphs(text) : chunkedText(text);
  let seed = numberField("#tts-seed", true);
  if (seed !== null && (seed < 0 || seed > Number.MAX_SAFE_INTEGER)) {
    throw new Error("Seed must be a whole number from 0 to 2^53.");
  }
  if (seed === null && texts.length > 1 && $("#tts-shared-seed").checked) {
    seed = Math.floor(Math.random() * 2 ** 32);
  }
  const parameters = speechParameters();
  const { instructions, voice } = voiceInputs();
  const bodies = texts.map((input) => SplashLogic.speechBody({
    input, parameters, controls: endpoint.controls, instructions, voice, seed, model: endpoint.model,
  }));
  return { bodies, seed };
}

/** How much faster than the model speaks the page plays its audio. */
function playbackSpeed() {
  const value = Number($("#tts-speed").value);
  return isFinite(value) && value > 0 ? Math.min(3, Math.max(0.5, value)) : 1.5;
}

/** The chunks a text is spoken in outside paragraph mode: several only for
 * a long text, on a server that continues speech, with the option on. */
function chunkedText(text) {
  if (!endpoint.controls.references || !$("#tts-chunks").checked) return [text];
  return SplashLogic.speechChunks(text);
}

function generateSpeech() {
  readEndpointFields();
  const text = $("#tts-text").value.trim();
  if (!text) {
    ttsStatus("Type some text first.", true);
    return;
  }
  let plan;
  try {
    plan = speechBodies(text);
  } catch (error) {
    ttsStatus(error.message, true);
    if (/system prompt/.test(error.message)) $("#tts-instructions").focus();
    return;
  }
  const { bodies, seed } = plan;
  $("#tts-seed-used").textContent = seed === null ? "" : `seed ${seed}${bodies.length > 1 ? " (every paragraph)" : ""}`;
  saveTtsSettings();
  const play = $("#tts-autoplay").checked;
  // Created or resumed in the click, so that Safari plays it.
  getAudioContext();
  if (tts.run) tts.run.dispose();
  $("#tts-download").hidden = true;
  setTtsRunning(true);
  tts.run = new SpeechRun(bodies, {
    play,
    speed: playbackSpeed(),
    anchored: endpoint.controls.references,
    chunked: !paragraphModeOn(),
  });
  tts.player = tts.run.player;
  tts.run.start();
}

/* ------------------------------------------------------------------ */
/* Transcription                                                       */

const asr = { file: null, name: "", duration: null, url: null, recording: null, controller: null };

async function setAsrInput(blob, name) {
  asr.file = blob;
  asr.name = name;
  asr.duration = null;
  if (asr.url) URL.revokeObjectURL(asr.url);
  asr.url = URL.createObjectURL(blob);
  const preview = $("#asr-preview");
  preview.src = asr.url;
  preview.hidden = false;
  $("#asr-file-info").hidden = false;
  $("#asr-file-name").textContent = name;
  $("#asr-file-meta").textContent = formatBytes(blob.size);
  $("#asr-go").disabled = false;
  try {
    const decoded = await decodeAudio(await blob.arrayBuffer());
    if (asr.file === blob) {
      asr.duration = decoded.duration;
      $("#asr-file-meta").textContent = `${formatBytes(blob.size)} · ${formatSeconds(decoded.duration)}`;
    }
  } catch (error) {
    /* The browser cannot decode it; the server may. */
  }
}

async function toggleRecording() {
  if (asr.recording) {
    stopRecording();
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    asrStatus("This browser does not offer the microphone to this page.", true);
    return;
  }
  const context = getAudioContext();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false } });
  } catch (error) {
    asrStatus(`Microphone unavailable: ${error.message}`, true);
    return;
  }
  const source = context.createMediaStreamSource(stream);
  const processor = context.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  processor.onaudioprocess = (event) => chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  source.connect(processor);
  processor.connect(context.destination);
  const started = performance.now();
  const button = $("#asr-record");
  button.classList.add("recording");
  const tick = () => { $("#asr-record-label").textContent = `Stop · ${formatSeconds((performance.now() - started) / 1000)}`; };
  tick();
  asr.recording = { stream, source, processor, chunks, rate: context.sampleRate, timer: setInterval(tick, 250) };
  asrStatus("Recording… press Stop when done.");
}

function stopRecording() {
  const recording = asr.recording;
  asr.recording = null;
  clearInterval(recording.timer);
  recording.processor.disconnect();
  recording.source.disconnect();
  for (const track of recording.stream.getTracks()) track.stop();
  $("#asr-record").classList.remove("recording");
  $("#asr-record-label").textContent = "Record";
  const length = recording.chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const samples = new Float32Array(length);
  let offset = 0;
  for (const chunk of recording.chunks) {
    samples.set(chunk, offset);
    offset += chunk.length;
  }
  if (!length) {
    asrStatus("Nothing was recorded.", true);
    return;
  }
  setAsrInput(wavBlob([int16FromFloat(samples)], recording.rate), `recording-${timestamp()}.wav`);
  asrStatus(`Recorded ${formatSeconds(length / recording.rate)}.`);
}

function asrStatus(message, error = false) {
  const status = $("#asr-status");
  status.textContent = message;
  status.classList.toggle("error-text", error);
}

async function transcribe() {
  readEndpointFields();
  if (!asr.file) return;
  const language = $("#asr-language").value;
  const format = $("#asr-format").value;
  store.set("asr", { language, format });
  const form = new FormData();
  form.append("file", asr.file, asr.name);
  if (endpoint.model) form.append("model", endpoint.model);
  if (language) form.append("language", language);
  form.append("response_format", format);
  const root = $("#asr-metrics");
  for (const name of ["total", "audio", "speed", "language"]) setMetric(root, name, "–", "");
  $("#asr-text").value = "";
  $("#asr-copy").disabled = true;
  $("#asr-raw-details").hidden = true;
  asr.controller = new AbortController();
  $("#asr-go").disabled = true;
  $("#asr-stop").disabled = false;
  asrStatus("Transcribing…");
  const started = performance.now();
  const timer = setInterval(() => setMetric(root, "total", formatMs(performance.now() - started), "running"), 200);
  try {
    const response = await api("/v1/audio/transcriptions", { method: "POST", body: form, signal: asr.controller.signal });
    const body = await response.text();
    const elapsed = performance.now() - started;
    clearInterval(timer);
    if (!response.ok) {
      const fake = new Response(body, { status: response.status, statusText: response.statusText });
      throw new Error(await errorMessage(fake));
    }
    let text = body;
    let detected = null;
    if (format !== "text") {
      try {
        const data = JSON.parse(body);
        text = typeof data.text === "string" ? data.text : body;
        detected = data.language || null;
      } catch (error) {
        /* Shown as it came. */
      }
    }
    $("#asr-text").value = text.trim();
    asrResize();
    $("#asr-copy").disabled = !text.trim();
    $("#asr-raw").textContent = format === "text" ? body : prettyJson(body);
    $("#asr-raw-details").hidden = false;
    const headerSeconds = Number(response.headers.get("X-Splash-Audio-Seconds"));
    const seconds = headerSeconds > 0 ? headerSeconds : asr.duration;
    const segments = response.headers.get("X-Splash-Segments");
    const tokens = response.headers.get("X-Splash-Output-Tokens");
    setMetric(root, "total", formatMs(elapsed), "end to end");
    setMetric(root, "audio", formatSeconds(seconds),
      [segments && segments !== "1" ? `${segments} segments` : "", tokens ? `${tokens} tokens` : ""].filter(Boolean).join(" · "));
    if (seconds) setMetric(root, "speed", `${(seconds / (elapsed / 1000)).toFixed(1)}×`, `RTF ${((elapsed / 1000) / seconds).toFixed(3)}`);
    const forced = $("#asr-language").selectedOptions[0];
    setMetric(root, "language", detected || (language ? forced.textContent : "auto"), detected ? "detected" : language ? "forced" : "");
    asrStatus("Done.");
  } catch (error) {
    clearInterval(timer);
    asrStatus(error.name === "AbortError" ? "Stopped." : `Failed: ${error.message}`, error.name !== "AbortError");
  } finally {
    $("#asr-go").disabled = !asr.file;
    $("#asr-stop").disabled = true;
  }
}

function prettyJson(text) {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch (error) {
    return text;
  }
}

let asrResize = () => {};

/* ------------------------------------------------------------------ */
/* Live transcription                                                  */

const LIVE_RATE = 16000;

/** Continuous transcription in the page: audio from the microphone (or a
 * file fed at its own pace), resampled to 16 kHz, cut into utterances by
 * SplashLogic.Segmenter, and each utterance sent as an ordinary
 * /v1/audio/transcriptions request with a 16 kHz mono WAV. While an
 * utterance grows, the audio so far is sent for a partial line, one request
 * at a time; a final line replaces it. */
class LiveSession {
  constructor({ source, file, play, language, options, partials, cleanup }) {
    this.source = source;
    this.file = file;
    this.playFile = play;
    this.language = language;
    this.options = options;
    this.partials = partials;
    this.cleanup = cleanup;
    this.segmenter = new SplashLogic.Segmenter({ rate: LIVE_RATE, ...options, partialMs: partials ? options.partialMs : 0 });
    this.resampler = null;
    this.running = false;
    this.started = 0;
    this.current = null;
    this.partialBusy = false;
    this.pending = new Set();
    this.controller = new AbortController();
    this.peakDb = -Infinity;
  }

  async start() {
    const context = getAudioContext();
    this.resampler = new SplashLogic.Resampler(context.sampleRate, LIVE_RATE);
    if (this.source === "file") {
      const decoded = await decodeAudio(await this.file.arrayBuffer());
      this.feedFile(context, decoded);
    } else {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("This browser does not offer the microphone to this page.");
      }
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: {
        channelCount: 1,
        echoCancellation: this.cleanup,
        noiseSuppression: this.cleanup,
        autoGainControl: this.cleanup,
      } });
      this.input = context.createMediaStreamSource(this.stream);
      // A ScriptProcessor, as the recorder uses: every browser has it,
      // Safari included, without a separate worklet file.
      this.processor = context.createScriptProcessor(4096, 1, 1);
      this.processor.onaudioprocess = (event) => this.feed(event.inputBuffer.getChannelData(0));
      this.input.connect(this.processor);
      this.processor.connect(context.destination);
    }
    this.running = true;
    this.started = performance.now();
  }

  /** A decoded file, fed as fast as it plays (and played, if asked). */
  feedFile(context, decoded) {
    const channels = [];
    for (let channel = 0; channel < decoded.numberOfChannels; channel++) channels.push(decoded.getChannelData(channel));
    const mono = new Float32Array(decoded.length);
    for (const data of channels) for (let i = 0; i < data.length; i++) mono[i] += data[i] / channels.length;
    const begin = context.currentTime + 0.05;
    if (this.playFile) {
      this.player = context.createBufferSource();
      this.player.buffer = decoded;
      this.player.connect(context.destination);
      this.player.start(begin);
    }
    let fed = 0;
    this.fileTimer = setInterval(() => {
      const due = Math.min(mono.length, Math.floor(Math.max(0, context.currentTime - begin) * decoded.sampleRate));
      if (due > fed) {
        this.feed(mono.subarray(fed, due));
        fed = due;
      }
      if (fed >= mono.length) {
        clearInterval(this.fileTimer);
        this.fileTimer = null;
        // The file's end ends the session.
        stopLive("The file has been transcribed.");
      }
    }, 50);
  }

  feed(samples) {
    if (!this.running) return;
    const events = this.segmenter.push(this.resampler.push(samples));
    this.peakDb = Math.max(this.peakDb, this.segmenter.levelDb);
    for (const event of events) this.handle(event);
  }

  handle(event) {
    if (event.type === "start") {
      this.current = { id: event.id, start: event.at, line: null };
    } else if (event.type === "partial") {
      if (!this.partialBusy && this.current && this.current.id === event.id) this.sendPartial(event);
    } else if (event.type === "drop") {
      if (this.current && this.current.id === event.id) {
        if (this.current.line) this.current.line.remove();
        this.current = null;
      }
    } else if (event.type === "final") {
      let line = this.current && this.current.id === event.id ? this.current.line : null;
      this.current = null;
      if (!line) line = liveLine();
      this.sendFinal(event, line);
    }
  }

  async transcribe(samples) {
    const form = new FormData();
    const wav = new Blob([SplashLogic.wavBytes([SplashLogic.int16FromFloat(samples)], LIVE_RATE)], { type: "audio/wav" });
    form.append("file", wav, "utterance.wav");
    if (endpoint.model) form.append("model", endpoint.model);
    if (this.language) form.append("language", this.language);
    form.append("response_format", "json");
    const response = await api("/v1/audio/transcriptions", { method: "POST", body: form, signal: this.controller.signal });
    const body = await response.text();
    if (!response.ok) {
      throw new Error(await errorMessage(new Response(body, { status: response.status, statusText: response.statusText })));
    }
    try {
      const data = JSON.parse(body);
      return typeof data.text === "string" ? data.text.trim() : body.trim();
    } catch (error) {
      return body.trim();
    }
  }

  async sendPartial(event) {
    const utterance = this.current;
    this.partialBusy = true;
    try {
      const text = await this.transcribe(event.samples);
      // A final line may have replaced it meanwhile.
      if (this.current === utterance && text) {
        if (!utterance.line) utterance.line = liveLine();
        fillLiveLine(utterance.line, { text, partial: true, start: utterance.start, seconds: event.seconds });
      }
    } catch (error) {
      /* A partial is provisional; the final line reports errors. */
    } finally {
      this.partialBusy = false;
    }
  }

  async sendFinal(event, line) {
    const cut = performance.now();
    fillLiveLine(line, { text: "", partial: true, pending: true, start: event.start, seconds: event.seconds });
    const request = this.transcribe(event.samples);
    this.pending.add(request);
    try {
      const text = await request;
      const latency = performance.now() - cut;
      fillLiveLine(line, { text, start: event.start, seconds: event.seconds, latency, reason: event.reason });
      live.latencies.push(latency);
    } catch (error) {
      fillLiveLine(line, { error: error.name === "AbortError" ? "stopped" : error.message, start: event.start, seconds: event.seconds });
    } finally {
      this.pending.delete(request);
      renderLiveMetrics();
    }
  }

  /** Stop listening; the utterance in progress is sent as a final line. */
  stop() {
    if (!this.running) return;
    this.running = false;
    if (this.fileTimer) clearInterval(this.fileTimer);
    if (this.player) {
      try { this.player.stop(); } catch (error) { /* Ended. */ }
    }
    if (this.processor) {
      this.processor.onaudioprocess = null;
      this.processor.disconnect();
    }
    if (this.input) this.input.disconnect();
    if (this.stream) for (const track of this.stream.getTracks()) track.stop();
    for (const event of this.segmenter.flush()) this.handle(event);
  }

  /** Stop and abandon the requests in flight. */
  abort() {
    this.stop();
    this.controller.abort();
  }
}

const live = { session: null, file: null, fileName: "", latencies: [], timer: null };

function liveLine() {
  $("#live-empty").hidden = true;
  const line = element("div", { class: "live-line partial" },
    element("span", { class: "live-time" }),
    element("span", { class: "live-text" }),
    element("span", { class: "live-meta" }));
  const box = $("#live-transcript");
  const nearBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 40;
  box.append(line);
  if (nearBottom) box.scrollTop = box.scrollHeight;
  updateLiveButtons();
  return line;
}

function fillLiveLine(line, { text = "", partial = false, pending = false, error = null, start, seconds, latency, reason }) {
  line.className = "live-line" + (partial ? " partial" : "") + (error ? " failed" : "");
  line.dataset.final = partial || error ? "" : "1";
  $(".live-time", line).textContent = formatClock(start || 0).replace(/\.\d$/, "");
  const shown = error ? `(${error})` : text || (pending ? "…" : "(no speech)");
  $(".live-text", line).textContent = shown + (pending && text ? " …" : "");
  const meta = [];
  if (seconds !== undefined) meta.push(`${seconds.toFixed(1)} s`);
  if (latency !== undefined) meta.push(formatMs(latency));
  if (reason === "max") meta.push("cut at the limit");
  $(".live-meta", line).textContent = meta.join(" · ");
  $(".live-meta", line).title = latency !== undefined
    ? "Utterance length · time from the cut (after the pause) to its transcript"
    : "";
  const box = $("#live-transcript");
  if (box.scrollTop + box.clientHeight >= box.scrollHeight - 80) box.scrollTop = box.scrollHeight;
}

function liveTranscriptText() {
  return $$(".live-line", $("#live-transcript"))
    .filter((line) => line.dataset.final === "1")
    .map((line) => $(".live-text", line).textContent)
    .filter((text) => text && text !== "(no speech)")
    .join("\n");
}

function renderLiveMetrics() {
  const root = $("#live-metrics");
  const lines = $$(".live-line", $("#live-transcript")).filter((line) => line.dataset.final === "1").length;
  const session = live.session;
  setMetric(root, "count", lines ? String(lines) : "–", session && session.pending.size ? `${session.pending.size} in flight` : "");
  const latencies = live.latencies;
  if (latencies.length) {
    const sorted = [...latencies].sort((a, b) => a - b);
    const middle = sorted.length >> 1;
    const median = sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
    setMetric(root, "last", formatMs(latencies[latencies.length - 1]), "after the cut");
    setMetric(root, "median", formatMs(median), `${latencies.length} line${latencies.length === 1 ? "" : "s"}`);
  } else {
    setMetric(root, "last", "–", "");
    setMetric(root, "median", "–", "");
  }
  if (session && session.running) {
    setMetric(root, "time", formatSeconds((performance.now() - session.started) / 1000),
      session.segmenter.utterance ? "speech" : "silence");
  }
  updateLiveButtons();
}

function renderLiveLevel() {
  const session = live.session;
  if (!session || !session.running) return;
  const level = session.peakDb;
  session.peakDb = -Infinity;
  const threshold = session.segmenter.threshold;
  const scale = (db) => Math.min(100, Math.max(0, ((db + 80) / 80) * 100));
  if (isFinite(level)) $("#live-meter").style.width = `${scale(level)}%`;
  $("#live-meter").classList.toggle("voiced", level > threshold);
  $("#live-threshold").style.left = `${scale(threshold)}%`;
  $("#live-level").textContent = `Input level ${isFinite(level) ? Math.round(level) : "–"} dBFS · threshold ${Math.round(threshold)} dBFS` +
    (session.thresholdAuto ? " (auto)" : "");
}

function updateLiveButtons() {
  const any = $$(".live-line", $("#live-transcript")).length > 0;
  $("#live-copy").disabled = !liveTranscriptText();
  $("#live-clear").disabled = !any;
}

function liveStatus(message, error = false) {
  const status = $("#live-status");
  status.textContent = message;
  status.classList.toggle("error-text", error);
}

function liveOptions() {
  const read = (id, fallback, low, high) => {
    const value = Number($(id).value);
    return $(id).value.trim() === "" || !isFinite(value) ? fallback : Math.min(Math.max(value, low), high);
  };
  const threshold = $("#live-threshold-db").value.trim();
  return {
    silenceMs: read("#live-silence", 700, 100, 5000),
    maxSeconds: read("#live-max", 15, 2, 60),
    partialMs: read("#live-partial-ms", 1000, 300, 5000),
    thresholdDb: threshold === "" || !isFinite(Number(threshold)) ? null : Math.min(Math.max(Number(threshold), -90), 0),
  };
}

function saveLiveSettings() {
  store.set("live", {
    language: $("#live-language").value,
    silence: $("#live-silence").value,
    max: $("#live-max").value,
    threshold: $("#live-threshold-db").value,
    partialMs: $("#live-partial-ms").value,
    partials: $("#live-partials").checked,
    cleanup: $("#live-cleanup").checked,
    play: $("#live-play-file").checked,
    source: liveSource(),
  });
}

function restoreLiveSettings() {
  const saved = store.get("live", {});
  for (const [id, key] of [["#live-silence", "silence"], ["#live-max", "max"], ["#live-threshold-db", "threshold"], ["#live-partial-ms", "partialMs"]]) {
    if (saved[key] !== undefined) $(id).value = saved[key];
  }
  for (const [id, key] of [["#live-partials", "partials"], ["#live-cleanup", "cleanup"], ["#live-play-file", "play"]]) {
    if (saved[key] !== undefined) $(id).checked = saved[key];
  }
  if (saved.source === "file") $('input[name="live-source"][value="file"]').checked = true;
  updateLiveSource();
}

function liveSource() {
  return ($('input[name="live-source"]:checked') || {}).value || "mic";
}

function updateLiveSource() {
  $("#live-file-row").hidden = liveSource() !== "file";
}

function setLiveFile(blob, name) {
  live.file = blob;
  live.fileName = name;
  $("#live-file-name").textContent = `${name} · ${formatBytes(blob.size)}`;
}

async function startLive() {
  readEndpointFields();
  saveLiveSettings();
  const source = liveSource();
  if (source === "file" && !live.file) {
    liveStatus("Choose an audio file first.", true);
    return;
  }
  const options = liveOptions();
  const session = new LiveSession({
    source,
    file: live.file,
    play: $("#live-play-file").checked,
    language: $("#live-language").value,
    options,
    partials: $("#live-partials").checked,
    cleanup: $("#live-cleanup").checked,
  });
  session.thresholdAuto = options.thresholdDb === null;
  live.session = session;
  setLiveRunning(true);
  liveStatus(source === "file" ? `Feeding ${live.fileName} in real time…` : "Starting the microphone (allow it if the browser asks)…");
  try {
    await session.start();
  } catch (error) {
    live.session = null;
    setLiveRunning(false);
    liveStatus(`Cannot listen: ${error.message}`, true);
    return;
  }
  if (live.session !== session) return;
  liveStatus(source === "file" ? `Transcribing ${live.fileName} as it plays…` : "Listening… speak, and pause to end a line.");
  clearInterval(live.timer);
  live.timer = setInterval(() => {
    renderLiveLevel();
    renderLiveMetrics();
  }, 100);
}

/** Stop listening; the lines in flight still arrive. */
function stopLive(message = "Stopped.") {
  const session = live.session;
  if (!session || !session.running) return;
  session.stop();
  clearInterval(live.timer);
  live.timer = null;
  setLiveRunning(false);
  $("#live-meter").style.width = "0";
  liveStatus(session.pending.size ? `${message} Waiting for the last line…` : message);
  Promise.allSettled([...session.pending]).then(() => {
    if (live.session === session) liveStatus(message);
    renderLiveMetrics();
  });
}

function setLiveRunning(running) {
  const button = $("#live-toggle");
  button.classList.toggle("recording", running);
  button.classList.toggle("primary", !running);
  $("#live-toggle-label").textContent = running ? "Stop" : "Start listening";
  for (const input of $$('input[name="live-source"]')) input.disabled = running;
  $("#live-language").disabled = running;
}

function clearLive() {
  $$(".live-line", $("#live-transcript")).forEach((line) => line.remove());
  $("#live-empty").hidden = false;
  live.latencies = [];
  if (live.session && live.session.current) live.session.current.line = null;
  for (const name of ["count", "last", "median"]) setMetric($("#live-metrics"), name, "–", "");
  updateLiveButtons();
}

/* ------------------------------------------------------------------ */
/* Wiring                                                              */

/** Whether voice chat's servers run, which this page's setup waits for. */
async function checkVoice() {
  try {
    // Log sequence numbers past any line: the phases without the logs.
    const big = Number.MAX_SAFE_INTEGER;
    const state = await backend(`/api/voice/state?asr=${big}&tts=${big}&llm=${big}`);
    app.voiceRunning = ["asr", "tts", "llm"].some((role) =>
      [...RUNNING_PHASES, "stopping"].includes(state[role].phase));
  } catch (error) {
    app.voiceRunning = false;
  }
}

async function poll() {
  try {
    if (app.view === "setup") {
      await checkVoice();
      // Voice chat started (here or in another tab): follow it there.
      if (app.voiceRunning) {
        location.replace("/voice");
        return;
      }
    }
    const state = await backend(`/api/state?since=${app.logSeq}`);
    app.backendDown = false;
    receiveState(state);
  } catch (error) {
    // The page's backend went away (Ctrl+C), and the server it started with
    // it: silence the page once, and keep trying.
    if (!app.backendDown && app.session && app.session.phase === "ready") haltAudio();
    app.backendDown = true;
  }
  const interval = app.view === "progress" ? 600 : 2500;
  setTimeout(poll, interval);
}

function bind() {
  const form = setup.form();
  form.addEventListener("submit", startServer);
  form.addEventListener("change", (event) => {
    if (event.target.name === "variant") setup.variant = event.target.value;
    if (event.target.name === "task" || event.target.name === "splash") {
      setup.renderSections();
      setup.renderSplash();
      if (setup.task() !== "voice") setup.checkPort();
      setup.renderBusy();
    }
    if (event.target.name === "task" || event.target.name === "variant") setup.renderModels();
    if (["asr_model", "tts_model", "voice_speculative"].includes(event.target.name)) setup.followVoiceMemory();
    setup.save();
    setup.renderSummary();
    setup.renderBusy();
  });
  form.addEventListener("input", (event) => {
    setup.renderSummary();
    if (event.target.name === "port") {
      setup.portChosen = true;
      setup.checkPort();
    }
  });
  $("#btn-connect").addEventListener("click", () => {
    $("#connect-panel").hidden = false;
    $("#connect-url").value = $("#port-warning").hidden && store.get("endpoint", {}).base
      ? store.get("endpoint", {}).base
      : `http://127.0.0.1:${form.port.value || 8000}`;
    $("#connect-url").focus();
  });
  $("#btn-connect-cancel").addEventListener("click", () => { $("#connect-panel").hidden = true; });
  $("#btn-connect-go").addEventListener("click", () => {
    const base = $("#connect-url").value.trim();
    if (!/^https?:\/\/[^/]+/.test(base)) return;
    $("#connect-panel").hidden = true;
    enterRun({ base, model: "", session: false });
  });
  $("#btn-cancel").addEventListener("click", cancelStart);
  $("#btn-back").addEventListener("click", backToSetup);
  $("#btn-use-running").addEventListener("click", () => {
    const selection = app.session.selection;
    enterRun({ base: `http://127.0.0.1:${selection.port}`, model: "", task: selection.task, session: false });
  });
  $("#btn-stop").addEventListener("click", () => stopServer(true));

  for (const id of ["#ep-base", "#ep-model", "#ep-key", "#ep-proxy"]) {
    $(id).addEventListener("change", readEndpointFields);
  }
  $("#ep-check").addEventListener("click", checkEndpoint);
  for (const tab of $$('[role="tab"]')) tab.addEventListener("click", () => selectTab(tab.dataset.tab));

  ttsResize = autogrow($("#tts-text"));
  asrResize = autogrow($("#asr-text"));
  $("#tts-text").addEventListener("input", updateTtsCounts);
  $("#tts-text").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && !$("#tts-go").disabled) generateSpeech();
  });
  $("#tts-paragraphs").addEventListener("change", () => { updateTtsCounts(); saveTtsSettings(); });
  $("#tts-chunks").addEventListener("change", () => { updateTtsCounts(); saveTtsSettings(); });
  $("#tts-go").addEventListener("click", generateSpeech);
  $("#tts-stop").addEventListener("click", () => { if (tts.run) tts.run.stop(); });
  $("#tts-params").addEventListener("change", saveTtsSettings);
  $("#tts-language").addEventListener("change", saveTtsSettings);
  $("#tts-speed").addEventListener("change", saveTtsSettings);
  $("#tts-instructions").addEventListener("input", updatePromptCount);
  $("#tts-instructions").addEventListener("change", saveTtsSettings);
  $("#tts-speaker").addEventListener("change", saveTtsSettings);
  $("#tts-voice").addEventListener("change", saveTtsSettings);
  $("#tts-play").addEventListener("click", togglePlayback);
  const timeline = $("#tts-timeline");
  timeline.addEventListener("click", seekFromPointer);
  timeline.addEventListener("keydown", seekByKey);

  const drop = $("#asr-drop");
  const fileInput = $("#asr-file");
  $("#asr-choose").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) setAsrInput(fileInput.files[0], fileInput.files[0].name);
    fileInput.value = "";
  });
  drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("over");
    const file = event.dataTransfer.files[0];
    if (file) setAsrInput(file, file.name);
  });
  $("#asr-record").addEventListener("click", toggleRecording);
  $("#asr-use-tts").addEventListener("click", () => {
    if (tts.lastWav) setAsrInput(tts.lastWav, "speech-output.wav");
  });
  $("#asr-go").addEventListener("click", transcribe);
  $("#asr-stop").addEventListener("click", () => { if (asr.controller) asr.controller.abort(); });
  $("#asr-copy").addEventListener("click", () => {
    const text = $("#asr-text").value;
    if (navigator.clipboard) navigator.clipboard.writeText(text).catch(() => {});
    else { $("#asr-text").select(); document.execCommand("copy"); }
  });

  $("#live-toggle").addEventListener("click", () => {
    // The audio context starts in the click, as Safari requires.
    getAudioContext();
    if (live.session && live.session.running) stopLive();
    else startLive();
  });
  for (const input of $$('input[name="live-source"]')) {
    input.addEventListener("change", () => { updateLiveSource(); saveLiveSettings(); });
  }
  const liveFile = $("#live-file");
  $("#live-choose").addEventListener("click", () => liveFile.click());
  liveFile.addEventListener("change", () => {
    if (liveFile.files[0]) setLiveFile(liveFile.files[0], liveFile.files[0].name);
    liveFile.value = "";
  });
  $("#live-use-tts").addEventListener("click", () => {
    if (tts.lastWav) setLiveFile(tts.lastWav, "speech-output.wav");
  });
  $("#live-params").addEventListener("change", saveLiveSettings);
  $("#live-language").addEventListener("change", saveLiveSettings);
  $("#live-clear").addEventListener("click", clearLive);
  $("#live-copy").addEventListener("click", () => {
    const text = liveTranscriptText();
    if (navigator.clipboard) navigator.clipboard.writeText(text).catch(() => {});
  });
}

async function init() {
  bind();
  setup.restore();
  restoreTtsSettings();
  restoreLiveSettings();
  const savedEndpoint = store.get("endpoint", {});
  if (savedEndpoint.base) endpoint.base = savedEndpoint.base;
  if (savedEndpoint.viaProxy === false) endpoint.viaProxy = false;
  const asrSaved = store.get("asr", {});
  if (asrSaved.format) $("#asr-format").value = asrSaved.format;
  fillLanguages();
  updateTtsCounts();
  try {
    app.catalog = await backend("/api/catalog");
  } catch (error) {
    document.body.prepend(element("p", { class: "error", text: `The page's backend is not answering: ${error.message}` }));
    return;
  }
  await checkVoice();
  if (app.voiceRunning) {
    // One task at a time: voice chat runs, so show it.
    location.replace("/voice");
    return;
  }
  setup.render();
  loadVoiceCatalog();
  let state = null;
  try {
    state = await backend("/api/state?since=0");
  } catch (error) {
    /* Shown as idle. */
  }
  app.session = state;
  if (state) appendLog(state.log || [], state.log_seq);
  const phase = state ? state.phase : "idle";
  if (phase === "ready") {
    enterRun({ base: state.base_url, model: state.selection.model, task: state.selection.task, session: true });
  } else if (["preparing", "loading", "stopping", "failed"].includes(phase)) {
    showView("progress");
    renderProgress();
  } else if (location.hash === "#run" && savedEndpoint.base) {
    enterRun({ base: savedEndpoint.base, model: savedEndpoint.model || "", task: store.get("tab", null), session: false });
  } else {
    showView("setup");
  }
  setTimeout(poll, 600);
}

init();
