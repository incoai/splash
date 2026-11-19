/* splash-omni voice chat: speech recognition → language model → speech.
 *
 * The page's backend (gui/app.py, gui/voice.py) only starts and stops the
 * three servers. The conversation runs here, as API calls through the
 * backend's transparent proxy: an utterance the microphone's voice
 * activity detection cuts goes to /v1/audio/transcriptions, the transcript
 * to Splash's /v1/chat/completions (streamed), and each sentence of the
 * answer, as soon as it is complete, to /v1/audio/speech (streamed), whose
 * audio plays in order on one timeline while later sentences are still
 * being written and spoken.
 */
"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const LIVE_RATE = 16000;
const SPEECH_RATE = 24000;
// Silence between sentences, in seconds (each sentence's audio ends with
// a short pause of its own).
const SENTENCE_GAP = 0.06;
// Pieces of an answer the timeline holds at most.
const MAX_PIECES = 256;
// Voice activity detection: the pause that ends a turn, and the shortest speech kept.
const VAD = { silenceMs: 600, minSpeechMs: 250, preRollMs: 300, maxSeconds: 30 };
const ROLE_NAMES = { asr: "Speech recognition", llm: "Assistant", tts: "Speech" };
const DEFAULT_SYSTEM = {
  "": "You are a friendly voice assistant running entirely on this Mac. Your replies are spoken aloud by a speech model, so answer in plain conversational sentences: no lists, markdown, emoji or code. Keep answers short, one to three sentences, unless asked for more. Reply in the language the user speaks.",
  en: "You are a friendly voice assistant running entirely on this Mac. Your replies are spoken aloud by a speech model, so answer in plain conversational English sentences: no lists, markdown, emoji or code. Keep answers short, one to three sentences, unless asked for more.",
  zh: "你是一个运行在这台 Mac 上的友好语音助手。你的回答会被语音模型朗读出来，所以请用自然口语化的中文句子回答，不要使用列表、Markdown、表情符号或代码。除非用户要求，回答保持简短，一到三句话。",
};
const GREETING = { "": "Hi! I'm listening.", en: "Hi! I'm listening.", zh: "你好！我在听。" };
// The voice every answer continues (reference_speech_id): a sample of
// ordinary length spoken once, unheard, when the chat starts or its voice
// changes. A short line such as the greeting is often drawn out, and its
// continuations keep its pace; of up to VOICE_TRIES takes, the first at the
// target pace (or else the fastest) is kept.
const VOICE_SAMPLE = {
  "": "Hello, I'm your voice assistant. I run entirely on this Mac, and I'm happy to help with questions, ideas, or anything else you need today.",
  en: "Hello, I'm your voice assistant. I run entirely on this Mac, and I'm happy to help with questions, ideas, or anything else you need today.",
  zh: "你好，我是你的语音助手，完全运行在这台 Mac 上。无论是问题、想法还是日常小事，我都很乐意帮忙。",
};
const VOICE_TRIES = 3;
// Target pace: words a second (English), characters a second (Chinese).
const TARGET_PACE = { words: 2.4, characters: 4.2 };
const TTS_LANGUAGE = { en: "english", zh: "chinese" };
const DEFAULT_DESIGN = "A warm, friendly young female voice, clear and natural, speaking at a relaxed pace.";

/* ------------------------------------------------------------------ */
/* Helpers                                                             */

const store = {
  get(key, fallback) {
    try {
      const value = localStorage.getItem("splash-voice." + key);
      return value === null ? fallback : JSON.parse(value);
    } catch (error) {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem("splash-voice." + key, JSON.stringify(value));
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
    else node.setAttribute(name, value === true ? "" : value);
  }
  for (const child of children) {
    if (child !== null && child !== undefined) node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

function formatMs(ms) {
  if (ms === null || ms === undefined || !isFinite(ms)) return "–";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

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

/** An API request to one of the servers, through the page's proxy. */
function api(base, path, init = {}) {
  const headers = new Headers(init.headers || {});
  headers.set("X-Splash-Upstream", base);
  return fetch("/proxy" + path, { ...init, headers, cache: "no-store" });
}

async function errorMessage(response) {
  const text = await response.text().catch(() => "");
  try {
    const error = JSON.parse(text).error;
    if (typeof error === "string") return `${response.status}: ${error}`;
    if (error && error.message) return `${response.status}: ${error.message}`;
  } catch (error) {
    /* Not JSON. */
  }
  return `${response.status} ${response.statusText}${text ? ": " + text.slice(0, 200) : ""}`;
}

let audioContext = null;

/** The page's AudioContext, resumed when called from a user gesture. */
function getAudioContext() {
  if (!audioContext) {
    const Context = window.AudioContext || window.webkitAudioContext;
    audioContext = new Context();
  }
  if (audioContext.state === "suspended") audioContext.resume().catch(() => {});
  return audioContext;
}

/** Raw 16-bit little-endian PCM as it arrives, as int16 samples. */
class PcmStream {
  constructor() {
    this.carry = null;
  }

  push(bytes) {
    let data = bytes;
    if (this.carry !== null) {
      data = new Uint8Array(bytes.length + 1);
      data[0] = this.carry;
      data.set(bytes, 1);
      this.carry = null;
    }
    if (data.length % 2) {
      this.carry = data[data.length - 1];
      data = data.subarray(0, data.length - 1);
    }
    const copy = new Uint8Array(data);
    return new Int16Array(copy.buffer, 0, copy.length / 2);
  }
}

/* ------------------------------------------------------------------ */
/* Views and the servers' state                                        */

const app = {
  view: null,
  catalog: null,
  state: null,
  since: { asr: 0, tts: 0, llm: 0 },
  logs: { asr: [], tts: [], llm: [] },
  bases: null,
  wasReady: false,
};

function showView(name) {
  app.view = name;
  for (const view of ["progress", "chat"]) $(`#view-${view}`).hidden = view !== name;
}

const RUNNING = ["preparing", "loading", "ready"];

function allReady(state) {
  return state && ["asr", "tts", "llm"].every((role) => state[role].phase === "ready");
}

function anyRunning(state) {
  return state && ["asr", "tts", "llm"].some((role) => RUNNING.includes(state[role].phase));
}

function receiveState(state) {
  app.state = state;
  for (const role of ["asr", "tts", "llm"]) {
    const entry = state[role];
    if (entry.log_seq < app.since[role]) app.logs[role] = [];
    for (const line of entry.log || []) app.logs[role].push(line.text);
    app.logs[role] = app.logs[role].slice(-200);
    app.since[role] = entry.log_seq;
  }
  renderPills();
  const ready = allReady(state);
  if (ready && !app.wasReady) {
    app.bases = { asr: state.asr.base_url, tts: state.tts.base_url, llm: state.llm.base_url };
    enterChat();
  }
  if (!ready && app.wasReady) {
    // A server stopped or exited: the conversation cannot go on.
    haltConversation("A server stopped.");
  }
  app.wasReady = ready;
  if (!ready) {
    const roles = ["asr", "tts", "llm"];
    if (!roles.some((role) => [...RUNNING, "stopping", "failed"].includes(state[role].phase))) {
      // Nothing runs: voice chat starts from the start page, which also
      // shows a server of its own if one runs.
      location.replace("/");
      return;
    }
    showView("progress");
    renderProgress();
  }
  $("#btn-stop").hidden = !anyRunning(state);
}

/** One pill for the three servers: ready when all are, else starting or
 * stopped; each server's model in its tooltip. */
function renderPills() {
  const pill = $("#server-pill");
  const roles = ["asr", "llm", "tts"];
  const phases = roles.map((role) => (app.state && app.state[role] ? app.state[role].phase : "idle"));
  let state = "";
  let text = "Voice chat";
  if (phases.every((phase) => phase === "ready")) [state, text] = ["ready", "Voice chat · ready"];
  else if (phases.includes("failed")) [state, text] = ["down", "Voice chat · stopped"];
  else if (phases.some((phase) => RUNNING.includes(phase) || phase === "stopping")) [state, text] = ["busy", "Voice chat · starting"];
  pill.className = "pill" + (state ? " " + state : "");
  $("#server-pill-text").textContent = text;
  pill.title = roles.map((role) => {
    const entry = app.state && app.state[role];
    return `${ROLE_NAMES[role]}: ${entry && entry.selection ? entry.selection.model : "–"}`;
  }).join("\n");
}

function renderProgress() {
  const rows = $("#server-rows");
  // Keep the logs that were open across the rebuild.
  const open = new Set(Array.from(rows.querySelectorAll("details[open]"), (node) => node.dataset.role));
  const roles = ["asr", "llm", "tts"];
  const failed = roles.some((role) => app.state[role].phase === "failed");
  $("#progress-title").textContent = failed ? "Voice chat did not start" : "Starting voice chat";
  $("#progress-text").textContent = failed
    ? "One of the servers stopped. Its details show why."
    : "Models already on this Mac load without a download; the first start downloads what is missing.";
  $("#btn-cancel").textContent = failed ? "Back" : "Cancel";
  rows.replaceChildren();
  for (const role of roles) {
    const entry = app.state[role];
    const phase = entry.phase;
    const state = phase === "ready" ? "ready" : phase === "failed" ? "down" : RUNNING.includes(phase) ? "busy" : "";
    const status = phase === "ready" ? "Ready"
      : phase === "failed" ? (entry.error || "Did not start")
      : phase === "loading" ? "Loading the model"
      : phase === "stopping" ? "Stopping"
      : (entry.detail || "Preparing");
    const log = element("pre", { class: "log" });
    log.textContent = app.logs[role].slice(-40).join("\n");
    const model = entry.selection ? entry.selection.model : "";
    rows.append(element("div", { class: "server-row" },
      element("span", { class: `server-dot ${state}` }),
      element("div", { class: "server-main" },
        element("strong", { text: ROLE_NAMES[role] }),
        element("small", { text: model })),
      element("span", { class: "server-status" + (phase === "failed" ? " error-text" : ""), text: status }),
      element("details", { "data-role": role, open: open.has(role) || phase === "failed" },
        element("summary", { text: "Details" }),
        element("p", { class: "command" }, element("code", { text: entry.command || "" })),
        log)));
    log.scrollTop = log.scrollHeight;
  }
}

async function poll() {
  try {
    receiveState(await backend(`/api/voice/state?asr=${app.since.asr}&tts=${app.since.tts}&llm=${app.since.llm}`));
  } catch (error) {
    // The page's backend went away (Ctrl+C), and the servers with it.
    if (app.wasReady) haltConversation("The page's backend stopped.");
  }
  setTimeout(poll, app.view === "chat" ? 2500 : 700);
}

/* ------------------------------------------------------------------ */
/* Stopping                                                            */

async function stopServers() {
  if (anyRunning(app.state) && !confirm("End voice chat? Its three models are unloaded, and you can choose another task.")) return;
  haltConversation("Servers stopped.");
  try {
    receiveState(await backend("/api/voice/stop", {}));
  } catch (error) {
    alert(error.message);
  }
}

/* ------------------------------------------------------------------ */
/* The servers' models and voices                                      */

const models = { asr: "", llm: "", tts: "", ttsType: null, voices: [], references: false };

async function firstModel(role) {
  const response = await api(app.bases[role], "/v1/models");
  if (!response.ok) throw new Error(await errorMessage(response));
  const data = await response.json();
  const list = Array.isArray(data.data) ? data.data : [];
  return list[0] || {};
}

async function enterChat() {
  showView("chat");
  setStatus("Connecting to the servers…");
  try {
    const [asr, llm, tts] = await Promise.all([firstModel("asr"), firstModel("llm"), firstModel("tts")]);
    models.asr = asr.id || "";
    models.llm = llm.id || "";
    models.tts = tts.id || "";
    models.ttsType = tts.tts_model_type || null;
    models.references = !!tts.references;
    models.voices = Array.isArray(tts.voices) ? tts.voices : [];
    if (!models.voices.length && models.ttsType === "custom_voice") {
      const response = await api(app.bases.tts, "/v1/audio/voices");
      if (response.ok) {
        const answer = await response.json();
        const items = Array.isArray(answer) ? answer : answer.voices || answer.data || [];
        models.voices = items.map((voice) => (typeof voice === "string" ? voice : voice.id || voice.name)).filter(Boolean);
      }
    }
  } catch (error) {
    setStatus(`Could not read the servers' models: ${error.message}`, true);
  }
  fillVoiceOptions();
  resetVoice();
  setStatus("Ready. Press Start talking, or type a message.");
}

function fillVoiceOptions() {
  const saved = store.get("chat", {});
  const select = $("#opt-voice");
  const voices = models.voices.map((voice) => (typeof voice === "string" ? voice : voice.id || voice.name));
  $("#voice-row").hidden = !voices.length;
  select.replaceChildren(...voices.map((voice) => element("option", { value: voice, selected: voice === (saved.voice || "vivian") },
    voice.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase()))));
  const design = models.ttsType === "voice_design";
  $("#style-row").hidden = models.ttsType === "base";
  $("#style-label").textContent = design ? "Description (required)" : "Style (optional)";
  $("#opt-style").value = saved.style ?? (design ? DEFAULT_DESIGN : "");
  $("#opt-language").value = saved.language || "";
  $("#opt-speed").value = saved.speed || "1.5";
  $("#opt-system").value = saved.system || DEFAULT_SYSTEM[$("#opt-language").value];
  $("#opt-barge").checked = !!saved.barge;
  $("#opt-greet").checked = saved.greet !== false;
  $("#opt-anchor").checked = saved.anchor !== false;
  $("#opt-anchor").closest("label").hidden = !models.references;
}

function saveChatOptions() {
  store.set("chat", {
    voice: $("#opt-voice").value,
    style: $("#opt-style").value,
    language: $("#opt-language").value,
    speed: $("#opt-speed").value,
    system: $("#opt-system").value,
    barge: $("#opt-barge").checked,
    greet: $("#opt-greet").checked,
    anchor: $("#opt-anchor").checked,
  });
}

/* ------------------------------------------------------------------ */
/* The conversation                                                    */

const chat = {
  history: [],
  turn: null,
  mic: null,
  // The voice sample's speech id, which every sentence continues so that
  // the voice stays one (reference_speech_id), once prepareVoice() has it.
  anchor: null,
  anchorReady: null,
  totals: [],
};

function setStatus(message, error = false) {
  const status = $("#chat-status");
  status.textContent = message;
  status.classList.toggle("error-text", error);
}

function bubble(role, text = "") {
  $("#conversation-empty").hidden = true;
  const node = element("div", { class: `bubble ${role}` }, element("span", { class: "text", text }));
  $("#conversation").append(node);
  node.scrollIntoView({ block: "end" });
  return node;
}

function bubbleMeta(node, text) {
  let meta = $(".bubble-meta", node);
  if (!meta) {
    meta = element("span", { class: "bubble-meta" });
    node.append(meta);
  }
  meta.textContent = text;
}

/** One exchange: the user's words (heard or typed), the model's answer
 * streamed, and its sentences spoken in order. */
class Turn {
  constructor({ heardAt = null } = {}) {
    this.heardAt = heardAt;
    this.controller = new AbortController();
    this.player = new SplashLogic.TimelinePlayer(MAX_PIECES, {
      context: getAudioContext,
      gap: SENTENCE_GAP,
      onStart: () => this.firstSound(),
    });
    this.player.play();
    this.pieces = [];
    this.speaking = [];
    this.metrics = {};
    this.done = false;
    this.stopped = false;
    this.answered = false;
    this.answer = "";
    this.node = null;
    this.timer = setInterval(() => this.tick(), 100);
  }

  /** Answering, or speaking what it answered. */
  get active() {
    return !this.done || (this.player.started && !this.player.ended);
  }

  firstSound() {
    const now = performance.now();
    if (this.heardAt !== null) this.metrics.total = now - this.heardAt;
    if (this.piece0Sent !== undefined) this.metrics.tts = now - this.piece0Sent;
    this.render();
  }

  tick() {
    this.player.tick();
    if (!this.active) {
      clearInterval(this.timer);
      if (chat.turn === this) {
        chat.turn = null;
        setStatus(chat.mic ? "Listening…" : "Ready.");
      }
    }
  }

  render() {
    const m = this.metrics;
    const parts = [];
    if (m.asr !== undefined) parts.push(`transcribed in ${formatMs(m.asr)}`);
    if (m.llm !== undefined) parts.push(`first token ${formatMs(m.llm)}`);
    if (m.tts !== undefined) parts.push(`first audio ${formatMs(m.tts)} after its sentence`);
    if (m.total !== undefined) parts.push(`${formatMs(m.total)} from the end of your turn`);
    if (this.node) bubbleMeta(this.node, parts.join(" · "));
    $("#metric-asr").textContent = formatMs(m.asr);
    $("#metric-llm").textContent = formatMs(m.llm);
    $("#metric-tts").textContent = formatMs(m.tts);
    $("#metric-total").textContent = formatMs(m.total);
  }

  /** Stop answering and speaking (a new turn, or barge-in). */
  stop() {
    this.stopped = true;
    this.controller.abort();
    this.player.dispose();
    this.done = true;
    clearInterval(this.timer);
  }

  /** The user's text, answered. */
  async respond(text) {
    chat.history.push({ role: "user", content: text });
    this.node = bubble("assistant pending", "…");
    setStatus("Thinking…");
    const splitter = new SplashLogic.SentenceSplitter();
    const started = performance.now();
    try {
      const response = await api(app.bases.llm, "/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: models.llm,
          messages: [{ role: "system", content: $("#opt-system").value.trim() || DEFAULT_SYSTEM[""] }, ...chat.history],
          stream: true,
          reasoning_effort: "none",
          temperature: 0.7,
          max_tokens: 600,
        }),
        signal: this.controller.signal,
      });
      if (!response.ok) throw new Error(await errorMessage(response));
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let end;
        while ((end = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, end).trim();
          buffer = buffer.slice(end + 1);
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") continue;
          let delta = "";
          try {
            const choice = JSON.parse(data).choices?.[0];
            delta = choice?.delta?.content || "";
          } catch (error) {
            continue;
          }
          if (!delta) continue;
          if (this.metrics.llm === undefined) {
            this.metrics.llm = performance.now() - started;
            this.render();
          }
          this.answer += delta;
          this.showAnswer();
          for (const piece of splitter.push(delta)) this.speak(piece);
        }
      }
      for (const piece of splitter.flush()) this.speak(piece);
    } catch (error) {
      if (this.stopped) return;
      this.node.classList.add("error");
      $(".text", this.node).textContent = `The language model failed: ${error.message}`;
      setStatus(error.message, true);
    } finally {
      this.answered = true;
      this.node.classList.remove("pending");
      if (this.answer) chat.history.push({ role: "assistant", content: this.answer });
      this.finishWhenSpoken();
    }
  }

  showAnswer() {
    $(".text", this.node).textContent = this.answer;
    this.node.scrollIntoView({ block: "end" });
  }

  /** Speak one piece; pieces play in the order they are written. */
  speak(text) {
    const index = this.pieces.length;
    this.pieces.push(text);
    if (index >= MAX_PIECES) return;
    if (index === 0) {
      this.piece0Sent = performance.now();
      setStatus("Speaking…");
    }
    this.speaking.push(this.synthesize(index, text, voiceAnchor()));
  }

  async synthesize(index, text, wait) {
    const body = speechBody(text);
    const anchor = await wait;
    if (anchor) body.reference_speech_id = anchor;
    try {
      const send = () => api(app.bases.tts, "/v1/audio/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: this.controller.signal,
      });
      let response = await send();
      if (!response.ok && body.reference_speech_id && response.status < 500) {
        // The server no longer holds the voice sample (it restarted, or
        // forgot it): speak without it, and prepare the voice again.
        delete body.reference_speech_id;
        if (chat.anchor === anchor) resetVoice();
        response = await send();
      }
      if (!response.ok) throw new Error(await errorMessage(response));
      const pcm = new PcmStream();
      // Played faster at the same pitch: the model speaks slowly.
      const stretcher = new SplashLogic.TimeStretcher(SPEECH_RATE, speechSpeed());
      const reader = response.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        const samples = stretcher.push(pcm.push(value));
        if (samples.length) this.player.push(index, samples, SPEECH_RATE);
      }
      const rest = stretcher.flush();
      if (rest.length) this.player.push(index, rest, SPEECH_RATE);
    } catch (error) {
      if (!this.stopped) setStatus(`Speech failed: ${error.message}`, true);
    } finally {
      this.player.finish(index);
    }
  }

  async finishWhenSpoken() {
    await Promise.allSettled(this.speaking);
    this.done = true;
    this.player.close();
    if (!this.pieces.length && chat.turn === this) {
      chat.turn = null;
      setStatus(chat.mic ? "Listening…" : "Ready.");
    }
    if (this.metrics.total !== undefined) {
      chat.totals.push(this.metrics.total);
    }
  }
}

/** A speech request's body in the chat's voice (language, speaker, style). */
function speechBody(text, extra = {}) {
  const body = { model: models.tts, input: text, stream: true, response_format: "pcm", ...extra };
  const language = TTS_LANGUAGE[$("#opt-language").value];
  if (language) body.language = language;
  if (models.ttsType === "custom_voice" && $("#opt-voice").value) body.voice = $("#opt-voice").value;
  const style = $("#opt-style").value.trim();
  if (models.ttsType === "voice_design") body.instructions = style || DEFAULT_DESIGN;
  else if (models.ttsType === "custom_voice" && style) body.instructions = style;
  return body;
}

/** The speech id answers continue, or null (no references, or the option
 * off): the voice sample, prepared once per voice. */
function voiceAnchor() {
  if (!models.references || !$("#opt-anchor").checked) return Promise.resolve(null);
  if (!chat.anchorReady) {
    const ready = prepareVoice().then((id) => {
      // Only the voice still chosen is kept.
      if (chat.anchorReady === ready) chat.anchor = id;
      return id;
    });
    chat.anchorReady = ready;
  }
  return chat.anchorReady;
}

/** Forget the voice (its speaker, style or language changed, or the server
 * lost it) and prepare the new one ahead of the next answer. */
function resetVoice() {
  chat.anchor = null;
  chat.anchorReady = null;
  if (app.bases) voiceAnchor();
}

/** Speak the voice sample unheard, up to VOICE_TRIES times, and keep the
 * first take at the target pace, else the fastest: its speech id. */
async function prepareVoice() {
  const text = VOICE_SAMPLE[$("#opt-language").value] || VOICE_SAMPLE[""];
  const cjk = (text.match(/[\u3400-\u9fff]/g) || []).length;
  const units = cjk || text.split(/\s+/).filter(Boolean).length;
  const target = cjk ? TARGET_PACE.characters : TARGET_PACE.words;
  let best = null;
  for (let attempt = 0; attempt < VOICE_TRIES; attempt++) {
    try {
      const body = speechBody(text, { stream: false, seed: Math.floor(Math.random() * 2 ** 31) });
      const response = await api(app.bases.tts, "/v1/audio/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(await errorMessage(response));
      const id = response.headers.get("X-Splash-Speech-Id");
      const seconds = speakingSeconds(new PcmStream().push(new Uint8Array(await response.arrayBuffer())));
      const pace = seconds > 0 ? units / seconds : 0;
      if (id && (!best || pace > best.pace)) best = { id, pace };
      if (best && best.pace >= target) break;
    } catch (error) {
      break;
    }
  }
  return best ? best.id : null;
}

/** Seconds from the first to the last 20 ms window above -35 dBFS. */
function speakingSeconds(samples) {
  const window = Math.round(SPEECH_RATE * 0.02);
  let first = -1;
  let last = -1;
  for (let start = 0, index = 0; start + window <= samples.length; start += window, index++) {
    let sum = 0;
    for (let i = start; i < start + window; i++) sum += samples[i] * samples[i];
    const level = 20 * Math.log10(Math.sqrt(sum / window) / 32768 + 1e-9);
    if (level > -35) {
      if (first < 0) first = index;
      last = index;
    }
  }
  return first < 0 ? 0 : (last - first + 1) * 0.02;
}

/** How much faster than the model speaks the answer is played. */
function speechSpeed() {
  const value = Number($("#opt-speed").value);
  return isFinite(value) && value > 0 ? Math.min(3, Math.max(0.5, value)) : 1.5;
}

/** Answer `text` (typed, or the transcript of `samples`). */
async function startTurn({ text = null, samples = null, heardAt = null }) {
  if (chat.turn) chat.turn.stop();
  getAudioContext();
  const turn = new Turn({ heardAt });
  chat.turn = turn;
  if (samples) {
    const userNode = bubble("user pending", "…");
    setStatus("Transcribing…");
    const started = performance.now();
    try {
      text = await transcribe(samples, turn.controller.signal);
    } catch (error) {
      if (turn.stopped) return;
      userNode.remove();
      turn.stop();
      chat.turn = null;
      setStatus(`Transcription failed: ${error.message}`, true);
      return;
    }
    turn.metrics.asr = performance.now() - started;
    userNode.classList.remove("pending");
    if (!text) {
      userNode.remove();
      turn.stop();
      chat.turn = null;
      setStatus(chat.mic ? "Listening… (heard nothing to answer)" : "Ready.");
      return;
    }
    $(".text", userNode).textContent = text;
  } else {
    bubble("user", text);
  }
  turn.render();
  await turn.respond(text);
}

async function transcribe(samples, signal) {
  const form = new FormData();
  const wav = new Blob([SplashLogic.wavBytes([SplashLogic.int16FromFloat(samples)], LIVE_RATE)], { type: "audio/wav" });
  form.append("file", wav, "utterance.wav");
  form.append("model", models.asr);
  const language = $("#opt-language").value;
  if (language) form.append("language", language);
  form.append("response_format", "json");
  const response = await api(app.bases.asr, "/v1/audio/transcriptions", { method: "POST", body: form, signal });
  if (!response.ok) throw new Error(await errorMessage(response));
  const data = await response.json();
  return (data.text || "").trim();
}

/** Speak a fixed line (the greeting), as the assistant. */
async function say(text) {
  if (chat.turn) chat.turn.stop();
  const turn = new Turn();
  chat.turn = turn;
  turn.node = bubble("assistant", text);
  turn.answered = true;
  turn.answer = text;
  turn.speak(text);
  turn.finishWhenSpoken();
}

/* ------------------------------------------------------------------ */
/* The microphone                                                      */

class Microphone {
  constructor() {
    this.segmenter = new SplashLogic.Segmenter({ rate: LIVE_RATE, ...VAD });
  }

  async start() {
    const context = getAudioContext();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("This browser does not offer the microphone to this page.");
    }
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: {
      channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true,
    } });
    this.resampler = new SplashLogic.Resampler(context.sampleRate, LIVE_RATE);
    this.input = context.createMediaStreamSource(this.stream);
    this.processor = context.createScriptProcessor(4096, 1, 1);
    this.processor.onaudioprocess = (event) => this.feed(event.inputBuffer.getChannelData(0));
    this.input.connect(this.processor);
    this.processor.connect(context.destination);
  }

  /** While the assistant answers, the microphone is ignored unless
   * barge-in is on (its own voice would be heard and answered). */
  get deaf() {
    return !!chat.turn && chat.turn.active && !$("#opt-barge").checked;
  }

  feed(samples) {
    const resampled = this.resampler.push(samples);
    if (this.deaf) {
      // What it heard of the answer is forgotten.
      if (!this.muted) this.segmenter = new SplashLogic.Segmenter({ rate: LIVE_RATE, ...VAD });
      this.muted = true;
      $("#mic-meter").style.width = "0";
      return;
    }
    this.muted = false;
    for (const event of this.segmenter.push(resampled)) this.handle(event);
    const level = this.segmenter.levelDb;
    $("#mic-meter").style.width = `${Math.max(0, Math.min(100, (level + 60) * 1.8))}%`;
  }

  handle(event) {
    if (event.type === "start") {
      if (chat.turn && chat.turn.active) {
        // Barge-in: the user speaks over the answer.
        chat.turn.stop();
        chat.turn = null;
      }
      setStatus("Hearing you…");
    } else if (event.type === "drop") {
      setStatus("Listening…");
    } else if (event.type === "final") {
      startTurn({ samples: event.samples, heardAt: performance.now() });
    }
  }

  stop() {
    if (this.processor) {
      this.processor.onaudioprocess = null;
      this.processor.disconnect();
    }
    if (this.input) this.input.disconnect();
    if (this.stream) for (const track of this.stream.getTracks()) track.stop();
    $("#mic-meter").style.width = "0";
  }
}

async function toggleMic() {
  const button = $("#btn-mic");
  if (chat.mic) {
    chat.mic.stop();
    chat.mic = null;
    button.setAttribute("aria-pressed", "false");
    $("#mic-label").textContent = "Start talking";
    if (!chat.turn) setStatus("Ready.");
    return;
  }
  getAudioContext();
  const mic = new Microphone();
  try {
    await mic.start();
  } catch (error) {
    setStatus(`Microphone: ${error.message}`, true);
    return;
  }
  chat.mic = mic;
  button.setAttribute("aria-pressed", "true");
  $("#mic-label").textContent = "Stop";
  setStatus("Listening…");
  if ($("#opt-greet").checked && !chat.history.length) say(GREETING[$("#opt-language").value]);
}

/** Stop listening, answering and speaking. */
function haltConversation(message) {
  if (chat.turn) chat.turn.stop();
  chat.turn = null;
  if (chat.mic) toggleMic();
  if (message && app.view === "chat") setStatus(message, true);
}

function newConversation() {
  if (chat.turn) chat.turn.stop();
  chat.turn = null;
  chat.history = [];
  resetVoice();
  $("#conversation").replaceChildren($("#conversation-empty"));
  $("#conversation-empty").hidden = false;
  for (const id of ["asr", "llm", "tts", "total"]) $(`#metric-${id}`).textContent = "–";
  setStatus(chat.mic ? "Listening…" : "Ready.");
}

/* ------------------------------------------------------------------ */

function bind() {
  $("#btn-stop").addEventListener("click", stopServers);
  $("#btn-cancel").addEventListener("click", stopServers);
  $("#btn-mic").addEventListener("click", toggleMic);
  $("#btn-clear").addEventListener("click", newConversation);
  $("#compose").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = $("#compose-text").value.trim();
    if (!text) return;
    $("#compose-text").value = "";
    startTurn({ text });
  });
  for (const id of ["#opt-voice", "#opt-style", "#opt-system", "#opt-speed", "#opt-barge", "#opt-greet", "#opt-anchor"]) {
    $(id).addEventListener("change", saveChatOptions);
  }
  $("#opt-voice").addEventListener("change", resetVoice);
  $("#opt-style").addEventListener("change", resetVoice);
  $("#opt-anchor").addEventListener("change", resetVoice);
  $("#opt-language").addEventListener("change", () => {
    const system = $("#opt-system");
    // Follow the language unless the instructions were edited.
    if (Object.values(DEFAULT_SYSTEM).includes(system.value)) system.value = DEFAULT_SYSTEM[$("#opt-language").value];
    saveChatOptions();
    resetVoice();
  });
}

async function main() {
  bind();
  showView("progress");
  poll();
}

main();
