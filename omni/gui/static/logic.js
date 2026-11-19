/* splash-omni page logic without the DOM: what the page decides from a
 * server's /v1/models listing, the bodies of its speech requests, and the
 * audio processing of the live transcription page (resampling, voice
 * activity segmentation, 16 kHz WAV encoding).
 *
 * Kept free of document, window and Web Audio so that the tests run it in
 * JavaScriptCore (dev/tests/gui/test_logic.py), the engine Safari uses.
 */
"use strict";

var SplashLogic = (function () {
  /* ---------------------------------------------------------------- */
  /* Voice controls a speech model takes                               */

  const BASE = "base";
  const VOICE_DESIGN = "voice_design";
  const CUSTOM_VOICE = "custom_voice";
  const MODEL_TYPES = [BASE, VOICE_DESIGN, CUSTOM_VOICE];

  /** What the page offers for a speech model, from its /v1/models entry
   * (tts_model_type, instructions, voices, references). A server that
   * names no model type (another framework) gets the generic controls: an
   * optional instruction and a free-text voice. */
  function voiceControls(model) {
    const entry = model && typeof model === "object" ? model : {};
    const type = MODEL_TYPES.includes(entry.tts_model_type) ? entry.tts_model_type : null;
    let instructions = ["required", "optional", "unsupported"].includes(entry.instructions)
      ? entry.instructions
      : null;
    if (instructions === null) {
      instructions = type === BASE ? "unsupported" : type === VOICE_DESIGN ? "required" : "optional";
    }
    const voices = Array.isArray(entry.voices)
      ? entry.voices.map((voice) => (voice && typeof voice === "object" ? voice.id || voice.name : voice))
        .filter((voice) => typeof voice === "string" && voice)
      : null;
    // The server speaks a request as a continuation of an earlier one
    // (reference_speech_id), in its voice.
    const references = entry.references === true;
    return {
      type,
      references,
      instructions,
      // A preset speaker list: a dropdown, and a choice the request needs.
      voices: voices && voices.length ? voices : null,
      voiceRequired: type === CUSTOM_VOICE,
      // A free-text voice for a server whose voices the page does not know.
      freeVoice: type === null && !(voices && voices.length),
      paragraphs: paragraphModeAllowed(type, references),
    };
  }

  /** Paragraph mode needs one voice across separate requests. A server
   * with references speaks every paragraph after the first as its
   * continuation, in its voice, whatever the model. Without them, a
   * VoiceDesign description or a CustomVoice speaker fixes the voice only
   * roughly, and the Base model has no fixed voice at all, so it is not
   * offered there. A server of unknown type keeps the mode; the voice it
   * sends decides. */
  /** The run view's tabs a server of `task` answers: a speech server's
   * one, a transcription server's two (they share its API), every tab while
   * the task is unknown. Another tab needs another server, which the page
   * starts only after stopping this one. */
  function runTabs(task) {
    if (task === "tts") return ["tts"];
    if (task === "asr") return ["asr", "live"];
    return ["tts", "asr", "live"];
  }

  /** A language model Splash serves, named for the voice chat's setup:
   * "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M" reads "Qwen3.6-35B-A3B · GGUF
   * Q4_K_M", "mlx-community/Qwen3.8-27B-4bit" reads "Qwen3.8-27B · MLX 4-bit". */
  function languageModelName(id) {
    const [repo, variant] = String(id).split(":");
    let name = repo.split("/").pop();
    const gguf = /-GGUF$/i.test(name);
    name = name.replace(/-GGUF$/i, "");
    const bits = /-(\d+)bit$/i.exec(name);
    if (bits) name = name.slice(0, bits.index);
    const quantization = variant ? variant.replace(/^UD-/, "") : bits ? `${bits[1]}-bit` : "";
    return `${name} · ${[gguf ? "GGUF" : "MLX", quantization].filter(Boolean).join(" ")}`;
  }

  function paragraphModeAllowed(type, references = false) {
    return references || type !== BASE;
  }

  /** The voices a /v1/audio/voices answer lists: a list of names or of
   * objects with an id or name, bare or under data or voices. */
  function voiceList(answer) {
    let items = answer;
    if (items && !Array.isArray(items)) items = items.voices || items.data || null;
    if (!Array.isArray(items)) return [];
    return items
      .map((voice) => (voice && typeof voice === "object" ? voice.id || voice.voice_id || voice.name : voice))
      .filter((voice) => typeof voice === "string" && voice);
  }

  /** A speech request's body. `controls` is voiceControls()'s answer;
   * `voice` and `instructions` are what the page holds. Throws, with the
   * message the page shows, for a request the model would refuse. Every
   * paragraph of a run is built by this with the same voice, instructions
   * and seed, so the voice stays the same. */
  function speechBody({ input, parameters = {}, controls, instructions = "", voice = "", seed = null, model = "" }) {
    const body = { ...parameters, input };
    if (model) body.model = model;
    if (seed !== null && seed !== undefined) body.seed = seed;
    delete body.voice;
    delete body.instructions;
    const kind = controls || voiceControls(null);
    const prompt = typeof instructions === "string" ? instructions.trim() : "";
    const speaker = typeof voice === "string" ? voice.trim() : "";
    if (kind.instructions === "required" && !prompt) {
      throw new Error("Describe the voice in the system prompt: this VoiceDesign model needs it.");
    }
    if (kind.instructions !== "unsupported" && prompt) body.instructions = prompt;
    if (kind.voiceRequired && !speaker) {
      throw new Error("Choose a speaker: this CustomVoice model needs one.");
    }
    if ((kind.voices || kind.freeVoice) && speaker && kind.type !== BASE && kind.type !== VOICE_DESIGN) {
      body.voice = speaker;
    }
    return body;
  }

  /** Text split at blank lines, as paragraph mode sends it. */
  function paragraphs(text) {
    return String(text).split(/\n[ \t]*\n/).map((part) => part.trim()).filter(Boolean);
  }

  /* ---------------------------------------------------------------- */
  /* Audio: resampling and WAV                                         */

  /** A streaming resampler to `outRate` (16 kHz for transcription): a
   * windowed-sinc low-pass below the output's Nyquist frequency, then
   * linear interpolation. Chunk boundaries do not change the output. */
  class Resampler {
    constructor(inRate, outRate = 16000, taps = 33) {
      this.inRate = inRate;
      this.outRate = outRate;
      this.step = inRate / outRate;
      this.filter = null;
      if (inRate > outRate) {
        const cutoff = (0.45 * outRate) / inRate;
        const half = (taps - 1) / 2;
        const filter = new Float32Array(taps);
        let sum = 0;
        for (let i = 0; i < taps; i++) {
          const x = i - half;
          const sinc = x === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * x) / (Math.PI * x);
          const window = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (taps - 1));
          filter[i] = sinc * window;
          sum += filter[i];
        }
        for (let i = 0; i < taps; i++) filter[i] /= sum;
        this.filter = filter;
      }
      const history = this.filter ? this.filter.length - 1 : 0;
      // The last inputs the filter needs, and filtered samples not yet used.
      this.history = new Float32Array(history);
      this.filtered = new Float32Array(0);
      // The next output's position in `filtered`.
      this.position = 0;
    }

    push(input) {
      let filtered;
      if (this.filter) {
        const taps = this.filter.length;
        const joined = new Float32Array(this.history.length + input.length);
        joined.set(this.history);
        joined.set(input, this.history.length);
        filtered = new Float32Array(input.length);
        for (let i = 0; i < input.length; i++) {
          let acc = 0;
          for (let k = 0; k < taps; k++) acc += joined[i + k] * this.filter[k];
          filtered[i] = acc;
        }
        this.history = joined.slice(joined.length - this.history.length);
      } else {
        filtered = input;
      }
      const data = new Float32Array(this.filtered.length + filtered.length);
      data.set(this.filtered);
      data.set(filtered, this.filtered.length);
      const out = [];
      let position = this.position;
      while (position + 1 < data.length) {
        const index = Math.floor(position);
        const fraction = position - index;
        out.push(data[index] * (1 - fraction) + data[index + 1] * fraction);
        position += this.step;
      }
      // Past the end of `data` when the step skips whole samples.
      const keep = Math.min(Math.floor(position), data.length);
      this.filtered = data.slice(keep);
      this.position = position - keep;
      return Float32Array.from(out);
    }
  }

  function int16FromFloat(samples) {
    const out = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      out[i] = Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767)));
    }
    return out;
  }

  /** A 16-bit mono PCM WAV file of `chunks` (Int16Arrays), as bytes. */
  function wavBytes(chunks, rate) {
    const length = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
    const bytes = new Uint8Array(44 + length);
    const header = new DataView(bytes.buffer);
    const write = (offset, text) => { for (let i = 0; i < text.length; i++) header.setUint8(offset + i, text.charCodeAt(i)); };
    write(0, "RIFF");
    header.setUint32(4, 36 + length, true);
    write(8, "WAVE");
    write(12, "fmt ");
    header.setUint32(16, 16, true);
    header.setUint16(20, 1, true);
    header.setUint16(22, 1, true);
    header.setUint32(24, rate, true);
    header.setUint32(28, rate * 2, true);
    header.setUint16(32, 2, true);
    header.setUint16(34, 16, true);
    write(36, "data");
    header.setUint32(40, length, true);
    let offset = 44;
    for (const chunk of chunks) {
      bytes.set(new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength), offset);
      offset += chunk.byteLength;
    }
    return bytes;
  }

  /* ---------------------------------------------------------------- */
  /* Live transcription: voice activity segmentation                   */

  /** Cuts a continuous 16 kHz stream into utterances by energy. A 20 ms
   * frame is voiced when its RMS level is above the threshold: a fixed
   * one, or the background noise level (followed while nobody speaks)
   * plus `marginDb`, never below `floorDb`. An utterance starts at the
   * first voiced frame, with `preRollMs` of audio before it; it is final
   * after `silenceMs` without a voiced frame, or once it is `maxSeconds`
   * long (the next one continues at once). One with less than
   * `minSpeechMs` of voiced frames is dropped as a click. While an
   * utterance grows, a partial is offered every `partialMs` of audio.
   *
   * push() returns the events the samples complete:
   *   {type: "start", at}                       an utterance began
   *   {type: "partial", id, samples, seconds}   the utterance so far
   *   {type: "final", id, samples, seconds, reason: "silence" | "max" | "flush"}
   *   {type: "drop", id}                        too little speech; ignore its partials
   * `at` and `seconds` are in seconds of the stream. */
  class Segmenter {
    constructor(options = {}) {
      this.rate = options.rate || 16000;
      this.frame = Math.round((this.rate * (options.frameMs || 20)) / 1000);
      this.silenceFrames = Math.max(1, Math.round((options.silenceMs ?? 700) / (options.frameMs || 20)));
      this.minSpeechFrames = Math.max(1, Math.round((options.minSpeechMs ?? 200) / (options.frameMs || 20)));
      this.preRoll = Math.round((this.rate * (options.preRollMs ?? 300)) / 1000);
      this.maxSamples = Math.round(this.rate * (options.maxSeconds || 15));
      this.partialSamples = options.partialMs ? Math.round((this.rate * options.partialMs) / 1000) : 0;
      this.thresholdDb = options.thresholdDb ?? null;
      this.marginDb = options.marginDb ?? 12;
      this.floorDb = options.floorDb ?? -50;
      this.noiseDb = null;
      this.pending = new Float32Array(0);
      this.recent = [];
      this.recentLength = 0;
      this.utterance = null;
      this.nextId = 1;
      this.consumed = 0;
      this.levelDb = -Infinity;
    }

    static levelDb(frame) {
      let sum = 0;
      for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
      const rms = Math.sqrt(sum / Math.max(1, frame.length));
      return rms > 0 ? 20 * Math.log10(rms) : -120;
    }

    /** The level above which a frame is voiced, in dBFS. */
    get threshold() {
      if (this.thresholdDb !== null) return this.thresholdDb;
      const noise = this.noiseDb === null ? -120 : this.noiseDb;
      return Math.max(this.floorDb, noise + this.marginDb);
    }

    push(samples) {
      const events = [];
      const data = new Float32Array(this.pending.length + samples.length);
      data.set(this.pending);
      data.set(samples, this.pending.length);
      let offset = 0;
      for (; offset + this.frame <= data.length; offset += this.frame) {
        this.frameEvents(data.slice(offset, offset + this.frame), events);
      }
      this.pending = data.slice(offset);
      return events;
    }

    /** Ends the stream: the utterance in progress is final. */
    flush() {
      const events = [];
      if (this.pending.length && this.utterance) this.append(this.pending);
      this.pending = new Float32Array(0);
      if (this.utterance) this.finish("flush", events);
      return events;
    }

    append(frame) {
      const utterance = this.utterance;
      utterance.chunks.push(frame);
      utterance.length += frame.length;
    }

    frameEvents(frame, events) {
      const level = Segmenter.levelDb(frame);
      this.levelDb = level;
      const voiced = level > this.threshold;
      const start = this.consumed;
      this.consumed += frame.length;
      if (!this.utterance) {
        if (!voiced) {
          // Follow the background: fall fast, rise slowly.
          if (this.noiseDb === null) this.noiseDb = level;
          else this.noiseDb += (level - this.noiseDb) * (level < this.noiseDb ? 0.3 : 0.02);
          this.remember(frame);
          return;
        }
        const preRoll = this.recent.splice(0);
        const before = this.recentLength;
        this.recentLength = 0;
        this.utterance = {
          id: this.nextId++,
          chunks: preRoll,
          length: before,
          start: (start - before) / this.rate,
          voiced: 0,
          silent: 0,
          lastPartial: 0,
        };
        events.push({ type: "start", id: this.utterance.id, at: this.utterance.start });
      }
      const utterance = this.utterance;
      this.append(frame);
      if (voiced) {
        utterance.voiced++;
        utterance.silent = 0;
      } else {
        utterance.silent++;
      }
      if (utterance.silent >= this.silenceFrames) {
        this.finish("silence", events);
        return;
      }
      if (utterance.length >= this.maxSamples) {
        this.finish("max", events);
        return;
      }
      // A partial while speech goes on, not during the closing pause.
      if (this.partialSamples && voiced && utterance.length - utterance.lastPartial >= this.partialSamples &&
          utterance.voiced >= this.minSpeechFrames) {
        utterance.lastPartial = utterance.length;
        events.push({ type: "partial", id: utterance.id, samples: this.samples(), seconds: utterance.length / this.rate });
      }
    }

    remember(frame) {
      this.recent.push(frame);
      this.recentLength += frame.length;
      while (this.recent.length && this.recentLength - this.recent[0].length >= this.preRoll) {
        this.recentLength -= this.recent.shift().length;
      }
    }

    samples(utterance = this.utterance) {
      const out = new Float32Array(utterance.length);
      let offset = 0;
      for (const chunk of utterance.chunks) {
        out.set(chunk, offset);
        offset += chunk.length;
      }
      return out;
    }

    finish(reason, events) {
      const utterance = this.utterance;
      this.utterance = null;
      if (utterance.voiced < this.minSpeechFrames) {
        events.push({ type: "drop", id: utterance.id });
        return;
      }
      let samples = this.samples(utterance);
      if (reason === "silence") {
        // Keep a little of the trailing silence, not all of it.
        const tail = Math.max(0, utterance.silent * this.frame - Math.round(this.rate * 0.2));
        samples = samples.slice(0, samples.length - tail);
      }
      events.push({
        type: "final",
        id: utterance.id,
        samples,
        seconds: samples.length / this.rate,
        start: utterance.start,
        reason,
      });
    }
  }

  /* ---------------------------------------------------------------- */
  /* Voice chat: a streamed reply cut into sentences to speak          */

  /** Text a language model streams, cut into pieces to speak as soon as
   * each is complete: at a sentence end (. ! ? ; and their CJK forms, or a
   * line break), with a piece shorter than `minChars` joined to the next
   * so that speech is not choppy. The first piece may be cut sooner, at a
   * comma once it is `firstCommaChars` long, for an early first sound.
   * Markdown marks, emoji and stray whitespace are removed, as a speech
   * model would read them aloud. push() returns the pieces the text
   * completes; flush() the rest. */
  class SentenceSplitter {
    constructor({ minChars = 24, firstMinChars = 2, firstCommaChars = 48, clean = true } = {}) {
      this.tidy = clean ? SentenceSplitter.clean : (text) => text.replace(/\s+/g, " ").trim();
      this.minChars = minChars;
      this.firstMinChars = firstMinChars;
      this.firstCommaChars = firstCommaChars;
      this.buffer = "";
      this.held = "";
      this.count = 0;
    }

    // A word whose full stop does not end a sentence: an initial, or a
    // common abbreviation.
    static ABBREVIATION = /(?:^|[\s(])(?:[A-Za-z]|Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|No|e\.g|i\.e|a\.m|p\.m|U\.S)\.$/;

    static clean(text) {
      return text
        .replace(/```[\s\S]*?```/g, " ")
        .replace(/`([^`]*)`/g, "$1")
        .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
        .replace(/^\s{0,3}(#{1,6}|>|[-*+]|\d+[.)])\s+/gm, "")
        .replace(/[*_~#]+/g, "")
        .replace(/\p{Extended_Pictographic}\uFE0F?/gu, "")
        .replace(/\s+/g, " ")
        .trim();
    }

    push(text) {
      this.buffer += text;
      const pieces = [];
      for (;;) {
        const cut = this.boundary();
        if (cut < 0) break;
        this.offer(this.buffer.slice(0, cut), pieces);
        this.buffer = this.buffer.slice(cut);
      }
      return pieces;
    }

    flush() {
      const pieces = [];
      const rest = this.tidy(this.join(this.buffer));
      this.buffer = "";
      this.held = "";
      if (rest) {
        pieces.push(rest);
        this.count++;
      }
      return pieces;
    }

    /** Where the buffer's first complete sentence ends, or -1. A Latin
     * full stop counts once whitespace follows it (not inside 3.14) and
     * not after an abbreviation (a.m., Dr.); a CJK one at once. */
    boundary() {
      const ends = /[.!?;:…]+["')\]]*\s|[。！？；…]+["'”’）」』]*|\n/g;
      let match;
      while ((match = ends.exec(this.buffer))) {
        const before = this.buffer.slice(0, match.index + 1);
        if (match[0][0] === "." && SentenceSplitter.ABBREVIATION.test(before)) continue;
        return match.index + match[0].length;
      }
      if (this.count === 0 && !this.held) {
        const comma = /[,，、]\s?/g;
        let found = -1;
        let hit;
        while ((hit = comma.exec(this.buffer))) {
          if (hit.index >= this.firstCommaChars) {
            found = hit.index + hit[0].length;
            break;
          }
        }
        return found;
      }
      return -1;
    }

    /** Text joined to what is held: CJK text without a space. */
    join(raw) {
      if (!this.held) return raw;
      const cjk = /[\u3000-\u303f\u3400-\u9fff\uff00-\uffef]$/.test(this.held);
      return this.held + (cjk ? "" : " ") + raw;
    }

    /** A piece's length as speech: a CJK character says about as much as
     * three Latin ones. */
    static weight(text) {
      const cjk = (text.match(/[\u3400-\u9fff]/g) || []).length;
      return text.length + 2 * cjk;
    }

    offer(raw, pieces) {
      const text = this.tidy(this.join(raw));
      const minimum = this.count === 0 ? this.firstMinChars : this.minChars;
      if (SentenceSplitter.weight(text) < minimum) {
        this.held = text;
        return;
      }
      this.held = "";
      pieces.push(text);
      this.count++;
    }
  }

  /** A long text cut, at sentence ends, into chunks of at most `maxWeight`
   * (SentenceSplitter.weight: about 25 s of speech by default), which the
   * page speaks as one request each, every chunk after the first a
   * continuation of the first: a long single request drifts quieter as it
   * goes (the model's own behaviour), short ones keep the level. A text of
   * at most `minimumWeight` stays whole. A sentence longer than a chunk is
   * a chunk of its own. */
  function speechChunks(text, { maxWeight = 360, minimumWeight = 480 } = {}) {
    const whole = String(text).trim();
    if (SentenceSplitter.weight(whole) <= minimumWeight) return whole ? [whole] : [];
    const splitter = new SentenceSplitter({ minChars: 0, firstMinChars: 0, firstCommaChars: Infinity, clean: false });
    const sentences = [...splitter.push(whole + "\n"), ...splitter.flush()];
    const chunks = [];
    let current = "";
    for (const sentence of sentences) {
      const joined = current ? current + (/[\u3000-\u303f\u3400-\u9fff\uff00-\uffef]$/.test(current) ? "" : " ") + sentence : sentence;
      if (current && SentenceSplitter.weight(joined) > maxWeight) {
        chunks.push(current);
        current = sentence;
      } else {
        current = joined;
      }
    }
    if (current) chunks.push(current);
    return chunks;
  }

  /* ---------------------------------------------------------------- */
  /* Speech speed: time-scale modification                             */

  /** Speech played `factor` times as fast at the same pitch (WSOLA:
   * waveform-similarity overlap-add), as int16 samples stream in. Frames
   * of 40 ms are taken from the input every 20 ms × factor and added at
   * 20 ms steps under a Hann window; each frame is moved by up to ±10 ms to
   * where it best continues the frame before it (cross-correlation on a
   * decimated copy), so the overlaps add in phase and pitch periods are
   * neither doubled nor cut. A factor of 1 passes the audio through.
   * push() returns the samples ready; flush() the rest at the end. */
  class TimeStretcher {
    constructor(rate, factor, { frameMs = 40, searchMs = 10, decimate = 4 } = {}) {
      this.factor = factor;
      this.bypass = !(factor > 0) || Math.abs(factor - 1) < 1e-3;
      this.size = 2 * Math.round((rate * frameMs) / 2000);
      this.hop = this.size / 2;
      this.search = Math.round((rate * searchMs) / 1000);
      this.decimate = decimate;
      this.window = new Float32Array(this.size);
      for (let i = 0; i < this.size; i++) this.window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / this.size);
      // Input not yet consumed, from absolute sample `base`.
      this.input = new Float32Array(0);
      this.base = 0;
      // Overlap-add of the frame in progress, and the frames taken.
      this.overlap = new Float32Array(this.size);
      this.frames = 0;
      // Where the last frame was taken from (absolute), or null.
      this.previous = null;
    }

    push(samples) {
      if (this.bypass) return samples;
      const joined = new Float32Array(this.input.length + samples.length);
      joined.set(this.input);
      for (let i = 0; i < samples.length; i++) joined[this.input.length + i] = samples[i];
      this.input = joined;
      return this.run(false);
    }

    flush() {
      if (this.bypass) return new Int16Array(0);
      return this.run(true);
    }

    /** The frames the input now allows; at the end, the input padded with
     * silence, and the overlap's last half. */
    run(end) {
      const out = [];
      for (;;) {
        const nominal = Math.round(this.frames * this.hop * this.factor);
        const reach = nominal + this.search + this.size;
        if (!end && reach > this.base + this.input.length) break;
        if (end && nominal >= this.base + this.input.length) break;
        const start = this.previous === null ? nominal : this.align(nominal);
        const frame = this.frame(start);
        const first = this.frames === 0;
        for (let i = 0; i < this.size; i++) {
          // The first frame has nothing to overlap: its first half is whole.
          const weight = first && i < this.hop ? 1 : this.window[i];
          this.overlap[i] += frame[i] * weight;
        }
        out.push(this.overlap.slice(0, this.hop));
        this.overlap.copyWithin(0, this.hop);
        this.overlap.fill(0, this.hop);
        this.previous = start;
        this.frames++;
        // Keep the input from the search window of the next frame.
        const keep = Math.max(0, Math.min(
          Math.round(this.frames * this.hop * this.factor) - this.search,
          this.previous + this.hop) - this.base);
        if (keep > 0) {
          this.input = this.input.slice(Math.min(keep, this.input.length));
          this.base += keep;
        }
      }
      if (end) {
        out.push(this.overlap.slice(0, this.hop));
        this.overlap.fill(0);
      }
      let length = 0;
      for (const part of out) length += part.length;
      const result = new Int16Array(length);
      let offset = 0;
      for (const part of out) {
        for (let i = 0; i < part.length; i++) {
          const value = Math.round(part[i]);
          result[offset++] = value > 32767 ? 32767 : value < -32768 ? -32768 : value;
        }
      }
      return result;
    }

    /** `size` input samples from absolute position `start`, silence beyond
     * what is here. */
    frame(start) {
      const out = new Float32Array(this.size);
      const from = start - this.base;
      for (let i = 0; i < this.size; i++) {
        const at = from + i;
        if (at >= 0 && at < this.input.length) out[i] = this.input[at];
      }
      return out;
    }

    /** The start near `nominal` whose frame best continues the previous
     * frame's natural successor (the input a hop after it). */
    align(nominal) {
      const target = this.previous + this.hop;
      const step = this.decimate;
      let best = nominal;
      let bestScore = -Infinity;
      for (let shift = -this.search; shift <= this.search; shift += step) {
        const start = nominal + shift;
        if (start < this.base) continue;
        let dot = 0;
        let energy = 1e-9;
        for (let i = 0; i < this.size; i += step) {
          const a = this.input[start - this.base + i] || 0;
          const b = this.input[target - this.base + i] || 0;
          dot += a * b;
          energy += a * a;
        }
        const score = dot / Math.sqrt(energy);
        if (score > bestScore) {
          bestScore = score;
          best = start;
        }
      }
      return best;
    }
  }

  /* ---------------------------------------------------------------- */
  /* Speech playback: one track                                        */

  /** The one track of a speech run. Segments' audio is placed on a single
   * timeline in order as it arrives (a segment once every segment before it
   * has finished, after a short gap), and played with Web Audio from any
   * position while it is still arriving: play, pause and seek act on this one
   * timeline, so exactly one set of sources sounds at a time, during
   * generation and after it. When playback catches up with generation it
   * waits for the next audio. `context` returns the AudioContext to play
   * with; `gap` is the silence between segments, in seconds. */
  class TimelinePlayer {
    constructor(count, { context, gap = 0.3, onStart } = {}) {
      this.context = context;
      this.gap = gap;
      this.queues = Array.from({ length: count }, () => []);
      this.finished = new Array(count).fill(false);
      this.cursor = 0;
      this.pieces = [];
      this.length = 0;
      this.gapPending = false;
      this.complete = count === 0;
      this.onStart = onStart;
      this.started = false;
      this.playing = false;
      this.paused = 0;
      this.origin = 0;
      this.from = 0;
      this.sources = new Set();
      this.disposed = false;
    }

    push(index, samples, rate) {
      if (this.disposed || !samples.length) return;
      this.queues[index].push({ samples, rate });
      this.pump();
    }

    finish(index) {
      this.finished[index] = true;
      this.pump();
    }

    /** Everything still to come has ended (a stop, or a failure). */
    close() {
      this.finished.fill(true);
      this.pump();
    }

    pump() {
      while (this.cursor < this.queues.length && !this.disposed) {
        const queue = this.queues[this.cursor];
        while (queue.length) this.place(queue.shift(), this.cursor);
        if (!this.finished[this.cursor]) return;
        this.cursor++;
        this.gapPending = this.started;
      }
      this.complete = true;
    }

    place({ samples, rate }, index) {
      const context = this.context();
      const buffer = context.createBuffer(1, samples.length, rate);
      const channel = buffer.getChannelData(0);
      for (let i = 0; i < samples.length; i++) channel[i] = samples[i] / 32768;
      const start = this.length + (this.gapPending ? this.gap : 0);
      this.gapPending = false;
      const piece = { start, buffer };
      this.pieces.push(piece);
      this.length = start + buffer.duration;
      if (!this.started) {
        this.started = true;
        if (this.onStart) this.onStart(index);
      }
      if (this.playing) {
        const now = context.currentTime;
        // Playback ran out of audio: resume at this piece.
        if (this.origin + start < now + 0.02) {
          this.origin = now + 0.05 - start;
          this.from = start;
        }
        this.schedule(piece, start);
      }
    }

    schedule(piece, from) {
      const context = this.context();
      const end = piece.start + piece.buffer.duration;
      if (end <= from) return;
      const offset = Math.max(0, from - piece.start);
      const source = context.createBufferSource();
      source.buffer = piece.buffer;
      source.connect(context.destination);
      source.start(this.origin + piece.start + offset, offset);
      this.sources.add(source);
      source.onended = () => this.sources.delete(source);
    }

    /** The playback position, in seconds of the timeline. */
    get position() {
      if (!this.playing) return this.paused;
      const now = this.context().currentTime;
      // Not behind where it started while the first source is still due.
      return Math.min(Math.max(now - this.origin, this.from), this.length);
    }

    get ended() {
      return this.complete && this.started && this.position >= this.length - 0.01;
    }

    play() {
      if (this.disposed || this.playing) return;
      const context = this.context();
      let from = this.paused;
      if (this.complete && from >= this.length - 0.01) from = 0;
      this.origin = context.currentTime + 0.05 - from;
      this.from = from;
      this.playing = true;
      for (const piece of this.pieces) this.schedule(piece, from);
    }

    pause() {
      if (!this.playing) return;
      this.paused = this.position;
      this.playing = false;
      this.silence();
    }

    seek(seconds) {
      const target = Math.min(Math.max(seconds, 0), this.length);
      if (this.playing) {
        this.silence();
        this.playing = false;
        this.paused = target;
        this.play();
      } else {
        this.paused = target;
      }
    }

    /** At the end of the audio, stop and rewind for the next play. */
    tick() {
      if (this.playing && this.ended) {
        this.playing = false;
        this.paused = this.length;
        this.silence();
      }
    }

    silence() {
      for (const source of this.sources) {
        try { source.stop(); } catch (error) { /* Already ended. */ }
      }
      this.sources.clear();
    }

    dispose() {
      this.disposed = true;
      this.playing = false;
      this.silence();
    }
  }

  return {
    BASE,
    VOICE_DESIGN,
    CUSTOM_VOICE,
    voiceControls,
    paragraphModeAllowed,
    runTabs,
    languageModelName,
    voiceList,
    speechBody,
    paragraphs,
    speechChunks,
    TimeStretcher,
    Resampler,
    int16FromFloat,
    wavBytes,
    Segmenter,
    SentenceSplitter,
    TimelinePlayer,
  };
})();

if (typeof globalThis !== "undefined") globalThis.SplashLogic = SplashLogic;
