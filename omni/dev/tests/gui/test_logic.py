"""The page's DOM-free logic (gui/static/logic.js), run in JavaScriptCore,
the engine Safari uses: what the page offers for each speech model type, the
bodies of its speech requests (paragraph mode included), the one-track
player that replaced the stream player and the audio element, and the live
transcription page's resampling, segmentation and WAV encoding."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

LOGIC = Path(__file__).resolve().parents[3] / "gui/static/logic.js"
JSC = shutil.which("jsc") or next(
    (
        path
        for path in (
            "/System/Library/Frameworks/JavaScriptCore.framework/Versions/Current/Helpers/jsc",
        )
        if Path(path).is_file()
    ),
    None,
)

# A Web Audio context whose clock the test moves, recording every source.
FAKE_AUDIO = r"""
function FakeContext() {
  this.currentTime = 0;
  this.sources = [];
}
FakeContext.prototype.createBuffer = function (channels, length, rate) {
  return { length, sampleRate: rate, duration: length / rate,
    getChannelData() { return new Float32Array(length); } };
};
FakeContext.prototype.createBufferSource = function () {
  const context = this;
  const source = {
    buffer: null, when: null, offset: 0, stopped: null, onended: null,
    connect() {},
    start(when, offset) { this.when = when; this.offset = offset || 0; context.sources.push(this); },
    stop() { if (this.stopped === null) this.stopped = context.currentTime; },
  };
  return source;
};
FakeContext.prototype.destination = {};
// The sources sounding at time t.
FakeContext.prototype.audible = function (t) {
  return this.sources.filter((s) => s.when <= t && t < s.when + s.buffer.duration - s.offset &&
    (s.stopped === null || s.stopped > t)).length;
};
FakeContext.prototype.most = function (from, to) {
  let most = 0;
  for (let t = from; t <= to; t += 0.005) most = Math.max(most, this.audible(t));
  return most;
};
const context = new FakeContext();
const samples = (seconds, rate = 24000) => new Int16Array(Math.round(seconds * rate)).fill(1000);
"""


@unittest.skipUnless(JSC, "JavaScriptCore's jsc is needed to run the page's logic")
class LogicTests(unittest.TestCase):
    def run_js(self, script):
        """The JSON that `script` prints, run after logic.js."""
        with tempfile.NamedTemporaryFile("w", suffix=".js") as test:
            test.write(FAKE_AUDIO + script)
            test.flush()
            result = subprocess.run(
                [JSC, str(LOGIC), test.name],
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        lines = result.stdout.strip().splitlines()
        self.assertTrue(lines, result.stderr)
        return json.loads(lines[-1])

    # What the page offers for each model type.

    def test_controls_follow_the_models_listing(self):
        found = self.run_js(
            r"""
            const L = SplashLogic;
            const listing = {
              base: { id: "b", task: "text-to-speech", tts_model_type: "base", instructions: "unsupported" },
              design: { id: "d", task: "text-to-speech", tts_model_type: "voice_design", instructions: "required" },
              custom: { id: "c", task: "text-to-speech", tts_model_type: "custom_voice", instructions: "optional",
                        voices: ["aiden", "ryan", "vivian"] },
              other: { id: "x", object: "model" },
            };
            const out = {};
            for (const [name, model] of Object.entries(listing)) out[name] = L.voiceControls(model);
            out.none = L.voiceControls(null);
            out.voiceList = [L.voiceList([{ id: "default" }]), L.voiceList({ voices: ["a", { name: "b" }] }),
              L.voiceList({ data: [{ voice_id: "c" }] }), L.voiceList("nonsense")];
            print(JSON.stringify(out));
            """
        )
        base, design, custom = found["base"], found["design"], found["custom"]
        self.assertEqual(base["instructions"], "unsupported")
        self.assertIsNone(base["voices"])
        self.assertFalse(base["paragraphs"])
        self.assertFalse(base["freeVoice"])
        self.assertEqual(design["instructions"], "required")
        self.assertTrue(design["paragraphs"])
        self.assertIsNone(design["voices"])
        self.assertEqual(custom["instructions"], "optional")
        self.assertEqual(custom["voices"], ["aiden", "ryan", "vivian"])
        self.assertTrue(custom["voiceRequired"])
        self.assertTrue(custom["paragraphs"])
        # Another framework's server: generic controls, paragraph mode kept.
        for other in (found["other"], found["none"]):
            self.assertIsNone(other["type"])
            self.assertEqual(other["instructions"], "optional")
            self.assertTrue(other["freeVoice"])
            self.assertTrue(other["paragraphs"])
        self.assertEqual(found["voiceList"], [["default"], ["a", "b"], ["c"], []])

    def test_speech_bodies_carry_what_the_model_takes(self):
        found = self.run_js(
            r"""
            const L = SplashLogic;
            const parameters = { language: "english", response_format: "wav", stream: true, voice: "stale" };
            const kinds = {
              base: L.voiceControls({ tts_model_type: "base", instructions: "unsupported" }),
              design: L.voiceControls({ tts_model_type: "voice_design", instructions: "required" }),
              custom: L.voiceControls({ tts_model_type: "custom_voice", instructions: "optional", voices: ["ryan"] }),
              other: L.voiceControls(null),
            };
            const body = (kind, instructions, voice) => {
              try {
                return L.speechBody({ input: "Hi.", parameters, controls: kinds[kind], instructions, voice,
                  seed: 7, model: "m" });
              } catch (error) {
                return { error: error.message };
              }
            };
            print(JSON.stringify({
              base: body("base", "A calm voice", "ryan"),
              designMissing: body("design", "   ", ""),
              design: body("design", "  A calm, low male voice.  ", "ryan"),
              customMissing: body("custom", "Happily", ""),
              custom: body("custom", "", "ryan"),
              customStyled: body("custom", "Speak happily.", "ryan"),
              other: body("other", "Cheerful", "alloy"),
              otherPlain: body("other", "", ""),
            }));
            """
        )
        base = found["base"]
        self.assertNotIn("instructions", base)
        self.assertNotIn("voice", base)
        self.assertEqual(
            {
                key: base[key]
                for key in ("input", "model", "seed", "language", "stream")
            },
            {
                "input": "Hi.",
                "model": "m",
                "seed": 7,
                "language": "english",
                "stream": True,
            },
        )
        self.assertIn("system prompt", found["designMissing"]["error"])
        self.assertEqual(found["design"]["instructions"], "A calm, low male voice.")
        self.assertNotIn("voice", found["design"])
        self.assertIn("speaker", found["customMissing"]["error"])
        self.assertEqual(found["custom"]["voice"], "ryan")
        self.assertNotIn("instructions", found["custom"])
        self.assertEqual(
            (found["customStyled"]["voice"], found["customStyled"]["instructions"]),
            ("ryan", "Speak happily."),
        )
        self.assertEqual(
            (found["other"]["voice"], found["other"]["instructions"]),
            ("alloy", "Cheerful"),
        )
        self.assertNotIn("voice", found["otherPlain"])
        self.assertNotIn("instructions", found["otherPlain"])

    def test_language_models_are_named_without_their_repositories(self):
        found = self.run_js(
            """
            print(JSON.stringify([
              "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M",
              "mlx-community/Qwen3.8-27B-4bit",
              "someone/Plain-Model",
            ].map((id) => SplashLogic.languageModelName(id))));
            """
        )
        self.assertEqual(
            found,
            [
                "Qwen3.6-35B-A3B · GGUF Q4_K_M",
                "Qwen3.8-27B · MLX 4-bit",
                "Plain-Model · MLX",
            ],
        )

    def test_a_server_opens_only_the_tabs_of_its_task(self):
        found = self.run_js(
            """
            const L = SplashLogic;
            print(JSON.stringify(["tts", "asr", null, "other"].map((task) => L.runTabs(task))));
            """
        )
        self.assertEqual(
            found,
            [
                ["tts"],
                ["asr", "live"],
                ["tts", "asr", "live"],
                ["tts", "asr", "live"],
            ],
        )

    def test_paragraph_mode_sends_one_voice_to_every_segment(self):
        found = self.run_js(
            r"""
            const L = SplashLogic;
            const text = "First paragraph.\n\nSecond one,\nstill second.\n  \t\nThird.";
            const parts = L.paragraphs(text);
            const run = (controls, instructions, voice) => parts.map((input) => L.speechBody({
              input, parameters: { stream: true }, controls, instructions, voice, seed: 42 }));
            print(JSON.stringify({
              parts,
              allowed: ["base", "voice_design", "custom_voice", null].map((type) => L.paragraphModeAllowed(type)),
              referenced: ["base", "voice_design", "custom_voice", null].map((type) =>
                L.voiceControls({ tts_model_type: type, references: true }).paragraphs),
              flags: [L.voiceControls({ tts_model_type: "base", references: true }).references,
                L.voiceControls({ tts_model_type: "base" }).references],
              design: run(L.voiceControls({ tts_model_type: "voice_design" }), "A warm narrator.", ""),
              custom: run(L.voiceControls({ tts_model_type: "custom_voice", voices: ["vivian"] }), "Calmly.", "vivian"),
            }));
            """
        )
        self.assertEqual(
            found["parts"],
            ["First paragraph.", "Second one,\nstill second.", "Third."],
        )
        # Base cannot keep one voice across independent requests: no
        # paragraph mode, unless the server continues the first paragraph.
        self.assertEqual(found["allowed"], [False, True, True, True])
        self.assertEqual(found["referenced"], [True, True, True, True])
        self.assertEqual(found["flags"], [True, False])
        for name, expected in (
            ("design", {"instructions": "A warm narrator.", "seed": 42}),
            ("custom", {"instructions": "Calmly.", "voice": "vivian", "seed": 42}),
        ):
            with self.subTest(model_type=name):
                bodies = found[name]
                self.assertEqual([body["input"] for body in bodies], found["parts"])
                for body in bodies:
                    for key, value in expected.items():
                        self.assertEqual(body[key], value)
                self.assertEqual(
                    {
                        json.dumps({k: v for k, v in b.items() if k != "input"})
                        for b in bodies
                    },
                    {json.dumps({k: v for k, v in bodies[0].items() if k != "input"})},
                )

    # The one-track player.

    def test_finished_audio_plays_as_one_track(self):
        found = self.run_js(
            r"""
            const player = new SplashLogic.TimelinePlayer(1, { context: () => context });
            player.play();                       // play as it arrives
            player.push(0, samples(0.5), 24000);
            context.currentTime = 0.1;
            player.push(0, samples(0.5), 24000);
            context.currentTime = 0.2;
            player.finish(0);                    // generation done, still playing
            const whilePlaying = context.sources.length;
            player.play();                       // a second play is no second track
            const afterSecondPlay = context.sources.length;
            context.currentTime = 1.3;
            player.tick();                       // the end
            const ended = !player.playing;
            player.play();                       // play again: from the start
            context.currentTime = 1.6;
            const position = player.position;
            player.pause();
            context.currentTime = 1.8;
            const before = context.sources.length;
            player.play();                       // resume
            const resumed = context.sources.slice(before).map((s) => s.offset);
            context.currentTime = 1.9;
            player.seek(0.1);                    // seek while playing
            print(JSON.stringify({
              whilePlaying, afterSecondPlay, ended, position,
              most: context.most(0, 4), length: player.length,
              resumed,
            }));
            """
        )
        self.assertEqual(found["whilePlaying"], 2)
        self.assertEqual(found["afterSecondPlay"], 2)
        self.assertTrue(found["ended"])
        self.assertAlmostEqual(found["length"], 1.0)
        self.assertAlmostEqual(found["position"], 0.25, places=2)
        self.assertAlmostEqual(found["resumed"][0], 0.25, places=2)
        # Never two sources at once: the double playback is gone.
        self.assertEqual(found["most"], 1)

    def test_playback_waits_for_audio_and_keeps_segment_order(self):
        found = self.run_js(
            r"""
            const player = new SplashLogic.TimelinePlayer(2, { context: () => context, gap: 0.3 });
            player.play();
            player.push(1, samples(0.4), 24000);  // the second paragraph first
            const before = player.length;
            player.push(0, samples(0.2), 24000);
            context.currentTime = 0.5;            // ran out of audio at 0.25
            const stalled = player.position;
            player.push(0, samples(0.2), 24000);
            const resumedAt = context.sources[context.sources.length - 1].when;
            context.currentTime = 0.6;
            const moving = player.position;
            player.finish(0);
            player.finish(1);
            print(JSON.stringify({
              before, stalled, resumedAt, moving,
              starts: player.pieces.map((piece) => Math.round(piece.start * 100) / 100),
              length: player.length, complete: player.complete,
              most: context.most(0, 3),
            }));
            """
        )
        self.assertEqual(found["before"], 0)
        self.assertAlmostEqual(found["stalled"], 0.2, places=3)
        self.assertAlmostEqual(found["resumedAt"], 0.55, places=3)
        self.assertAlmostEqual(found["moving"], 0.25, places=3)
        self.assertEqual(found["starts"], [0, 0.2, 0.7])
        self.assertAlmostEqual(found["length"], 1.1, places=3)
        self.assertTrue(found["complete"])
        self.assertEqual(found["most"], 1)

    def test_a_whole_file_waits_for_play(self):
        found = self.run_js(
            r"""
            const player = new SplashLogic.TimelinePlayer(1, { context: () => context });
            player.push(0, samples(1.0), 24000);  // not streamed, not autoplayed
            player.finish(0);
            const silent = context.sources.length;
            context.currentTime = 2;
            player.play();
            context.currentTime = 2.5;
            const position = player.position;
            player.dispose();
            print(JSON.stringify({ silent, position, sources: context.sources.length,
              stopped: context.sources.every((s) => s.stopped !== null), most: context.most(0, 4) }));
            """
        )
        self.assertEqual(found["silent"], 0)
        self.assertAlmostEqual(found["position"], 0.45, places=3)
        self.assertEqual(found["sources"], 1)
        self.assertTrue(found["stopped"])
        self.assertEqual(found["most"], 1)

    # Live transcription.

    SIGNAL = r"""
    const rate = 16000;
    function signal(parts) {
      const total = parts.reduce((sum, [seconds]) => sum + Math.round(seconds * rate), 0);
      const out = new Float32Array(total);
      let offset = 0, seed = 1;
      const noise = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };
      for (const [seconds, amplitude] of parts) {
        const n = Math.round(seconds * rate);
        for (let i = 0; i < n; i++) {
          out[offset + i] = amplitude * Math.sin(2 * Math.PI * 220 * i / rate) + 0.002 * noise();
        }
        offset += n;
      }
      return out;
    }
    function feed(segmenter, data, chunk) {
      const events = [];
      for (let i = 0; i < data.length; i += chunk) {
        for (const event of segmenter.push(data.subarray(i, i + chunk))) events.push(event);
      }
      for (const event of segmenter.flush()) events.push(event);
      return events.map((e) => ({ type: e.type, id: e.id, reason: e.reason,
        seconds: e.seconds === undefined ? undefined : Math.round(e.seconds * 100) / 100,
        start: e.start === undefined ? (e.at === undefined ? undefined : Math.round(e.at * 100) / 100)
                                      : Math.round(e.start * 100) / 100 }));
    }
    """

    def test_segmenter_cuts_utterances_at_pauses(self):
        found = self.run_js(
            self.SIGNAL
            + r"""
            const data = signal([[0.6, 0], [1.2, 0.3], [1.0, 0], [0.04, 0.3], [1.0, 0],
                                 [2.5, 0.2], [1.0, 0]]);
            const options = { silenceMs: 600, partialMs: 1000, preRollMs: 300, minSpeechMs: 200 };
            const whole = feed(new SplashLogic.Segmenter(options), data, data.length);
            const chunked = feed(new SplashLogic.Segmenter(options), data, 1365);
            print(JSON.stringify({ whole, chunked }));
            """
        )
        events = found["chunked"]
        self.assertEqual(events, found["whole"])
        finals = [event for event in events if event["type"] == "final"]
        self.assertEqual([event["reason"] for event in finals], ["silence", "silence"])
        # Speech at 0.6 s with 0.3 s of pre-roll; 0.2 s of the pause kept.
        self.assertAlmostEqual(finals[0]["start"], 0.3, places=1)
        self.assertAlmostEqual(finals[0]["seconds"], 0.3 + 1.2 + 0.2, delta=0.06)
        self.assertAlmostEqual(finals[1]["seconds"], 0.3 + 2.5 + 0.2, delta=0.06)
        # The 40 ms click is dropped, not transcribed.
        drops = [event for event in events if event["type"] == "drop"]
        self.assertEqual(len(drops), 1)
        # Partials about once a second of utterance, for the long one twice.
        partials = [event for event in events if event["type"] == "partial"]
        self.assertEqual(
            [event["id"] for event in partials],
            [finals[0]["id"], finals[1]["id"], finals[1]["id"]],
        )
        self.assertEqual([event["seconds"] for event in partials], [1.0, 1.0, 2.0])

    def test_segmenter_cuts_long_speech_and_flushes_at_stop(self):
        found = self.run_js(
            self.SIGNAL
            + r"""
            const data = signal([[0.5, 0], [7.0, 0.3]]);
            const events = feed(new SplashLogic.Segmenter({ maxSeconds: 3, preRollMs: 0 }), data, 4096);
            const loud = new SplashLogic.Segmenter({ thresholdDb: -10 });
            print(JSON.stringify({ events, loud: feed(loud, data, 4096), threshold: loud.threshold }));
            """
        )
        finals = [event for event in found["events"] if event["type"] == "final"]
        self.assertEqual([event["reason"] for event in finals], ["max", "max", "flush"])
        self.assertEqual([event["seconds"] for event in finals], [3.0, 3.0, 1.0])
        # A fixed threshold above the speech hears nothing.
        self.assertEqual(found["loud"], [])
        self.assertEqual(found["threshold"], -10)

    def test_resampler_and_wav(self):
        found = self.run_js(
            r"""
            const L = SplashLogic;
            const tone = (frequency, rate, seconds) => {
              const out = new Float32Array(Math.round(rate * seconds));
              for (let i = 0; i < out.length; i++) out[i] = 0.5 * Math.sin(2 * Math.PI * frequency * i / rate);
              return out;
            };
            const rms = (x, from) => Math.sqrt(x.subarray(from).reduce((s, v) => s + v * v, 0) / (x.length - from));
            const crossings = (x) => { let n = 0; for (let i = 1; i < x.length; i++) if (x[i - 1] < 0 && x[i] >= 0) n++; return n; };
            const out = {};
            for (const rate of [48000, 44100]) {
              const input = tone(440, rate, 1);
              const whole = new L.Resampler(rate).push(input);
              const resampler = new L.Resampler(rate);
              const parts = [];
              for (let i = 0; i < input.length; i += 4096) parts.push(...resampler.push(input.subarray(i, i + 4096)));
              let difference = 0;
              for (let i = 0; i < whole.length; i++) difference = Math.max(difference, Math.abs(whole[i] - parts[i]));
              out[rate] = { length: whole.length, chunked: parts.length, difference,
                crossings: crossings(whole), rms: rms(whole, 100),
                aliased: rms(new L.Resampler(rate).push(tone(12000, rate, 1)), 100) };
            }
            const bytes = L.wavBytes([new Int16Array([1, -2, 3])], 16000);
            const view = new DataView(bytes.buffer);
            const tag = (o) => String.fromCharCode(...bytes.subarray(o, o + 4));
            out.wav = { riff: tag(0), wave: tag(8), size: view.getUint32(4, true), format: view.getUint16(20, true),
              channels: view.getUint16(22, true), rate: view.getUint32(24, true), bits: view.getUint16(34, true),
              data: view.getUint32(40, true), first: view.getInt16(44, true), second: view.getInt16(46, true),
              length: bytes.length };
            print(JSON.stringify(out));
            """
        )
        for rate in ("48000", "44100"):
            with self.subTest(rate=rate):
                result = found[rate]
                self.assertLessEqual(abs(result["length"] - 16000), 2)
                self.assertLessEqual(abs(result["chunked"] - result["length"]), 1)
                self.assertLess(result["difference"], 1e-5)
                self.assertLessEqual(abs(result["crossings"] - 440), 1)
                self.assertAlmostEqual(result["rms"], 0.5 / 2**0.5, delta=0.02)
                # 12 kHz is above the 16 kHz output's Nyquist frequency.
                self.assertLess(result["aliased"], 0.02)
        self.assertEqual(
            found["wav"],
            {
                "riff": "RIFF",
                "wave": "WAVE",
                "size": 42,
                "format": 1,
                "channels": 1,
                "rate": 16000,
                "bits": 16,
                "data": 6,
                "first": 1,
                "second": -2,
                "length": 50,
            },
        )

    # Voice chat: a streamed reply spoken sentence by sentence.

    def test_sentence_splitter_cuts_a_stream_into_speakable_pieces(self):
        found = self.run_js(
            r"""
            const L = SplashLogic;
            const run = (text, step, options) => {
              const splitter = new L.SentenceSplitter(options);
              const pieces = [];
              for (let i = 0; i < text.length; i += step) pieces.push(...splitter.push(text.slice(i, i + step)));
              pieces.push(...splitter.flush());
              return pieces;
            };
            const english = "Sure! The **museum** opens at 9 a.m. and costs 3.50 dollars. " +
              "Ok. It closes at six; the cafe stays open later.\nEnjoy 🎉";
            const chinese = "你好！博物馆早上九点开门，门票三块五。晚上六点关门；咖啡馆开得更晚。";
            const long = "When the weather is good, which it usually is in the late spring, the park fills with families, and the lake is busy.";
            print(JSON.stringify({
              english: run(english, 3),
              englishWhole: run(english, 1000),
              chinese: run(chinese, 2),
              long: run(long, 5),
              markdown: L.SentenceSplitter.clean("## Plan\n- **first** `step` [link](http://x)"),
            }));
            """
        )
        self.assertEqual(
            found["english"],
            [
                "Sure!",
                "The museum opens at 9 a.m. and costs 3.50 dollars.",
                # Too short alone: joined to the next sentence.
                "Ok. It closes at six; the cafe stays open later.",
                "Enjoy",
            ],
        )
        self.assertEqual(found["englishWhole"], found["english"])
        self.assertEqual(
            found["chinese"],
            [
                "你好！",
                "博物馆早上九点开门，门票三块五。",
                "晚上六点关门；咖啡馆开得更晚。",
            ],
        )
        # The first piece is cut at a comma once long enough; later ones are not.
        self.assertEqual(
            found["long"],
            [
                "When the weather is good, which it usually is in the late spring,",
                "the park fills with families, and the lake is busy.",
            ],
        )
        self.assertEqual(found["markdown"], "Plan first step link")

    def test_long_text_is_cut_into_chunks_at_sentence_ends(self):
        found = self.run_js(
            r"""
            const L = SplashLogic;
            const english = "Every autumn the small fishing village holds a festival to celebrate the end of the season. Boats are decorated with lanterns, children run between stalls selling roasted chestnuts, and the oldest sailors tell stories about storms they survived decades ago. Visitors come from the nearby towns, and for one evening the quiet harbor becomes the busiest place on the coast. Learning a musical instrument as an adult is slower than it is for children, but it has its own rewards. Adults tend to practice with more purpose, they understand why certain exercises matter, and they often enjoy the music they choose to play. The hardest part is usually patience: progress comes in small steps, and the first few months can feel frustrating before everything starts to fall into place. The engineers tested the bridge by driving loaded trucks across it one at a time while sensors recorded how much the steel beams bent. The measurements matched their calculations closely, and only one joint needed extra reinforcement. A week later the city opened the bridge to traffic, and commuters on the east side saved almost twenty minutes each morning. When the library introduced evening hours, nobody expected how popular they would become. Students used the quiet tables to study after work, retired neighbors joined a weekly reading circle, and parents brought their children for storytelling on Thursday nights. Within a year the evening visitors outnumbered those who came during the day, and the library hired two more staff members.";
            const chinese = "每年秋天，这个小渔村都会举办一个节日来庆祝捕鱼季的结束。渔船上挂满了灯笼，孩子们在摊位之间跑来跑去，买烤栗子吃，年纪最大的老水手们讲述几十年前经历过的风暴。附近城镇的游客也纷纷赶来，在这一个晚上，安静的港口变成了海岸上最热闹的地方。成年人学习乐器比孩子慢，但也有自己的收获。成年人练习时目的更明确，明白每个练习的意义，也往往更喜欢自己选择演奏的音乐。最难的通常是耐心：进步是一点一点来的，最初几个月可能会让人感到沮丧，然后一切才慢慢变得顺利起来。工程师们让装满货物的卡车一辆接一辆地驶过大桥，同时用传感器记录钢梁弯曲的程度。测量结果与他们的计算非常接近，只有一个连接处需要额外加固。一周以后，城市正式开放了这座桥，住在东边的上班族每天早上能节省将近二十分钟。图书馆开始在晚上开放时，没有人想到会这么受欢迎。下班后的学生在安静的桌子旁学习，退休的邻居们参加每周一次的读书会，父母们在周四晚上带孩子来听故事。不到一年，晚上来的读者就超过了白天，图书馆又新招了两名工作人员。";
            print(JSON.stringify({
              english: L.speechChunks(english),
              chinese: L.speechChunks(chinese),
              short: L.speechChunks("  One short paragraph. Two sentences.  "),
              weights: L.speechChunks(english).map((c) => L.SentenceSplitter.weight(c)),
            }));
            """
        )
        english = "Every autumn the small fishing village holds a festival to celebrate the end of the season. Boats are decorated with lanterns, children run between stalls selling roasted chestnuts, and the oldest sailors tell stories about storms they survived decades ago. Visitors come from the nearby towns, and for one evening the quiet harbor becomes the busiest place on the coast. Learning a musical instrument as an adult is slower than it is for children, but it has its own rewards. Adults tend to practice with more purpose, they understand why certain exercises matter, and they often enjoy the music they choose to play. The hardest part is usually patience: progress comes in small steps, and the first few months can feel frustrating before everything starts to fall into place. The engineers tested the bridge by driving loaded trucks across it one at a time while sensors recorded how much the steel beams bent. The measurements matched their calculations closely, and only one joint needed extra reinforcement. A week later the city opened the bridge to traffic, and commuters on the east side saved almost twenty minutes each morning. When the library introduced evening hours, nobody expected how popular they would become. Students used the quiet tables to study after work, retired neighbors joined a weekly reading circle, and parents brought their children for storytelling on Thursday nights. Within a year the evening visitors outnumbered those who came during the day, and the library hired two more staff members."
        chinese = "每年秋天，这个小渔村都会举办一个节日来庆祝捕鱼季的结束。渔船上挂满了灯笼，孩子们在摊位之间跑来跑去，买烤栗子吃，年纪最大的老水手们讲述几十年前经历过的风暴。附近城镇的游客也纷纷赶来，在这一个晚上，安静的港口变成了海岸上最热闹的地方。成年人学习乐器比孩子慢，但也有自己的收获。成年人练习时目的更明确，明白每个练习的意义，也往往更喜欢自己选择演奏的音乐。最难的通常是耐心：进步是一点一点来的，最初几个月可能会让人感到沮丧，然后一切才慢慢变得顺利起来。工程师们让装满货物的卡车一辆接一辆地驶过大桥，同时用传感器记录钢梁弯曲的程度。测量结果与他们的计算非常接近，只有一个连接处需要额外加固。一周以后，城市正式开放了这座桥，住在东边的上班族每天早上能节省将近二十分钟。图书馆开始在晚上开放时，没有人想到会这么受欢迎。下班后的学生在安静的桌子旁学习，退休的邻居们参加每周一次的读书会，父母们在周四晚上带孩子来听故事。不到一年，晚上来的读者就超过了白天，图书馆又新招了两名工作人员。"
        self.assertEqual(found["short"], ["One short paragraph. Two sentences."])
        self.assertGreater(len(found["english"]), 3)
        self.assertTrue(all(weight <= 360 for weight in found["weights"]))
        self.assertEqual(" ".join(found["english"]), english)
        self.assertTrue(all(chunk.endswith(".") for chunk in found["english"]))
        self.assertGreater(len(found["chinese"]), 2)
        self.assertEqual("".join(found["chinese"]), chinese)
        self.assertTrue(all(chunk.endswith("。") for chunk in found["chinese"]))

    def test_time_stretcher_keeps_pitch_and_level_at_another_speed(self):
        found = self.run_js(
            r"""
            const L = SplashLogic;
            const rate = 24000;
            const tone = new Int16Array(rate * 2);
            for (let i = 0; i < tone.length; i++) {
              // Two harmonics, the second's level drifting, like a voice.
              tone[i] = Math.round(8000 * Math.sin(2 * Math.PI * 180 * i / rate) +
                3000 * (1 + Math.sin(2 * Math.PI * 1.5 * i / rate)) * Math.sin(2 * Math.PI * 360 * i / rate));
            }
            const stretch = (factor, step) => {
              const s = new L.TimeStretcher(rate, factor);
              const parts = [];
              for (let i = 0; i < tone.length; i += step) parts.push(...s.push(tone.subarray(i, i + step)));
              parts.push(...s.flush());
              return Int16Array.from(parts);
            };
            const crossings = (x) => { let n = 0; for (let i = 1; i < x.length; i++) if (x[i - 1] < 0 && x[i] >= 0) n++; return n; };
            const rms = (x) => Math.sqrt(x.reduce((s, v) => s + v * v, 0) / x.length);
            const envelope = (x) => {
              const out = [];
              for (let i = 0; i + 480 <= x.length; i += 480) out.push(rms(x.subarray(i, i + 480)));
              return out;
            };
            const out = {};
            for (const factor of [1.5, 2]) {
              const whole = stretch(factor, tone.length);
              const chunked = stretch(factor, 997);
              let same = whole.length === chunked.length;
              for (let i = 0; same && i < whole.length; i++) same = whole[i] === chunked[i];
              const middle = whole.subarray(2400, whole.length - 2400);
              const env = envelope(middle);
              out[factor] = {
                length: whole.length,
                same,
                // Zero crossings per second, input and output.
                rateIn: crossings(tone) / (tone.length / rate),
                rateOut: crossings(middle) / (middle.length / rate),
                rmsIn: rms(tone.subarray(2400, tone.length - 2400)),
                rmsOut: rms(middle),
                dip: Math.min(...env) / Math.max(...env),
              };
            }
            out.bypass = stretch(1, 1000).length === tone.length;
            print(JSON.stringify(out));
            """
        )
        self.assertTrue(found["bypass"])
        for factor in ("1.5", "2"):
            with self.subTest(factor=factor):
                result = found[factor]
                self.assertTrue(result["same"])
                self.assertAlmostEqual(
                    result["length"], 48000 / float(factor), delta=1000
                )
                self.assertAlmostEqual(result["rateOut"], result["rateIn"], delta=4)
                self.assertAlmostEqual(
                    result["rmsOut"] / result["rmsIn"], 1.0, delta=0.08
                )
                # No beating from overlaps added out of phase.
                self.assertGreater(result["dip"], 0.5)


if __name__ == "__main__":
    unittest.main()
