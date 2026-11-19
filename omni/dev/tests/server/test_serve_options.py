import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from install import launcher
from server import audio_runtime, audio_server, serve_options
from server.origins import ANY_ORIGIN

ROOT = Path(__file__).resolve().parents[3]
# Each parser, with the arguments it requires.
SERVER_ARGS = ["model", "--tokenizer", "tokenizer", "--model", "owner/repo"]
LAUNCHER_ARGS = ["serve", "--model", "owner/repo"]
PARSERS = ((audio_server.parse_args, SERVER_ARGS), (launcher.parse_args, LAUNCHER_ARGS))


def values(flag, accepted, refused, repeatable=False):
    """Arguments that give `flag` each accepted value, with the value both
    parsers parse it to, and arguments that give it each refused value."""
    return (
        [
            ([f"{flag}={text}"], [parsed] if repeatable else parsed)
            for text, parsed in accepted.items()
        ],
        [[f"{flag}={text}"] for text in refused],
    )


# Every shared option: accepted arguments with the value they parse to, and
# refused arguments.
OPTIONS = {
    "--host": values(
        "--host", {"0.0.0.0": "0.0.0.0", "mymac.local": "mymac.local"}, ()
    ),
    "--served-model-name": values(
        "--served-model-name",
        {"local": "local", "-local": "-local", "org/model": "org/model"},
        ("", "has space", "a?b", "a#b", "a%b", "a\\b", "/a", "a//b", "a/../b"),
        repeatable=True,
    ),
    "--announce-served-name": (
        [(["--announce-served-name", "--served-model-name=local"], True)],
        [["--announce-served-name"]],
    ),
    "--max-memory": values(
        "--max-memory",
        {
            "auto": None,
            "28G": 28 * 1024**3,
            "1GiB": 1024**3,
            "512mb": 512 * 1024**2,
            "1073741824": 1024**3,
        },
        ("0", "-1G", "bad", "G", str(2**64)),
    ),
    "--idle-release": values(
        "--idle-release",
        {"600": 600.0, "90s": 90.0, "30M": 1800.0, "1.5h": 5400.0, "off": math.inf},
        ("0", "-1m", "inf", "nan", "10x", "m", "never"),
    ),
    "--allowed-host": values(
        "--allowed-host", {"proxy.example": "proxy.example"}, (), repeatable=True
    ),
    "--allowed-origin": values(
        "--allowed-origin",
        {
            "tauri://localhost": ("tauri", "localhost", None),
            "http://[::1]:3000": ("http", "::1", 3000),
            "*": ANY_ORIGIN,
        },
        ("http://localhost/app", "tauri://*", "localhost"),
        repeatable=True,
    ),
    "--max-request-size": values(
        "--max-request-size",
        {"256M": 256 * 1024**2, "1G": 1024**3},
        ("auto", "0", "-1G", "bad", str(2**64)),
    ),
    "--request-timeout": values(
        "--request-timeout",
        {"3600": 3600.0, "0.5": 0.5, "90s": 90.0, "30M": 1800.0, "1.5h": 5400.0},
        ("0", "-1", "-1m", "inf", "nan", "10x", "m", "off", "soon"),
    ),
    "--queue-size": values(
        "--queue-size", {"64": 64, "1": 1}, ("0", "-1", "1.5", "many")
    ),
    "--lanes": values("--lanes", {"8": 8, "1": 1, "64": 64}, ("0", "65", "-1", "few")),
    "--disable-ane": (
        [([], False), (["--disable-ane"], True)],
        [["--disable-ane=yes"]],
    ),
    "--segment-seconds": values(
        "--segment-seconds", {"30": 30.0, "0": 0.0, "45.5": 45.5}, ("-1", "nan", "soon")
    ),
    # A directory (made absolute) or a repository ID.
    "--draft-model": values(
        "--draft-model",
        {
            str(ROOT): str(ROOT),
            "mlx-community/Qwen3-ASR-0.6B-4bit": "mlx-community/Qwen3-ASR-0.6B-4bit",
        },
        ("", "a/b/c", "no model", "owner/../repo"),
    ),
    # On by default for Qwen3-ASR-1.7B; it refuses a draft or its settings.
    "--no-draft-model": (
        [([], False), (["--no-draft-model"], True)],
        [
            ["--no-draft-model=yes"],
            ["--no-draft-model", "--draft-model=owner/draft"],
        ],
    ),
    # Each needs a draft: the default one, or --draft-model's.
    "--draft-tokens": (
        [
            ([f"--draft-tokens={text}"], tokens)
            for text, tokens in (("1", 1), ("15", 15))
        ]
        + [(["--draft-model=owner/draft", "--draft-tokens=4"], 4)],
        [
            *([f"--draft-tokens={text}"] for text in ("0", "16", "-1", "few")),
            ["--no-draft-model", "--draft-tokens=4"],
        ],
    ),
    "--speculative-lanes": (
        [
            ([f"--speculative-lanes={text}"], lanes)
            for text, lanes in (("1", 1), ("8", 8))
        ]
        + [(["--draft-model=owner/draft", "--speculative-lanes=2"], 2)],
        [
            *([f"--speculative-lanes={text}"] for text in ("0", "9", "few")),
            ["--no-draft-model", "--speculative-lanes=2"],
        ],
    ),
    "--api-key": values(
        "--api-key",
        {"secret-key": "secret-key"},
        ("", "two words", "key\n", "非ASCII"),
    ),
}


def dest(flag):
    return flag.removeprefix("--").replace("-", "_")


class ServeOptionsTests(unittest.TestCase):
    def setUp(self):
        # The parsers' defaults ignore the caller's Splash settings.
        self.enterContext(mock.patch.dict(os.environ))
        os.environ.pop("SPLASH_API_KEY", None)

    def refuse(self, parse, arguments):
        with (
            mock.patch("sys.stderr", io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as raised,
        ):
            parse(arguments)
        self.assertEqual(raised.exception.code, 2)
        return stderr.getvalue()

    def test_both_parsers_accept_and_reject_the_same_values(self):
        self.assertEqual(
            set(OPTIONS), {option.flag for option in serve_options.SERVE_OPTIONS}
        )
        defaults = [vars(parse(required)) for parse, required in PARSERS]
        for option in serve_options.SERVE_OPTIONS:
            self.assertEqual(
                defaults[0][option.dest], defaults[1][option.dest], option.flag
            )
        for flag, (accepted, refused) in OPTIONS.items():
            for (parse, required), (arguments, parsed) in (
                (parser, case) for parser in PARSERS for case in accepted
            ):
                with self.subTest(parser=parse.__module__, arguments=arguments):
                    self.assertEqual(
                        getattr(parse([*required, *arguments]), dest(flag)), parsed
                    )
            for (parse, required), arguments in (
                (parser, case) for parser in PARSERS for case in refused
            ):
                with self.subTest(parser=parse.__module__, arguments=arguments):
                    self.refuse(parse, [*required, *arguments])

    def test_both_parsers_check_defaults_from_the_environment(self):
        name, value, flag = "SPLASH_API_KEY", "environment-key", "--api-key"
        for parse, required in PARSERS:
            with (
                self.subTest(parser=parse.__module__, name=name),
                mock.patch.dict(os.environ, {name: value}),
            ):
                self.assertEqual(getattr(parse(required), dest(flag)), value)
                # The command line outranks the environment.
                explicit = OPTIONS[flag][0][0]
                self.assertEqual(
                    getattr(parse([*required, *explicit[0]]), dest(flag)),
                    explicit[1],
                )
            with (
                self.subTest(parser=parse.__module__, name=name, valid=False),
                mock.patch.dict(os.environ, {name: "two words"}),
            ):
                self.assertIn(flag, self.refuse(parse, required))

    def test_each_parse_has_its_own_lists(self):
        for parse, required in PARSERS:
            for option in serve_options.SERVE_OPTIONS:
                if option.options.get("action") != "append":
                    continue
                with self.subTest(parser=parse.__module__, flag=option.flag):
                    getattr(parse(required), option.dest).append("changed")
                    self.assertEqual(getattr(parse(required), option.dest), [])

    def test_launcher_forwards_exactly_what_the_server_parses(self):
        launched = [
            *(
                argument
                for flag, (accepted, _) in OPTIONS.items()
                # It contradicts the draft options (below).
                if flag != "--no-draft-model"
                for arguments, _ in accepted[-1:]
                for argument in arguments
            ),
            # A repeatable option keeps every value, in order.
            "--served-model-name=stable",
            "--allowed-host=proxy.local",
            "--allowed-origin=tauri://localhost",
        ]
        args = launcher.parse_args([*LAUNCHER_ARGS, *launched])
        argv = serve_options.serve_argv(args)
        environment = serve_options.serve_environment(args)
        # Every option but the key travels on the command line; the key only
        # in the environment.
        self.assertEqual(
            {argument.partition("=")[0] for argument in argv},
            {
                option.flag
                for option in serve_options.SERVE_OPTIONS
                if not option.secret and option.flag != "--no-draft-model"
            },
        )
        self.assertFalse(any("secret-key" in argument for argument in argv))
        self.assertEqual(environment, {"SPLASH_API_KEY": "secret-key"})
        with mock.patch.dict(os.environ, environment):
            served = audio_server.parse_args([*SERVER_ARGS, *argv])
        for option in serve_options.SERVE_OPTIONS:
            self.assertEqual(
                getattr(served, option.dest), getattr(args, option.dest), option.flag
            )
        plain = launcher.parse_args([*LAUNCHER_ARGS, "--no-draft-model"])
        self.assertEqual(serve_options.serve_argv(plain), ["--no-draft-model"])
        self.assertTrue(
            audio_server.parse_args(
                [*SERVER_ARGS, *serve_options.serve_argv(plain)]
            ).no_draft_model
        )
        # Defaults stay with the server, which reads them itself.
        defaults = launcher.parse_args(LAUNCHER_ARGS)
        self.assertEqual(serve_options.serve_argv(defaults), [])
        self.assertEqual(serve_options.serve_environment(defaults), {})

    def test_serve_options_imports_only_the_standard_library(self):
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "import server.serve_options; "
                "print(sorted({name.partition('.')[0] for name in sys.modules} "
                "- set(sys.stdlib_module_names)))",
                str(ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "['__main__', 'server']")

    def test_server_requires_explicit_model_and_paths(self):
        model = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"
        package = audio_server.ROOT / "install/models" / model
        required = [
            str(package),
            "--tokenizer",
            str(package / "tokenizer"),
            "--model",
            model,
        ]
        for arguments in (
            [],
            required[:-2],
            [*required[:-1], "qwen3-tts"],
            [*required[:-1], "Qwen3-TTS-12Hz-1.7B-Base-4bit"],
            [*required[:-1], "https://huggingface.co/community/model"],
            [*required[:-1], "community/../model"],
            # A GGUF variant, which no speech or transcription model has.
            [*required[:-1], "community/model:Q4_K_M"],
        ):
            with (
                self.subTest(arguments=arguments),
                mock.patch("sys.stderr"),
                self.assertRaises(SystemExit),
            ):
                audio_server.parse_args(arguments)
        args = audio_server.parse_args(required)
        self.assertEqual(Path(args.model_root), package)
        self.assertEqual(Path(args.tokenizer), package / "tokenizer")
        self.assertIsNone(args.request_timeout)
        self.assertEqual(args.model, model)
        self.assertEqual(Path(args.binary).name, "splash")
        with mock.patch("sys.stderr"):
            for value in ("-1", "65536"):
                with self.assertRaises(SystemExit):
                    audio_server.parse_args([*required, "--port", value])

    def test_native_command_passes_each_serve_option(self):
        model = "mlx-community/Qwen3-ASR-1.7B-4bit"
        package = audio_server.ROOT / "install/models" / model
        required = [
            str(package),
            "--tokenizer",
            str(package / "tokenizer"),
            "--model",
            model,
        ]
        args = audio_server.parse_args(required)
        # The engine's command line for each option it takes; the engine
        # keeps its own default of an option the server does not pass.
        for options, draft, arguments in (
            ([], None, []),
            (["--max-memory", "28G"], None, ["--max-memory", str(28 * 1024**3)]),
            (["--lanes", "4"], None, ["--lanes", "4"]),
            (["--disable-ane"], None, ["--disable-ane"]),
            (["--idle-release", "10m"], None, ["--idle-release", "600.0"]),
            (["--idle-release", "off"], None, ["--idle-release", "off"]),
            (
                ["--max-memory", "1G", "--lanes", "16", "--idle-release", "30s"],
                None,
                [
                    "--max-memory",
                    str(1024**3),
                    "--lanes",
                    "16",
                    "--idle-release",
                    "30.0",
                ],
            ),
            (
                ["--draft-tokens", "4", "--speculative-lanes", "2"],
                "/models/draft",
                [
                    "--draft-model",
                    "/models/draft",
                    "--draft-tokens",
                    "4",
                    "--speculative-lanes",
                    "2",
                ],
            ),
            # A draft's settings go only with a draft (the default one's
            # download may fail).
            (["--draft-tokens", "4", "--speculative-lanes", "2"], None, []),
            # Options the server alone reads.
            (
                ["--queue-size", "4", "--request-timeout", "30s", "--api-key", "k"],
                None,
                [],
            ),
        ):
            with self.subTest(options=options, draft=draft):
                self.assertEqual(
                    audio_server._native_command(
                        audio_server.parse_args([*required, *options]), draft
                    ),
                    [args.binary, "serve-omni", args.model_root, *arguments],
                )


class DraftModelTests(unittest.TestCase):
    """Which draft the server decodes with: the default DFlash draft beside a
    Qwen3-ASR-1.7B checkpoint unless --no-draft-model or --draft-model says
    otherwise, fetched into the Hub cache when missing."""

    DRAFT = serve_options.DEFAULT_DRAFT_MODEL

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # An empty Hub cache of the test's own: never the user's.
        self.cache = self.root / "hub"
        self.cache.mkdir()
        self.enterContext(
            mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(self.cache)})
        )
        self.status = self.enterContext(mock.patch.object(audio_server, "print_status"))
        self.download = self.enterContext(
            mock.patch.object(serve_options, "download_model")
        )

    def model(self, name, config):
        """A model directory holding config.json."""
        directory = self.root / name
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(config))
        return directory

    @staticmethod
    def asr_config(hidden_size=2048, quantized=True):
        config = {
            "model_type": "qwen3_asr",
            "thinker_config": {"text_config": {"hidden_size": hidden_size}},
        }
        if quantized:
            config["quantization_config"] = {"bits": 4, "group_size": 64}
        return config

    def cache_draft(self, repo_id=DRAFT, commit="c" * 40):
        """repo_id's snapshot in the test's Hub cache, as a download leaves it."""
        folder = self.cache / ("models--" + repo_id.replace("/", "--"))
        snapshot = folder / "snapshots" / commit
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        (folder / "refs").mkdir()
        (folder / "refs" / "main").write_text(commit)
        return str(snapshot)

    def draft(self, root, *options, profile=None):
        args = audio_server.parse_args(
            [str(root), "--tokenizer", str(root), "--model", "owner/repo", *options]
        )
        if profile is None:
            profile = audio_server.ModelProfile(
                "qwen3-asr" if "asr" in root.name else "qwen3-tts", ()
            )
        return audio_server._draft_directory(args, profile)

    def statuses(self):
        return [call.args[0] for call in self.status.call_args_list]

    def test_default_draft_names_only_qwen3_asr_1_7b(self):
        for name, config, expected in (
            ("asr-4bit", self.asr_config(), self.DRAFT),
            ("asr-bf16", self.asr_config(quantized=False), self.DRAFT),
            # The engine reads a decoder without hidden_size as the 1.7B's.
            (
                "asr-unsized",
                {"model_type": "qwen3_asr", "thinker_config": {"text_config": {}}},
                self.DRAFT,
            ),
            # The 0.6B itself, speech, and what is not a model.
            ("asr-0.6b", self.asr_config(hidden_size=1024), None),
            (
                "tts",
                {"model_type": "qwen3_tts", "thinker_config": {"text_config": {}}},
                None,
            ),
            ("odd", {"model_type": "qwen3_asr", "thinker_config": []}, None),
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    serve_options.default_draft_model(self.model(name, config)),
                    expected,
                )
        self.assertIsNone(serve_options.default_draft_model(self.root / "missing"))

    def test_qwen3_asr_1_7b_takes_the_cached_default_draft(self):
        snapshot = self.cache_draft()
        for quantized in (True, False):
            with self.subTest(quantized=quantized):
                root = self.model(
                    f"asr-{quantized}", self.asr_config(quantized=quantized)
                )
                self.assertEqual(self.draft(root), snapshot)
        self.download.assert_not_called()
        self.assertIn(
            f"Draft · {self.DRAFT} · speculative transcription by default; "
            "--no-draft-model turns it off",
            self.statuses(),
        )

    def test_no_draft_model_serves_plainly(self):
        self.cache_draft()
        root = self.model("asr", self.asr_config())
        self.assertIsNone(self.draft(root, "--no-draft-model"))
        self.download.assert_not_called()
        self.status.assert_not_called()

    def test_an_explicit_draft_replaces_the_default(self):
        root = self.model("asr", self.asr_config())
        other = self.model("other-draft", self.asr_config(hidden_size=1024))
        self.assertEqual(self.draft(root, "--draft-model", str(other)), str(other))
        snapshot = self.cache_draft("owner/draft")
        self.assertEqual(self.draft(root, "--draft-model", "owner/draft"), snapshot)
        # Another transcription model takes an explicit draft too.
        small = self.model("asr-0.6b", self.asr_config(hidden_size=1024))
        self.assertEqual(self.draft(small, "--draft-model", str(other)), str(other))
        self.download.assert_not_called()
        self.assertIn(
            "Draft · owner/draft · speculative transcription", self.statuses()
        )

    def test_speech_and_other_models_have_no_draft(self):
        self.cache_draft()
        tts = self.model("tts", {"model_type": "qwen3_tts"})
        small = self.model("asr-0.6b", self.asr_config(hidden_size=1024))
        for root in (tts, small):
            with self.subTest(root=root.name):
                self.assertIsNone(self.draft(root))
                # A draft's settings need a draft.
                for option in ("--draft-tokens=4", "--speculative-lanes=2"):
                    with self.assertRaisesRegex(
                        audio_runtime.EngineUnhealthy, "needs a draft model"
                    ):
                        self.draft(root, option)
        with self.assertRaisesRegex(
            audio_runtime.EngineUnhealthy, "takes a transcription model"
        ):
            self.draft(tts, "--draft-model", str(small))
        self.download.assert_not_called()
        self.status.assert_not_called()

    def test_a_missing_draft_is_downloaded(self):
        root = self.model("asr", self.asr_config())
        self.download.return_value = "/downloaded/draft"
        self.assertEqual(self.draft(root), "/downloaded/draft")
        self.download.assert_called_once_with(self.DRAFT)
        statuses = self.statuses()
        self.assertIn(
            f"Downloading · draft {self.DRAFT} for speculative transcription "
            "(about 0.7 GB); --no-draft-model skips it",
            statuses,
        )
        self.assertTrue(statuses[-1].startswith(f"Draft · {self.DRAFT}"))
        # An explicit repository ID is downloaded the same way.
        self.download.reset_mock()
        self.assertEqual(
            self.draft(root, "--draft-model", "owner/draft"), "/downloaded/draft"
        )
        self.download.assert_called_once_with("owner/draft")

    def test_a_failed_download_serves_without_the_default_draft(self):
        root = self.model("asr", self.asr_config())
        for error in (
            OSError("Network is unreachable"),
            RuntimeError("offline mode is enabled\nand the file is not cached"),
        ):
            with self.subTest(error=type(error).__name__):
                self.status.reset_mock()
                self.download.side_effect = error
                self.assertIsNone(
                    self.draft(root, "--draft-tokens=4", "--speculative-lanes=2")
                )
                (warning,) = (
                    call.args[0]
                    for call in self.status.call_args_list
                    if call.kwargs.get("error")
                )
                self.assertIn(f"cannot download draft {self.DRAFT}", warning)
                self.assertIn(" ".join(str(error).split()), warning)
                self.assertIn("without speculative decoding", warning)
                # An explicit draft that cannot be had ends the startup.
                with self.assertRaisesRegex(
                    audio_runtime.EngineUnhealthy, "cannot download draft model"
                ):
                    self.draft(root, "--draft-model", "owner/draft")

    def test_the_download_is_huggingface_hubs_snapshot_download(self):
        with mock.patch(
            "huggingface_hub.snapshot_download", return_value="/snapshot"
        ) as download:
            self.assertEqual(DOWNLOAD_MODEL(self.DRAFT), "/snapshot")
        download.assert_called_once_with(self.DRAFT)

    def test_the_cached_snapshot_is_the_one_refs_main_names(self):
        root = self.model("asr", self.asr_config())
        folder = self.cache / ("models--" + self.DRAFT.replace("/", "--"))
        for commit in ("a" * 40, "b" * 40):
            (folder / "snapshots" / commit).mkdir(parents=True)
            (folder / "snapshots" / commit / "config.json").write_text("{}")
        (folder / "refs").mkdir()
        (folder / "refs" / "main").write_text("a" * 40)
        self.assertEqual(self.draft(root), str(folder / "snapshots" / ("a" * 40)))
        # Without refs/main, the newest snapshot.
        (folder / "refs" / "main").unlink()
        os.utime(folder / "snapshots" / ("a" * 40), (1, 1))
        self.assertEqual(self.draft(root), str(folder / "snapshots" / ("b" * 40)))
        self.download.assert_not_called()


# serve_options.download_model itself, before DraftModelTests replaces it.
DOWNLOAD_MODEL = serve_options.download_model


if __name__ == "__main__":
    unittest.main()
