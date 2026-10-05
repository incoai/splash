import io
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from install import launcher
from server import images, serve_options
from server import server as api
from server.origins import ANY_ORIGIN

ROOT = Path(__file__).resolve().parents[3]
# Each parser, with the arguments it requires.
SERVER_ARGS = ["model", "--tokenizer", "tokenizer", "--model", "owner/repo"]
LAUNCHER_ARGS = ["serve", "--model", "owner/repo"]
PARSERS = ((api.parse_args, SERVER_ARGS), (launcher.parse_args, LAUNCHER_ARGS))


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
    "--default-reasoning-effort": values(
        "--default-reasoning-effort",
        {effort: effort for effort in serve_options.REASONING_EFFORTS},
        ("", "turbo", "XHIGH"),
    ),
    "--kv-format": values("--kv-format", {"int8": "int8", "bf16": "bf16"}, ("fp16",)),
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
    "--max-cache-disk": values(
        "--max-cache-disk",
        {"0": 0, "5G": 5 * 1024**3},
        ("auto", "-1", "0G", "5X", "nan"),
    ),
    "--persistent-cache": (
        [(["--persistent-cache", "--max-cache-disk=5G"], True)],
        [["--persistent-cache"], ["--persistent-cache", "--max-cache-disk=0"]],
    ),
    "--cache-dir": (
        [
            (
                ["--cache-dir=/srv/cache", "--persistent-cache", "--max-cache-disk=5G"],
                Path("/srv/cache"),
            ),
            (
                ["--cache-dir=~/cache", "--persistent-cache", "--max-cache-disk=5G"],
                Path.home() / "cache",
            ),
        ],
        [
            ["--cache-dir=/srv/cache"],
            ["--cache-dir=", "--persistent-cache", "--max-cache-disk=5G"],
        ],
    ),
    "--max-context": values(
        "--max-context",
        {"auto": None, "100K": 102400, "256k": 262144, "262144": 262144, "1": 1},
        ("0", "-1", "257K", "262145", "bad"),
    ),
    "--decode-share": values(
        "--decode-share",
        {"0": 0.0, "0.25": 0.25, "2": 2.0},
        ("-0.5", "inf", "nan", "half"),
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
    "--max-image-pixels": values(
        "--max-image-pixels",
        {str(images.MIN_PIXELS): images.MIN_PIXELS, "1048576": 1048576},
        (str(images.MIN_PIXELS - 1), str(images.MAX_PIXELS + 1), "-1", "many"),
    ),
    "--request-timeout": values(
        "--request-timeout",
        {"3600": 3600.0, "0.5": 0.5, "90s": 90.0, "30M": 1800.0, "1.5h": 5400.0},
        ("0", "-1", "-1m", "inf", "nan", "10x", "m", "off", "soon"),
    ),
    "--queue-size": values(
        "--queue-size", {"64": 64, "1": 1}, ("0", "-1", "1.5", "many")
    ),
    "--api-key": values(
        "--api-key",
        {"secret-key": "secret-key"},
        ("", "two words", "key\n", "非ASCII"),
    ),
    "--allow-idle-sleep": ([(["--allow-idle-sleep"], True)], []),
    "--no-webui": ([(["--no-webui"], True)], []),
}


def dest(flag):
    return flag.removeprefix("--").replace("-", "_")


def native_command_lines():
    """(name, serve options, serve-native arguments after the model directory)
    of each line of the command lines both sides read."""
    golden = api.ROOT / "dev/tests/engine/native_command_golden.txt"
    for line in golden.read_text().splitlines():
        if line and not line.startswith("#"):
            name, options, arguments = line.split("|")
            yield name.strip(), options.split(), arguments.split()


class ServeOptionsTests(unittest.TestCase):
    def setUp(self):
        # The parsers' defaults ignore the caller's Splash settings.
        self.enterContext(mock.patch.dict(os.environ))
        for name in ("SPLASH_API_KEY", "SPLASH_DEFAULT_REASONING_EFFORT"):
            os.environ.pop(name, None)

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
        for name, value, flag in (
            ("SPLASH_DEFAULT_REASONING_EFFORT", "low", "--default-reasoning-effort"),
            ("SPLASH_API_KEY", "environment-key", "--api-key"),
        ):
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
                for accepted, _ in OPTIONS.values()
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
                if not option.secret
            },
        )
        self.assertFalse(any("secret-key" in argument for argument in argv))
        self.assertEqual(environment, {"SPLASH_API_KEY": "secret-key"})
        with mock.patch.dict(os.environ, environment):
            served = api.parse_args([*SERVER_ARGS, *argv])
        for option in serve_options.SERVE_OPTIONS:
            self.assertEqual(
                getattr(served, option.dest), getattr(args, option.dest), option.flag
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
        model = "community/custom-splash"
        package = api.ROOT / "install/models" / model
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
            [*required[:-1], "qwen3.8-27b"],
            [*required[:-1], "Qwen3.8-27B"],
            [*required[:-1], "https://huggingface.co/community/model"],
            [*required[:-1], "community/../model"],
        ):
            with (
                self.subTest(arguments=arguments),
                mock.patch("sys.stderr"),
                self.assertRaises(SystemExit),
            ):
                api.parse_args(arguments)
        args = api.parse_args(required)
        self.assertEqual(Path(args.model_root), package)
        self.assertEqual(Path(args.tokenizer), package / "tokenizer")
        self.assertIsNone(args.request_timeout)
        self.assertEqual(args.model, model)
        self.assertEqual(Path(args.binary).name, "splash")
        with mock.patch("sys.stderr"):
            for value in ("-1", "65536"):
                with self.assertRaises(SystemExit):
                    api.parse_args([*required, "--port", value])

    def test_native_command_passes_each_serve_option(self):
        model = "community/custom-splash"
        package = api.ROOT / "install/models" / model
        required = [
            str(package),
            "--tokenizer",
            str(package / "tokenizer"),
            "--model",
            model,
        ]
        args = api.parse_args(required)
        # The engine's command line for each option, which the engine reads as
        # the server means it (native_arguments_test.cpp). The engine keeps
        # its own default of an option the server does not pass, and the
        # per-image patch limit is the most patches a resized image within
        # the pixel cap can have, a whole number of merge units.
        for name, options, arguments in native_command_lines():
            with self.subTest(name=name):
                self.assertEqual(
                    api._native_command(api.parse_args([*required, *options])),
                    [args.binary, "serve-native", args.model_root, *arguments],
                )
        # A persistent cache names its directory; the server picks the default.
        persistent_args = api.parse_args(
            [*required, "--max-cache-disk", "5G", "--persistent-cache"]
        )
        self.assertEqual(
            api._native_command(persistent_args)[-2:],
            ["--cache-dir", str(serve_options.DEFAULT_CACHE_DIR)],
        )


if __name__ == "__main__":
    unittest.main()
