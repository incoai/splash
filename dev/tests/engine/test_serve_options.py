import io
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
    "--max-cache-disk": values(
        "--max-cache-disk",
        {"0": 0, "5G": 5 * 1024**3},
        ("auto", "-1", "0G", "5X", "nan"),
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
        {"3600": 3600.0, "0.5": 0.5},
        ("0", "-1", "inf", "nan", "soon"),
    ),
    "--queue-size": values(
        "--queue-size", {"64": 64, "1": 1}, ("0", "-1", "1.5", "many")
    ),
    "--api-key": values(
        "--api-key",
        {"secret-key": "secret-key"},
        ("", "two words", "key\n", "非ASCII"),
    ),
    "--no-webui": ([(["--no-webui"], True)], []),
}


def dest(flag):
    return flag.removeprefix("--").replace("-", "_")


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


if __name__ == "__main__":
    unittest.main()
