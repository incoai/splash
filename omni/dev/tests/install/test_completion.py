import os
import pty
import select
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
COMPLETIONS = REPO / "install/completions"
SUGGESTED = ("suggested/Model-4bit", "suggested/Model-bf16")
# Installed models: a selection link to an assembly, and one through a
# linked directory.
LOCAL = ("community/custom-model", "community/linked-model")
UPSTREAM = ("mlx-community/Model-4bit",)
MODELS = sorted((*SUGGESTED, *LOCAL, *UPSTREAM))


def bash_paths():
    candidates = (
        "/bin/bash",
        "/opt/homebrew/bin/bash",
        "/usr/local/bin/bash",
        os.environ.get("SPLASH_TEST_BASH", ""),
    )
    return tuple(dict.fromkeys(path for path in candidates if os.access(path, os.X_OK)))


FISH = next(
    (
        path
        for path in (
            os.environ.get("SPLASH_TEST_FISH", ""),
            shutil.which("fish") or "",
            "/opt/homebrew/bin/fish",
            "/usr/local/bin/fish",
        )
        if path and os.access(path, os.X_OK)
    ),
    None,
)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="splash completion ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.home = self.root / "user home"
        self.home.mkdir()
        blocked = self.root / "blocked commands"
        blocked.mkdir()
        self.executed = self.root / "unexpected-command"
        for name in (
            "python",
            "python3",
            "python3.12",
            "python3.13",
            "python3.14",
            "curl",
            "wget",
            "uv",
            "splash-omni",
        ):
            executable = blocked / name
            executable.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$0" >> "$UNEXPECTED_COMMAND"\nexit 97\n'
            )
            executable.chmod(0o755)
        self.env = {
            **os.environ,
            "HOME": str(self.home),
            "ZDOTDIR": str(self.home),
            "PATH": f"{blocked}:/usr/bin:/bin",
            "TERM": "xterm",
            "UNEXPECTED_COMMAND": str(self.executed),
        }
        self.env.pop("BASH_ENV", None)
        self.env.pop("ENV", None)
        self.addCleanup(self.assert_no_runtime)

    def assert_no_runtime(self):
        self.assertFalse(
            self.executed.exists(),
            self.executed.read_text() if self.executed.exists() else "",
        )

    def layout(self, name="source tree"):
        root = self.root / name
        directory = root / "install/completions"
        shutil.copytree(COMPLETIONS, directory)
        (directory / "suggested-models.txt").write_text("\n".join(SUGGESTED) + "\n")
        models = root / "install/models"
        model_root = models / LOCAL[0]
        model_root.mkdir(parents=True, exist_ok=True)
        (model_root / "model.json").write_text("{}")
        linked = self.root / (name + " external model")
        linked.mkdir()
        (linked / "model.json").write_text("{}")
        (models / LOCAL[1]).unlink(missing_ok=True)
        (models / LOCAL[1]).symlink_to(linked, target_is_directory=True)
        (models / "community/broken-link").unlink(missing_ok=True)
        (models / "community/broken-link").symlink_to(self.root / "missing")
        (models / "community/incomplete").mkdir(exist_ok=True)
        (models / "community/directory-record/model.json").mkdir(
            parents=True, exist_ok=True
        )
        # A legacy Splash package, which this release does not serve.
        (models / "incoai/Qwen3.8-27B-Splash").mkdir(parents=True, exist_ok=True)
        (models / "incoai/Qwen3.8-27B-Splash/manifest.json").write_text("{}")
        # Installations link assemblies that record model.json.
        for model in UPSTREAM:
            assembly = self.root / (name + " assembly " + model.replace("/", " "))
            assembly.mkdir()
            (assembly / "model.json").write_text("{}")
            (models / model).parent.mkdir(parents=True, exist_ok=True)
            (models / model).unlink(missing_ok=True)
            (models / model).symlink_to(assembly, target_is_directory=True)
        # A selection root (.selections/<sha256>) records model.json too, but
        # it is a hidden installation, not a model id.
        selection = self.root / (name + " selection assembly")
        selection.mkdir()
        (selection / "model.json").write_text("{}")
        (models / ".selections").mkdir(exist_ok=True)
        (models / ".selections/0123").unlink(missing_ok=True)
        (models / ".selections/0123").symlink_to(selection, target_is_directory=True)
        for invalid in (
            "bad owner/model",
            "community/bad--name",
            "community/bad..name",
            "community/model:variant",
            "community/model:a:b",
        ):
            path = models / invalid
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.json").write_text("{}")
        return root, directory

    def run_helper(self, directory, prefix=""):
        result = subprocess.run(
            [str(directory / "models"), prefix],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        self.assertEqual(result.stderr, "")
        return result.stdout.splitlines()

    def bash_complete(self, shell, script, words, *, upgrade=None, unbroken=""):
        setup = ""
        if upgrade:
            link, new, old = upgrade
            setup = (
                f"/bin/rm {shlex.quote(str(link))}\n"
                f"/bin/ln -s {shlex.quote(str(new))} {shlex.quote(str(link))}\n"
                f"/bin/rm -rf {shlex.quote(str(old))}\n"
            )
        if unbroken:
            setup += f"COMP_WORDBREAKS=${{COMP_WORDBREAKS//[{unbroken}]/}}\n"
        program = (
            'source "$1"\nshift\n' + setup + "registration=$(complete -p splash-omni)\n"
            "function=${registration##* -F }\nfunction=${function%% *}\n"
            'COMP_WORDS=("$@")\nCOMP_CWORD=$((${#COMP_WORDS[@]} - 1))\n'
            'COMP_LINE="${COMP_WORDS[*]}"\nCOMP_POINT=${#COMP_LINE}\n'
            '"$function" splash-omni "${COMP_WORDS[COMP_CWORD]}" '
            '"${COMP_WORDS[COMP_CWORD-1]}"\n'
            'if ((${#COMPREPLY[@]})); then printf "%s\\0" "${COMPREPLY[@]}"; fi\n'
        )
        result = subprocess.run(
            [
                shell,
                "--noprofile",
                "--norc",
                "-c",
                program,
                "completion",
                str(script),
                *words,
            ],
            env=self.env,
            capture_output=True,
            timeout=10,
            check=True,
        )
        self.assertEqual(result.stderr, b"")
        return result.stdout.decode().split("\0")[:-1]

    def fish_complete(self, script, line, *paths):
        """The candidates fish offers for line after running script, which
        reads line as $argv[1] and paths from $argv[2]."""
        result = subprocess.run(
            [FISH, "--no-config", "-c", script, line, *map(str, paths)],
            env={
                **self.env,
                "XDG_CONFIG_HOME": str(self.home / ".config"),
                "XDG_DATA_HOME": str(self.home / ".local/share"),
                "XDG_CACHE_HOME": str(self.home / ".cache"),
            },
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        self.assertEqual(result.stderr, "")
        return sorted(
            candidate.split("\t")[0] for candidate in result.stdout.splitlines()
        )

    def test_model_helper_is_offline(self):
        _, directory = self.layout()
        self.assertEqual(self.run_helper(directory), MODELS)
        self.assertEqual(self.run_helper(directory, "suggested/"), list(SUGGESTED))
        self.assertEqual(self.run_helper(directory, "community/l"), [LOCAL[1]])
        self.assertEqual(self.run_helper(directory, "community/*"), [])
        self.assertEqual(self.run_helper(directory, "["), [])
        self.assertEqual(self.run_helper(directory, "incoai/"), [])

    def test_the_bundled_list_names_the_supported_models(self):
        from install import families

        self.assertEqual(
            (COMPLETIONS / "suggested-models.txt").read_text().split(),
            sorted(families.SUPPORTED_MODELS),
        )
        self.assertFalse((COMPLETIONS / "official-models.txt").exists())

    def test_actual_bash_completion(self):
        _, directory = self.layout()
        cases = (
            (["splash-omni", ""], ["serve", "ui"]),
            (["splash-omni", "se"], ["serve"]),
            (["splash-omni", "u"], ["ui"]),
            (["splash-omni", "ui", "--model", ""], []),
            (["splash-omni", "co"], []),
            (["splash-omni", "serve", "--model", ""], MODELS),
            (["splash-omni", "serve", "--model", "community/l"], [LOCAL[1]]),
            (["splash-omni", "serve", "--model=community/l"], [LOCAL[1]]),
            (["splash-omni", "serve", "--model", "=", "community/l"], [LOCAL[1]]),
            (["splash-omni", "serve", "--model", "="], MODELS),
            (["splash-omni", "serve", "--lanes", ""], []),
            (["splash-omni", "serve", "--lanes", "community/l"], []),
            (["splash-omni", "serve", "community/l"], []),
        )
        # Words as they are split once these characters leave COMP_WORDBREAKS.
        unbroken_cases = (
            (
                ["splash-omni", "serve", "--model=community/l"],
                "=",
                [f"--model={LOCAL[1]}"],
            ),
            (["splash-omni", "serve", "--model", "community/l"], ":", [LOCAL[1]]),
        )
        for shell in bash_paths():
            for words, expected in cases:
                with self.subTest(shell=shell, words=words):
                    self.assertEqual(
                        self.bash_complete(
                            shell, directory / "splash-omni.bash", words
                        ),
                        expected,
                    )
            for words, unbroken, expected in unbroken_cases:
                with self.subTest(shell=shell, words=words, unbroken=unbroken):
                    self.assertEqual(
                        self.bash_complete(
                            shell,
                            directory / "splash-omni.bash",
                            words,
                            unbroken=unbroken,
                        ),
                        expected,
                    )
            # The coding clients the language server connected are no
            # commands.
            for agent in ("claude", "codex", "opencode", "hermes", "pi"):
                with self.subTest(shell=shell, agent=agent):
                    self.assertEqual(
                        self.bash_complete(
                            shell,
                            directory / "splash-omni.bash",
                            ["splash-omni", agent, "--model", ""],
                        ),
                        [],
                    )

    def shell_tab(
        self,
        script,
        line,
        *,
        shell="/bin/zsh",
        upgrade=None,
        autoload=False,
        unbroken="",
    ):
        zsh = shell == "/bin/zsh"
        capture = self.root / "shell captured result"
        capture.unlink(missing_ok=True)
        # The capture is read as soon as it exists, so each writer renames
        # its complete result into place.
        publish = '> "$CAPTURE.part" && mv "$CAPTURE.part" "$CAPTURE"'
        environment = {
            **self.env,
            "COMPLETION_SCRIPT": str(script),
            "COMPLETION_DIR": str(script.parent),
            "CAPTURE": str(capture),
        }
        pid, master = pty.fork()
        if pid == 0:
            os.chdir(self.root)
            arguments = (
                [shell, "-f", "-i"] if zsh else [shell, "--noprofile", "--norc", "-i"]
            )
            os.execve(shell, arguments, environment)
        output = bytearray()
        try:
            if zsh:
                setup = (
                    'fpath=("$COMPLETION_DIR" $fpath); autoload -Uz compinit; compinit -D -u; '
                    if autoload
                    else 'autoload -Uz compinit; compinit -D -u; source "$COMPLETION_SCRIPT"; '
                )
                setup += (
                    'function capture_buffer() { print -r -- "$BUFFER" '
                    + publish
                    + "; }; "
                    "zle -N capture_buffer; bindkey '^X' capture_buffer; "
                    "print -r -- COMPLETION_READY\n"
                )
                keys = b"\t\x18"
            else:
                # Enter runs this test-only argv recorder, never the real launcher.
                recorder = self.root / "splash-omni"
                recorder.write_text('#!/bin/sh\nprintf "%s\\n" "$@" ' + publish + "\n")
                recorder.chmod(0o755)
                setup = (
                    'source "$COMPLETION_SCRIPT"; '
                    'splash-omni() { printf "%s\\n" "$@" ' + publish + "; }; "
                )
                if unbroken:
                    setup += f"COMP_WORDBREAKS=${{COMP_WORDBREAKS//[{unbroken}]/}}; "
                setup += 'printf "COMPLETION_READY\\n"\n'
                keys = b"\t\n"
            # Typing waits for the prompt: keys that arrive before the line
            # editor takes the terminal go through its line discipline instead.
            os.write(master, b"PS1='[splash-test] '; " + setup.encode())
            deadline = time.monotonic() + 15
            ready = b"COMPLETION_READY\r\n"
            while ready not in output or (
                b"[splash-test] " not in output[output.index(ready) :]
            ):
                self.assertLess(
                    time.monotonic(), deadline, output.decode(errors="replace")
                )
                if select.select([master], [], [], 0.1)[0]:
                    output.extend(os.read(master, 65536))
            if upgrade:
                # Load the autoloaded function before replacing its installation.
                os.write(master, line.encode() + keys)
                while not capture.exists():
                    self.assertLess(
                        time.monotonic(), deadline, output.decode(errors="replace")
                    )
                    if select.select([master], [], [], 0.1)[0]:
                        output.extend(os.read(master, 65536))
                capture.unlink()
                os.write(master, b"\x15")
                link, new, old = upgrade
                link.unlink()
                link.symlink_to(new, target_is_directory=True)
                shutil.rmtree(old)
            os.write(master, line.encode() + keys)
            while not capture.exists():
                self.assertLess(
                    time.monotonic(), deadline, output.decode(errors="replace")
                )
                if select.select([master], [], [], 0.1)[0]:
                    output.extend(os.read(master, 65536))
            return capture.read_text().rstrip("\n")
        finally:
            os.close(master)
            try:
                os.kill(pid, signal.SIGHUP)
            except ProcessLookupError:
                pass
            os.waitpid(pid, 0)

    @unittest.skipUnless(os.access("/bin/zsh", os.X_OK), "Zsh is not installed")
    def test_actual_zsh_tab(self):
        _, directory = self.layout()
        for line, expected in (
            ("splash-omni se", "splash-omni serve "),
            ("splash-omni u", "splash-omni ui "),
            ("splash-omni p", "splash-omni p"),
            (
                "splash-omni serve --model community/l",
                f"splash-omni serve --model {LOCAL[1]} ",
            ),
            (
                "splash-omni serve --model=community/l",
                f"splash-omni serve --model={LOCAL[1]} ",
            ),
            (
                "splash-omni serve --model suggested/Model-4",
                "splash-omni serve --model suggested/Model-4bit ",
            ),
            (
                "splash-omni claude --model community/l",
                "splash-omni claude --model community/l",
            ),
            (
                "splash-omni pi --model community/l",
                "splash-omni pi --model community/l",
            ),
        ):
            with self.subTest(line=line):
                self.assertEqual(
                    self.shell_tab(directory / "_splash-omni", line), expected
                )

    def test_actual_bash_tab_wordbreaks(self):
        _, directory = self.layout()
        for shell in bash_paths():
            for unbroken in ("", "=", ":"):
                for command, option in (
                    ("splash-omni", "--model="),
                    ("splash-omni", "--model "),
                    ("./splash-omni", "--model="),
                ):
                    for typed, model in (
                        ("community/l", LOCAL[1]),
                        ("suggested/Model-b", SUGGESTED[1]),
                    ):
                        with self.subTest(
                            shell=shell,
                            unbroken=unbroken,
                            command=command,
                            option=option,
                            typed=typed,
                        ):
                            actual = self.shell_tab(
                                directory / "splash-omni.bash",
                                f"{command} serve {option}{typed}",
                                shell=shell,
                                unbroken=unbroken,
                            )
                            expected = (
                                ["serve", "--model=" + model]
                                if option.endswith("=")
                                else ["serve", "--model", model]
                            )
                            self.assertEqual(actual.splitlines(), expected)

    @unittest.skipUnless(FISH, "fish is not installed")
    def test_actual_fish_completion(self):
        _, directory = self.layout()
        cases = (
            ("splash-omni ", ["serve", "ui"]),
            ("splash-omni se", ["serve"]),
            ("splash-omni u", ["ui"]),
            ("splash-omni co", []),
            ("splash-omni serve --model ", MODELS),
            ("splash-omni serve --model community/l", [LOCAL[1]]),
            ("splash-omni serve --model=community/l", [f"--model={LOCAL[1]}"]),
            ("splash-omni serve --model suggested/", list(SUGGESTED)),
            ("splash-omni serve --port 8001 --model community/l", [LOCAL[1]]),
            ("splash-omni serve --lanes ", []),
            # Neither model IDs as arguments nor option names, as in Bash.
            ("splash-omni serve ", []),
            ("splash-omni serve -", []),
            # A command only as the first word, and serve's --model only
            # after serve as the first word.
            ("splash-omni --version s", []),
            ("splash-omni opencode serve --model ", []),
            *(
                (f"splash-omni {agent} --model ", [])
                for agent in ("claude", "codex", "opencode", "hermes", "pi")
            ),
        )
        for line, expected in cases:
            with self.subTest(line=line):
                self.assertEqual(
                    self.fish_complete(
                        "source $argv[2]; complete -C $argv[1]",
                        line,
                        directory / "splash-omni.fish",
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
