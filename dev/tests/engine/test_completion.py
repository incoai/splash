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
OFFICIAL = ("official/Model-A", "official/Model-B")
SUGGESTED = ("suggested/Model-4bit",)
LOCAL = ("community/custom-splash", "community/linked-splash")
GGUF = ("unsloth/Model-GGUF:Q8_0", "unsloth/Model-GGUF:UD-Q4_K_M")
UPSTREAM = (*GGUF, "mlx-community/Model-4bit")


def bash_paths():
    candidates = (
        "/bin/bash",
        "/opt/homebrew/bin/bash",
        "/usr/local/bin/bash",
        os.environ.get("SPLASH_TEST_BASH", ""),
    )
    return tuple(dict.fromkeys(path for path in candidates if os.access(path, os.X_OK)))


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
            "splash",
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

    def layout(self, name="source tree", *, release=False):
        root = self.root / name
        directory = root / "install/completions"
        shutil.copytree(COMPLETIONS, directory)
        (directory / "official-models.txt").write_text("\n".join(OFFICIAL) + "\n")
        (directory / "suggested-models.txt").write_text("\n".join(SUGGESTED) + "\n")
        if release:
            (root / "release.json").write_text("{}")
            models = self.home / "Library/Application Support/Splash/models"
        else:
            models = root / "install/models"
        for model in (*OFFICIAL, LOCAL[0]):
            model_root = models / model
            model_root.mkdir(parents=True, exist_ok=True)
            (model_root / "manifest.json").write_text("{}")
        linked = self.root / (name + " external model")
        linked.mkdir()
        (linked / "manifest.json").write_text("{}")
        (models / LOCAL[1]).unlink(missing_ok=True)
        (models / LOCAL[1]).symlink_to(linked, target_is_directory=True)
        (models / "community/broken-link").unlink(missing_ok=True)
        (models / "community/broken-link").symlink_to(self.root / "missing")
        (models / "community/incomplete").mkdir(exist_ok=True)
        (models / "community/directory-manifest/manifest.json").mkdir(
            parents=True, exist_ok=True
        )
        # Upstream installations link assemblies that record model.json.
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
            "community/model:bad variant",
            "community/model:a:b",
        ):
            path = models / invalid
            path.mkdir(parents=True, exist_ok=True)
            (path / "manifest.json").write_text("{}")
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
            'source "$1"\nshift\n' + setup + "registration=$(complete -p splash)\n"
            "function=${registration##* -F }\nfunction=${function%% *}\n"
            'COMP_WORDS=("$@")\nCOMP_CWORD=$((${#COMP_WORDS[@]} - 1))\n'
            'COMP_LINE="${COMP_WORDS[*]}"\nCOMP_POINT=${#COMP_LINE}\n'
            '"$function" splash "${COMP_WORDS[COMP_CWORD]}" '
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

    def test_model_helper_source_and_release_are_offline(self):
        for release in (False, True):
            with self.subTest(release=release):
                _, directory = self.layout(str(release), release=release)
                self.assertEqual(
                    self.run_helper(directory),
                    sorted((*OFFICIAL, *SUGGESTED, *LOCAL, *UPSTREAM)),
                )
                self.assertEqual(self.run_helper(directory, "unsloth/"), list(GGUF))
                self.assertEqual(self.run_helper(directory, "community/l"), [LOCAL[1]])
                self.assertEqual(self.run_helper(directory, "community/*"), [])
                self.assertEqual(self.run_helper(directory, "["), [])

    def test_cached_catalog_adds_valid_entries_in_source_and_release(self):
        for release in (False, True):
            with self.subTest(release=release):
                root, directory = self.layout(str(release), release=release)
                data = (
                    self.home / "Library/Application Support/Splash"
                    if release
                    else root / "build/runtime"
                )
                cache = data / "catalog/official-models.txt"
                cache.parent.mkdir(parents=True)
                cache.write_text(
                    "official/New\n../outside\ninvalid\nofficial/Model-A\n"
                )
                self.assertEqual(
                    self.run_helper(directory),
                    sorted((*OFFICIAL, *SUGGESTED, *LOCAL, *UPSTREAM, "official/New")),
                )
                self.assertEqual(
                    self.run_helper(directory, "official/N"), ["official/New"]
                )
                cache.write_text("invalid\n")
                self.assertEqual(
                    self.run_helper(directory),
                    sorted((*OFFICIAL, *SUGGESTED, *LOCAL, *UPSTREAM)),
                )

    def test_actual_bash_completion(self):
        _, directory = self.layout()
        cases = (
            (
                ["splash", ""],
                ["serve", "claude", "codex", "opencode", "hermes", "pi"],
            ),
            (["splash", "co"], ["codex"]),
            (
                ["splash", "serve", "--model", ""],
                sorted((*OFFICIAL, *SUGGESTED, *LOCAL, *UPSTREAM)),
            ),
            (["splash", "serve", "--model", "community/l"], [LOCAL[1]]),
            (["splash", "serve", "--model=community/l"], [LOCAL[1]]),
            (["splash", "serve", "--model", "=", "community/l"], [LOCAL[1]]),
            (
                ["splash", "serve", "--model", "="],
                sorted((*OFFICIAL, *SUGGESTED, *LOCAL, *UPSTREAM)),
            ),
            (["splash", "serve", "--", "--model", ""], []),
            (["splash", "serve", "--max-context", ""], []),
            # Bash 3.2 keeps owner/repo:VARIANT in one word, and readline
            # replaces only the text after its ':'.
            (["splash", "serve", "--model", "unsloth/Model-GGUF:UD"], ["UD-Q4_K_M"]),
            (["splash", "serve", "--model=unsloth/Model-GGUF:UD"], ["UD-Q4_K_M"]),
            # Bash 4 and later also split the word at the ':'.
            (
                ["splash", "serve", "--model", "unsloth/Model-GGUF", ":", "UD"],
                ["UD-Q4_K_M"],
            ),
            (
                ["splash", "serve", "--model", "unsloth/Model-GGUF", ":"],
                ["Q8_0", "UD-Q4_K_M"],
            ),
            (
                ["splash", "serve", "--model", "=", "unsloth/Model-GGUF", ":", "UD"],
                ["UD-Q4_K_M"],
            ),
            (
                ["splash", "serve", "--max-context", "unsloth/Model-GGUF", ":", "UD"],
                [],
            ),
            (["splash", "serve", "unsloth/Model-GGUF", ":"], []),
        )
        # Words as they are split once these characters leave COMP_WORDBREAKS.
        unbroken_cases = (
            (["splash", "serve", "--model=community/l"], "=", [f"--model={LOCAL[1]}"]),
            (
                ["splash", "serve", "--model=unsloth/Model-GGUF", ":", "UD"],
                "=",
                ["UD-Q4_K_M"],
            ),
            (
                ["splash", "serve", "--model", "unsloth/Model-GGUF:UD"],
                ":",
                ["unsloth/Model-GGUF:UD-Q4_K_M"],
            ),
            (
                ["splash", "serve", "--model=unsloth/Model-GGUF:UD"],
                "=:",
                ["--model=unsloth/Model-GGUF:UD-Q4_K_M"],
            ),
        )
        for shell in bash_paths():
            for words, expected in cases:
                with self.subTest(shell=shell, words=words):
                    self.assertEqual(
                        self.bash_complete(shell, directory / "splash.bash", words),
                        expected,
                    )
            for words, unbroken, expected in unbroken_cases:
                with self.subTest(shell=shell, words=words, unbroken=unbroken):
                    self.assertEqual(
                        self.bash_complete(
                            shell, directory / "splash.bash", words, unbroken=unbroken
                        ),
                        expected,
                    )
            for agent in ("claude", "codex", "opencode", "hermes", "pi"):
                with self.subTest(shell=shell, agent=agent):
                    self.assertEqual(
                        self.bash_complete(
                            shell,
                            directory / "splash.bash",
                            ["splash", agent, "--model", ""],
                        ),
                        [],
                    )

    def test_bash_loaded_completion_survives_release_upgrade(self):
        for index, shell in enumerate(bash_paths()):
            with self.subTest(shell=shell):
                old, _ = self.layout(f"old-{index}", release=True)
                new = self.root / f"new-{index}"
                shutil.copytree(old, new)
                link = self.root / f"current-{index}"
                link.symlink_to(old, target_is_directory=True)
                self.assertEqual(
                    self.bash_complete(
                        shell,
                        link / "install/completions/splash.bash",
                        ["splash", "serve", "--model", "community/l"],
                        upgrade=(link, new, old),
                    ),
                    [LOCAL[1]],
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
                recorder = self.root / "splash"
                recorder.write_text('#!/bin/sh\nprintf "%s\\n" "$@" ' + publish + "\n")
                recorder.chmod(0o755)
                setup = (
                    'source "$COMPLETION_SCRIPT"; '
                    'splash() { printf "%s\\n" "$@" ' + publish + "; }; "
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
            ("splash se", "splash serve "),
            ("splash p", "splash pi "),
            ("splash serve --model community/l", f"splash serve --model {LOCAL[1]} "),
            ("splash serve --model=community/l", f"splash serve --model={LOCAL[1]} "),
            (
                "splash serve --model unsloth/Model-GGUF:UD",
                f"splash serve --model {GGUF[1]} ",
            ),
            ("splash claude --model community/l", "splash claude --model community/l"),
            ("splash pi --model community/l", "splash pi --model community/l"),
        ):
            with self.subTest(line=line):
                self.assertEqual(self.shell_tab(directory / "_splash", line), expected)

    def test_actual_bash_tab_wordbreaks(self):
        _, directory = self.layout()
        for shell in bash_paths():
            for unbroken in ("", "=", ":"):
                for command, option in (
                    ("splash", "--model="),
                    ("splash", "--model "),
                    ("./splash", "--model="),
                ):
                    for typed, model in (
                        ("community/l", LOCAL[1]),
                        ("unsloth/Model-GGUF:UD", GGUF[1]),
                    ):
                        with self.subTest(
                            shell=shell,
                            unbroken=unbroken,
                            command=command,
                            option=option,
                            typed=typed,
                        ):
                            actual = self.shell_tab(
                                directory / "splash.bash",
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

    @unittest.skipUnless(os.access("/bin/zsh", os.X_OK), "Zsh is not installed")
    def test_zsh_loaded_completion_survives_release_upgrade(self):
        old, _ = self.layout("old release", release=True)
        new = self.root / "new release"
        shutil.copytree(old, new)
        link = self.root / "app current"
        link.symlink_to(old, target_is_directory=True)
        self.assertEqual(
            self.shell_tab(
                link / "install/completions/_splash",
                "splash serve --model community/l",
                upgrade=(link, new, old),
            ),
            f"splash serve --model {LOCAL[1]} ",
        )

    @unittest.skipUnless(os.access("/bin/zsh", os.X_OK), "Zsh is not installed")
    def test_zsh_autoload_through_global_symlink_survives_upgrade(self):
        old, _ = self.layout("old keg", release=True)
        new = self.root / "new keg"
        shutil.copytree(old, new)
        opt = self.root / "opt/splash"
        opt.parent.mkdir()
        opt.symlink_to(old, target_is_directory=True)
        entry = self.root / "share/zsh/site-functions/_splash"
        entry.parent.mkdir(parents=True)
        entry.symlink_to(opt / "install/completions/_splash")
        self.assertEqual(
            self.shell_tab(
                entry,
                "splash serve --model=community/l",
                upgrade=(opt, new, old),
                autoload=True,
            ),
            f"splash serve --model={LOCAL[1]} ",
        )


if __name__ == "__main__":
    unittest.main()
