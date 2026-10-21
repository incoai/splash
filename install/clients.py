"""Launch installed coding clients against one ready local server.

Connection/model configuration and safe launch defaults belong here. Tool
inventories, prompts and permission enforcement remain the client's responsibility.
"""

import fcntl
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from server import serve_options

INSTALL_URLS = {
    "claude": "https://code.claude.com/docs/en/overview",
    "opencode": "https://opencode.ai/docs/",
    "codex": "https://developers.openai.com/codex/cli/",
    "hermes": "https://hermes-agent.nousresearch.com/docs/getting-started/installation/",
    "pi": "https://pi.dev/",
}
# The output one response may use in the OpenCode, Pi and Hermes
# configurations the launchers write. Qwen recommends 32,768 for most Qwen3.6
# queries and evaluated Qwen3.8-27B under Claude Code with it in a 256K
# window; the 131,072 that model's card names is for the answer alone, beside
# a separate reasoning limit, in a 1M window. OpenCode sends at most 32,000
# of it unless OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX raises that; Pi, given
# none, asks for 16,384. OpenCode and Hermes get a quarter of a window
# smaller than four times this instead (_Server.response_tokens).
CLIENT_RESPONSE_TOKENS = 32768
OPENCODE_AGENTS = ("build", "plan", "general", "explore", "title", "compaction")
OPENCODE_EFFORTS = ("none", "low", "medium", "high", "xhigh")
OPENCODE_CONFIG_ERROR = "OPENCODE_CONFIG_CONTENT must be a JSON object"


class ClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class _Server:
    """The ready local server a client connects to."""

    base_url: str
    model: str
    context: int
    input_modalities: list
    api_key: str

    @property
    def endpoint(self):
        return self.base_url + "/v1"

    @property
    def response_tokens(self):
        """The output allowance of a client that reserves it out of the
        input it compacts at, OpenCode and Hermes: CLIENT_RESPONSE_TOKENS,
        or a quarter of the window when that is less, so a small window
        keeps room for the conversation."""
        return min(CLIENT_RESPONSE_TOKENS, max(1, self.context // 4))

    @property
    def name(self):
        """splash for the default port and splash-<port> for another, so a
        client's session keeps the server it was started for."""
        port = urllib.parse.urlsplit(self.base_url).port
        return "splash" if port == serve_options.DEFAULT_PORT else f"splash-{port}"


def find_executable(name):
    path = shutil.which(name)
    if path is None:
        raise ClientError(
            f"{name} is not installed or is not on PATH. "
            f"Install it first: {INSTALL_URLS[name]}"
        )
    return path


def major_version(text):
    """The major version a client's --version output names; None if none."""
    match = re.match(
        r"(?:opencode\s+)?v?(\d+)\.\d+(?:\.\d+)?(?:[-+\s]|$)",
        text.strip(),
    )
    return int(match.group(1)) if match else None


def probe_major_version(path):
    """Best-effort major version of an installed client; None if probing fails.

    Callers treat None as 'unknown' and keep the launch they already had.
    """
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if result.returncode:
        return None
    return major_version(result.stdout)


def command(
    name,
    path,
    base_url,
    model,
    context,
    environment=None,
    *,
    input_modalities,
    client_args=(),
    client_version=None,
    hermes_profile=None,
):
    """Return argv and a private environment; never mutate the caller's env.
    input_modalities is what the served model accepts, as /v1/models
    reports it. hermes_profile names the Hermes profile to configure, by
    default the server's name."""
    if not isinstance(model, str) or not model:
        raise ClientError("The server did not report a model name")
    if type(context) is not int or context <= 0:
        raise ClientError("The server did not report a valid context limit")
    if (
        not isinstance(input_modalities, list)
        or "text" not in input_modalities
        or not all(isinstance(m, str) for m in input_modalities)
    ):
        raise ClientError(
            "The server did not report its input modalities; it predates this "
            "launcher, so restart it with this version of Splash"
        )
    environment = dict(os.environ if environment is None else environment)
    server = _Server(
        base_url.rstrip("/"),
        model,
        context,
        input_modalities,
        environment.get("SPLASH_API_KEY") or "local",
    )
    # Each launch configures the private environment copy in place.
    match name:
        case "claude":
            argv = _claude(path, server, environment, client_args)
        case "opencode":
            argv = _opencode(path, server, environment, client_args, client_version)
        case "codex":
            argv = _codex(path, server, environment, client_args)
        case "hermes":
            argv = _hermes(path, server, environment, client_args, hermes_profile)
        case "pi":
            argv = _pi(path, server, environment, client_args)
        case _:
            raise ClientError(f"Unknown coding client: {name}")
    return argv, environment


def _claude(path, server, environment, arguments):
    environment.update(
        ANTHROPIC_BASE_URL=server.base_url,
        ANTHROPIC_AUTH_TOKEN=server.api_key,
        ANTHROPIC_MODEL=server.model,
        ANTHROPIC_DEFAULT_OPUS_MODEL=server.model,
        ANTHROPIC_DEFAULT_SONNET_MODEL=server.model,
        ANTHROPIC_DEFAULT_HAIKU_MODEL=server.model,
        ANTHROPIC_SMALL_FAST_MODEL=server.model,
        CLAUDE_CODE_MAX_CONTEXT_TOKENS=str(server.context),
        CLAUDE_CODE_AUTO_COMPACT_WINDOW=str(server.context),
    )
    # Select the direct endpoint even in a shell configured for a cloud
    # provider. Keep compaction, tools and user settings unchanged.
    environment.pop("ANTHROPIC_API_KEY", None)
    environment.update(
        CLAUDE_CODE_USE_BEDROCK="0",
        CLAUDE_CODE_USE_VERTEX="0",
        CLAUDE_CODE_USE_FOUNDRY="0",
    )
    # Auto mode adds classifier inference to tool approval. Start in the
    # normal approval mode, even if the user's global default is auto.
    return [
        path,
        "--disallowedTools",
        "WebSearch",
        "--model",
        server.model,
        "--permission-mode",
        "default",
        *arguments,
    ]


def _opencode(path, server, environment, arguments, version):
    try:
        config = json.loads(environment.get("OPENCODE_CONFIG_CONTENT", "{}"))
    except ValueError as error:
        raise ClientError(OPENCODE_CONFIG_ERROR) from error
    if not isinstance(config, dict):
        raise ClientError(OPENCODE_CONFIG_ERROR)
    served = f"splash/{server.model}"
    config.update(model=served, small_model=served)
    # A user's global config may pin a model per agent, and an agent-level
    # model outranks the top-level one; point the built-in agents at the
    # served model too, leaving their other settings.
    agents = config["agent"] = _opencode_object(config, "agent")
    for agent in OPENCODE_AGENTS:
        agents[agent] = {**_opencode_object(agents, agent), "model": served}
    provider = config["provider"] = _opencode_object(config, "provider")
    variants = {effort: {"reasoningEffort": effort} for effort in OPENCODE_EFFORTS}
    variants |= _opencode_object(provider, "splash", "models", server.model, "variants")
    output = server.response_tokens
    provider["splash"] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Splash",
        "options": {"baseURL": server.endpoint, "apiKey": server.api_key},
        "models": {
            server.model: {
                "name": server.model,
                "reasoning": True,
                "variants": variants,
                # Attachments are images and PDFs.
                "attachment": server.input_modalities != ["text"],
                "modalities": {"input": server.input_modalities, "output": ["text"]},
                # The shared window includes the output allowance. An explicit
                # input budget also lets OpenCode retain its normal compaction
                # reserve for the next turn.
                "limit": {
                    "context": server.context,
                    "input": max(1, server.context - output),
                    "output": output,
                },
            }
        },
    }
    environment["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    argv = [path, *arguments]
    # OpenCode 2 loads configuration inside its persistent background
    # service, which this process's environment never reaches; a private
    # server keeps the inline configuration authoritative. OpenCode 1
    # reads the environment directly and rejects the flag.
    if (
        type(version) is int
        and version >= 2
        and not _selects_opencode_server(arguments)
    ):
        # V2 parses this as a command-local flag, so it must follow any
        # subcommand and precede the end-of-options separator.
        argv.insert(argv.index("--") if "--" in argv else len(argv), "--standalone")
    return argv


def _opencode_object(value, *keys):
    """The object at keys in the user's inline config; {} where it is unset."""
    for key in keys:
        value = value.get(key, {})
        if not isinstance(value, dict):
            raise ClientError(f"OPENCODE_CONFIG_CONTENT: {key!r} must be a JSON object")
    return value


def _selects_opencode_server(arguments):
    """The user already chose the server: an explicit URL or a private one."""
    if "--" in arguments:
        arguments = arguments[: arguments.index("--")]
    return any(
        argument == "--standalone"
        or argument == "--server"
        or argument.startswith("--server=")
        for argument in arguments
    )


def _codex(path, server, environment, arguments):
    environment["SPLASH_API_KEY"] = server.api_key
    settings = {
        "model": json.dumps(server.model),
        "web_search": json.dumps("disabled"),
        "model_provider": json.dumps("splash"),
        "model_providers.splash": (
            f'{{name="Splash",base_url={json.dumps(server.endpoint)},'
            'env_key="SPLASH_API_KEY",wire_api="responses"}'
        ),
        "model_context_window": str(server.context),
        # Override a possible threshold from the user's other model. The
        # remaining 10% is room for completion and compaction itself.
        "model_auto_compact_token_limit": str(server.context * 9 // 10),
    }
    argv = [path]
    for key, value in settings.items():
        argv += ["-c", f"{key}={value}"]
    # User overrides retain their order and take precedence over defaults.
    # Config is global in Codex. Keep its complete list at the root so a
    # subcommand's overrides cannot replace the connection settings.
    overrides, arguments = _codex_config_args(arguments)
    return [*argv, *overrides, *arguments]


def _codex_config_args(arguments):
    """Keep config overrides together: nested Clap levels can replace the root list."""
    config, remaining = [], []
    arguments = iter(arguments)
    for argument in arguments:
        if argument == "--":
            remaining.extend([argument, *arguments])
            break
        if argument in ("-c", "--config"):
            value = next(arguments, None)
            if value is None:
                raise ClientError(f"{argument} requires a config override")
            config.extend(["-c", value])
        elif argument.startswith("--config="):
            config.extend(["-c", argument.split("=", 1)[1]])
        elif argument.startswith("-c") and len(argument) > 2:
            config.extend(["-c", argument[2:].removeprefix("=")])
        else:
            remaining.append(argument)
    return config, remaining


def _hermes(path, server, environment, arguments, profile):
    # Hermes keeps its managed tools, and the launchers of the user's hermes
    # command that run them, in its root, and each profile's configuration
    # and sessions in <root>/profiles/<name>. Any other HERMES_HOME is a root
    # of its own: Hermes would install its tools there and point the user's
    # hermes command at them. So run in a profile of the user's root, with
    # the complete default tool surface, and leave the root's config.yaml.
    home = hermes_profile_home(environment, profile or server.name)
    created = not (home / "config.yaml").exists() and _create_hermes_profile(
        path, home, environment
    )
    _write_hermes_profile(home, server, environment, created)
    environment.update(
        HERMES_HOME=str(home),
        CUSTOM_BASE_URL=server.endpoint,
        OPENAI_BASE_URL=server.endpoint,
        OPENAI_API_KEY=server.api_key,
    )
    # Hermes takes --provider and --model before any subcommand and runs chat
    # without one, so the user's arguments follow as Hermes itself takes them.
    return [path, "--provider", "custom", "--model", server.model, *arguments]


def hermes_profile_home(environment, name):
    """The home of profile name in the Hermes root of environment's
    HERMES_HOME, by Hermes's own rule (hermes_constants.get_default_hermes_root):
    ~/.hermes for a home inside it, <root> for <root>/profiles/<another>, and
    otherwise that home itself."""
    root = Path.home() / ".hermes"
    if value := environment.get("HERMES_HOME", "").strip():
        home = Path(os.path.expandvars(value)).expanduser()
        if not home.resolve().is_relative_to(root.resolve()):
            root = home.parent.parent if home.parent.name == "profiles" else home
    return root / "profiles" / name


def _create_hermes_profile(path, home, environment):
    """Create the profile with Hermes, as a user would, and say whether this
    call created it. Hermes seeds its files and lifts the mark that `hermes
    profile delete` leaves, without which Hermes refuses the profile. No
    alias: Hermes would add a command named after the profile."""
    try:
        result = subprocess.run(
            [path, "profile", "create", home.name, "--no-alias"],
            env=environment,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
    except OSError as error:
        raise ClientError(f"Could not run {path}: {error}") from error
    # Another launch may have created it meanwhile.
    if result.returncode and not home.is_dir():
        output = (result.stderr or result.stdout).strip()[-2000:]
        raise ClientError(f"hermes profile create {home.name} failed: {output}")
    return result.returncode == 0


def _write_hermes_profile(home, server, environment, created=False):
    # The launcher imports this module before the environment that provides
    # PyYAML is installed.
    import yaml

    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = home / "config.yaml"
    invalid = f"Invalid Hermes profile: {path}"
    try:
        profile = yaml.safe_load(path.read_text()) if path.exists() else None
    except (yaml.YAMLError, ValueError) as error:
        raise ClientError(invalid) from error
    if profile is None:
        profile = {}
    if not isinstance(profile, dict):
        raise ClientError(invalid)
    # Hermes gives a new profile a copy of the user's default model.
    if created or profile.get("model") is None:
        profile["model"] = {}
    if not isinstance(profile["model"], dict):
        raise ClientError(invalid)
    profile["model"].update(
        default=server.model,
        provider="custom",
        base_url=server.endpoint,
        # Hermes expands the reference from its environment, which the
        # launch passes on; the server's key is never saved.
        api_key="${SPLASH_API_KEY}" if environment.get("SPLASH_API_KEY") else "local",
        api_mode="chat_completions",
        supports_vision="image" in server.input_modalities,
        context_length=server.context,
        # Hermes before 2026.9.7 asks for this output limit on every request
        # and compacts at 75% of the window less it; left unset, it asks a
        # custom endpoint for 65,536, which its compaction does not reserve.
        # Later versions ignore it and leave the limit to the server.
        max_tokens=server.response_tokens,
    )
    _replace_file(path, yaml.safe_dump(profile, sort_keys=False))


def _pi(path, server, environment, arguments):
    # Pi reads custom providers only from models.json in its agent directory,
    # beside the user's sessions, settings and extensions. Replace this
    # server's provider there, named for the server, and leave everything
    # else as it is.
    provider = server.name
    _write_pi_provider(_pi_models_path(environment), provider, server, environment)
    return [path, "--provider", provider, "--model", server.model, *arguments]


def _pi_models_path(environment):
    agent = environment.get("PI_CODING_AGENT_DIR")
    directory = Path(agent).expanduser() if agent else Path.home() / ".pi/agent"
    return directory / "models.json"


def _write_pi_provider(path, provider, server, environment):
    # Write through a symlinked models.json, as dotfile managers link it.
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Launches for servers on other ports add their providers to the same
    # file: each reads and replaces it under the lock, so none replaces it
    # with a copy that lacks another's provider.
    with _locked(path.parent):
        invalid = f"Invalid Pi models.json: {path}"
        try:
            config = json.loads(path.read_text()) if path.exists() else {}
        except ValueError as error:
            raise ClientError(invalid) from error
        providers = config.get("providers", {}) if isinstance(config, dict) else None
        if not isinstance(providers, dict):
            raise ClientError(invalid)
        # Pi expands the variable for each request. The server's key is never
        # saved, nor run as a command when it begins with "!".
        api_key = "$SPLASH_API_KEY" if environment.get("SPLASH_API_KEY") else "local"
        # Pi accepts only text and image input; any other entry invalidates
        # the user's whole models.json.
        vision = "image" in server.input_modalities
        model = {
            "id": server.model,
            "reasoning": True,
            # Pi sends no effort when thinking is off, which leaves the
            # template's default; "none" turns thinking off.
            "thinkingLevelMap": {"off": "none"},
            "input": ["text", "image"] if vision else ["text"],
            "contextWindow": server.context,
            # Pi lowers this on each request to what its context leaves, and
            # compacts at a fixed distance from the window whatever it is.
            "maxTokens": CLIENT_RESPONSE_TOKENS,
        }
        config["providers"] = {
            **providers,
            provider: {
                "baseUrl": server.endpoint,
                "api": "openai-completions",
                "apiKey": api_key,
                "models": [model],
            },
        }
        _replace_file(path, json.dumps(config, indent=2) + "\n")


@contextmanager
def _locked(directory):
    """Hold the exclusive lock of directory, which the launchers take to
    update a file in it."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _replace_file(path, text):
    """Replace path only once text is complete, keeping its mode; a new file
    is private. Two launching shells never expose a partially written
    configuration to the client."""
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            os.fchmod(output.fileno(), mode)
            output.write(text)
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
