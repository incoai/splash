"""Launch installed coding clients against one ready local server.

Connection/model configuration and safe launch defaults belong here. Tool
inventories, prompts and permission enforcement remain the client's responsibility.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

INSTALL_URLS = {
    "claude": "https://code.claude.com/docs/en/overview",
    "opencode": "https://opencode.ai/docs/",
    "codex": "https://developers.openai.com/codex/cli/",
    "hermes": "https://hermes-agent.nousresearch.com/docs/getting-started/installation/",
}


class ClientError(RuntimeError):
    pass


def _output_budget(context):
    return min(32768, max(1, context // 4))


def find_executable(name):
    path = shutil.which(name)
    if path is None:
        raise ClientError(
            f"{name} is not installed or is not on PATH. "
            f"Install it first: {INSTALL_URLS[name]}"
        )
    return path


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
    match = re.match(
        r"(?:opencode\s+)?v?(\d+)\.\d+(?:\.\d+)?(?:[-+\s]|$)",
        result.stdout.strip(),
    )
    return int(match.group(1)) if match else None


def _hermes_config(home, model, endpoint, context, api_key):
    # HERMES_HOME is Hermes's supported profile boundary. Keep sessions and
    # the complete default tool surface, without touching ~/.hermes/config.yaml.
    import yaml

    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = home / "config.yaml"
    try:
        config = yaml.safe_load(path.read_text()) if path.exists() else {}
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise ValueError("expected a mapping")
        if config.get("model") is None:
            config["model"] = {}
        config["model"].update(
            default=model,
            provider="custom",
            base_url=endpoint,
            api_key=api_key,
            api_mode="chat_completions",
            supports_vision=True,
            context_length=context,
            # Leave the input room expected by Hermes's 75% small-context
            # compaction threshold; do not inherit a cloud model's output cap.
            max_tokens=_output_budget(context),
        )
    except (yaml.YAMLError, ValueError, AttributeError) as error:
        raise ClientError(f"Invalid Hermes profile: {path}") from error
    # Replace only after the complete config is ready; two launching shells
    # never expose a partially written YAML file to Hermes.
    with tempfile.NamedTemporaryFile(mode="w", dir=home, delete=False) as output:
        temporary = Path(output.name)
        try:
            yaml.safe_dump(config, output, sort_keys=False)
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


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


def command(
    name,
    path,
    base_url,
    model,
    context,
    runtime_dir,
    environment=None,
    *,
    client_args=(),
    client_version=None,
):
    """Return argv and a private environment; never mutate the caller's env."""
    if not isinstance(model, str) or not model:
        raise ClientError("The server did not report a model name")
    if type(context) is not int or context <= 0:
        raise ClientError("The server did not report a valid context limit")
    environment = dict(os.environ if environment is None else environment)
    endpoint = base_url.rstrip("/") + "/v1"
    api_key = environment.get("SPLASH_API_KEY") or "local"

    if name == "claude":
        environment.update(
            ANTHROPIC_BASE_URL=base_url.rstrip("/"),
            ANTHROPIC_AUTH_TOKEN=api_key,
            ANTHROPIC_MODEL=model,
            ANTHROPIC_DEFAULT_OPUS_MODEL=model,
            ANTHROPIC_DEFAULT_SONNET_MODEL=model,
            ANTHROPIC_DEFAULT_HAIKU_MODEL=model,
            ANTHROPIC_SMALL_FAST_MODEL=model,
            CLAUDE_CODE_MAX_CONTEXT_TOKENS=str(context),
            CLAUDE_CODE_AUTO_COMPACT_WINDOW=str(context),
        )
        # Select the direct endpoint even in a shell configured for a cloud
        # provider. Keep compaction, tools and user settings unchanged.
        environment.pop("ANTHROPIC_API_KEY", None)
        for key in (
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_USE_VERTEX",
            "CLAUDE_CODE_USE_FOUNDRY",
        ):
            environment[key] = "0"
        # Auto mode adds classifier inference to tool approval. Start in the
        # normal approval mode, even if the user's global default is auto.
        return [
            path,
            "--disallowedTools",
            "WebSearch",
            "--model",
            model,
            "--permission-mode",
            "default",
            *client_args,
        ], environment

    if name == "opencode":
        output = _output_budget(context)
        try:
            config = json.loads(environment.get("OPENCODE_CONFIG_CONTENT", "{}"))
            config.update(model=f"splash/{model}", small_model=f"splash/{model}")
            # A user's global config may pin a model per agent, and an
            # agent-level model outranks the top-level one; point the built-in
            # agents at the served model too, leaving their other settings.
            for agent in ("build", "plan", "general", "explore", "title", "compaction"):
                config.setdefault("agent", {}).setdefault(agent, {})["model"] = (
                    f"splash/{model}"
                )
            variants = (
                config.get("provider", {})
                .get("splash", {})
                .get("models", {})
                .get(model, {})
                .get("variants", {})
            )
            config.setdefault("provider", {})["splash"] = {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Splash",
                "options": {"baseURL": endpoint, "apiKey": api_key},
                "models": {
                    model: {
                        "name": model,
                        "reasoning": True,
                        "variants": {
                            "none": {"reasoningEffort": "none"},
                            "low": {"reasoningEffort": "low"},
                            "medium": {"reasoningEffort": "medium"},
                            "high": {"reasoningEffort": "high"},
                            "xhigh": {"reasoningEffort": "xhigh"},
                            **variants,
                        },
                        "attachment": True,
                        "modalities": {
                            "input": ["text", "image", "pdf"],
                            "output": ["text"],
                        },
                        # The shared window includes the output allowance.
                        # An explicit input budget also lets OpenCode retain
                        # its normal compaction reserve for the next turn.
                        "limit": {
                            "context": context,
                            "input": max(1, context - output),
                            "output": output,
                        },
                    }
                },
            }
        except (ValueError, TypeError, AttributeError) as error:
            raise ClientError(
                "OPENCODE_CONFIG_CONTENT must be a JSON object"
            ) from error
        environment["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
        # OpenCode 2 loads configuration inside its persistent background
        # service, which this process's environment never reaches; a private
        # server keeps the inline configuration authoritative. OpenCode 1
        # reads the environment directly and rejects the flag.
        standalone = (
            type(client_version) is int
            and client_version >= 2
            and not _selects_opencode_server(client_args)
        )
        argv = [path, *client_args]
        if standalone:
            # V2 parses this as a command-local flag, so it must follow any
            # subcommand and precede the end-of-options separator.
            index = argv.index("--") if "--" in argv else len(argv)
            argv.insert(index, "--standalone")
        return argv, environment

    if name == "codex":
        environment["SPLASH_API_KEY"] = api_key
        settings = {
            "web_search": json.dumps("disabled"),
            "model_provider": json.dumps("splash"),
            "model_providers.splash": (
                '{name="Splash",base_url='
                + json.dumps(endpoint)
                + ',env_key="SPLASH_API_KEY",wire_api="responses"}'
            ),
            "model_context_window": str(context),
            # Override a possible threshold from the user's other model. The
            # remaining 10% is room for completion and compaction itself.
            "model_auto_compact_token_limit": str(context * 9 // 10),
        }
        argv = [path]
        settings = {"model": json.dumps(model), **settings}
        for key, value in settings.items():
            argv.extend(["-c", f"{key}={value}"])
        # User overrides retain their order and take precedence over defaults.
        # Config is global in Codex. Keep its complete list at the root so
        # a subcommand's overrides cannot replace the connection settings.
        overrides, arguments = _codex_config_args(client_args)
        return [*argv, *overrides, *arguments], environment

    if name == "hermes":
        home = runtime_dir / "hermes"
        _hermes_config(home, model, endpoint, context, api_key)
        environment.update(
            HERMES_HOME=str(home),
            CUSTOM_BASE_URL=endpoint,
            OPENAI_BASE_URL=endpoint,
            OPENAI_API_KEY=api_key,
        )
        return [
            path,
            "chat",
            "--provider",
            "custom",
            "--model",
            model,
            *client_args,
        ], environment

    raise ClientError(f"Unknown coding client: {name}")
