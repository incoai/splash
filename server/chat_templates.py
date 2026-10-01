"""Choose the chat templates once at startup, with later system messages.

Agent clients add instructions during a conversation: a system message after
the first message (request preparation merges the leading system and developer
messages into one, so every later one follows another role). Splash renders
it where it occurs, as a system turn in the template's own markup. Upstream
templates differ: the official Qwen templates raise for it and Unsloth's
Qwen3.6 GGUF template skips it silently.

Every template the tokenizer defines is probed once, when the server starts,
by rendering a canary conversation whose later system message carries a
marker. The outcome is one of:

- ``native``: the marker renders in place; the template is used unchanged.
- ``patched``: the template rejected or dropped the message. A structural
  patch at the construct responsible -- the ``raise_exception`` in the message
  loop's system branch, or the loop condition that skips system messages --
  renders it with the template's own system block. The patch is kept only if
  ordinary conversations still render byte-identically and the canary renders
  in place.
- ``unsupported``: the template renders the message out of place, has no
  single such construct, renders something before its system block (such as
  a BOS token), or its patch failed a probe. A request with a later system
  message fails with a 400 instead of losing it.

The probe also records the generation prompt, the text a template appends
for ``add_generation_prompt``, under each reasoning effort, and under other
template variables a request sets when it first sets them. A request's prompt
must end with it, and it says whether generation starts in a think block. The
next turn may render it differently, so the engine keeps a turn's reusable
state before it. No chat format's markers are assumed.

Requests and the probe share the reasoning efforts, template options and
alias retry defined here, so the probe renders exactly as requests do.
Tokenizer files and the tokenizer object are never modified.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from jinja2 import Environment, TemplateError, TemplateSyntaxError, nodes
from jinja2.ext import Extension
from jinja2.lexer import Token

# Template arguments request preparation sets itself, and apply_chat_template's
# own controls: a request's template kwargs cannot set them.
RESERVED_TEMPLATE_KWARGS = frozenset(
    {
        "add_generation_prompt",
        "chat_template",
        "continue_final_message",
        "conversation",
        "documents",
        "messages",
        "return_dict",
        "tokenize",
        "tools",
    }
)
# Template options that cannot shape the generation prompt: those (the probe
# renders without tools) and preserve_thinking, which only changes history.
_HISTORY_OPTIONS = RESERVED_TEMPLATE_KWARGS | {"preserve_thinking"}
# Combinations of other template variables whose generation prompts are kept.
_PROBED_OPTIONS = 64

# What Splash does with a later system message.
NATIVE = "native"
PATCHED = "patched"
UNSUPPORTED = "unsupported"
# What the unmodified template does with the canary's later system message.
RENDERS = "renders"
REJECTS = "rejects"
DROPS = "drops"
MISPLACES = "misplaces"

LATER_SYSTEM_UNSUPPORTED = (
    "this model's chat template does not accept system messages after the first message"
)


class ChatTemplateError(ValueError):
    """The tokenizer has no chat template Splash can serve."""


REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
# What a template that rejects one of these efforts renders instead.
REASONING_EFFORT_ALIASES = {"high": "xhigh", "max": "xhigh", "minimal": "low"}


def template_options(
    *, reasoning_effort, preserve_thinking, tools, add_generation_prompt
):
    """Template variables exactly as request preparation passes them."""
    options = {"add_generation_prompt": add_generation_prompt}
    if reasoning_effort is not None:
        options["enable_thinking"] = reasoning_effort != "none"
        if reasoning_effort != "none":
            options["reasoning_effort"] = reasoning_effort
    if preserve_thinking is not None:
        options["preserve_thinking"] = preserve_thinking
    if tools:
        options["tools"] = tools
    return options


def render_chat_template(tokenizer, messages, options):
    """Render as requests and the startup probe do: a template that rejects
    the reasoning effort renders its alias instead."""
    try:
        return tokenizer.apply_chat_template(messages, **options)
    except TemplateError:
        alias = REASONING_EFFORT_ALIASES.get(options.get("reasoning_effort"))
        if alias is None:
            raise
        return tokenizer.apply_chat_template(
            messages, **{**options, "reasoning_effort": alias}
        )


@dataclass(frozen=True, slots=True)
class ChatTemplate:
    source: str
    # NATIVE, PATCHED or UNSUPPORTED.
    later_system: str
    # RENDERS, REJECTS, DROPS or MISPLACES.
    original: str
    # The generation prompt's text and tokens by the options that shape it
    # (_generation_key), for each reasoning effort; absent where it is empty
    # or the probes disagree.
    generation_prompts: dict
    # The same for other options, probed on first use.
    probe: Callable

    def accepts(self, messages):
        """Whether requests with these normalized messages are served: a
        system message after the first needs a template that renders it."""
        return self.later_system != UNSUPPORTED or all(
            message["role"] != "system" for message in messages[1:]
        )

    def generation_prompt(self, options):
        """The text and tokens add_generation_prompt appends under these
        request options; empty where the probe found none."""
        key = _generation_key(options)
        found = self.generation_prompts.get(key)
        return found if found is not None else self.probe(key)


class ChatTemplates:
    """The tokenizer's templates, probed and chosen once at startup."""

    def __init__(self, tokenizer):
        defined = tokenizer.chat_template
        named = dict(defined) if isinstance(defined, dict) else {None: defined}
        if isinstance(defined, dict) and "default" not in named:
            raise ChatTemplateError(
                "the tokenizer's named chat templates have no default"
            )
        if not all(isinstance(source, str) and source for source in named.values()):
            raise ChatTemplateError("the tokenizer defines no chat template")
        render, encode = _renderer(tokenizer), _encoder(tokenizer)
        prepared = {}
        for source in named.values():
            if source not in prepared:
                prepared[source] = _prepare(render, encode, source)
        self.templates = {name: prepared[source] for name, source in named.items()}

    def select(self, tools):
        """The template the tokenizer itself would choose for this request."""
        if None in self.templates:
            return self.templates[None]
        if tools and "tool_use" in self.templates:
            return self.templates["tool_use"]
        return self.templates["default"]

    def status(self):
        if None in self.templates:
            return {"later_system": self.templates[None].later_system}
        return {
            "later_system": {
                name: template.later_system for name, template in self.templates.items()
            }
        }

    def describe(self):
        """One line for the startup log."""
        return " · ".join(
            ("" if name is None else f"{name} ")
            + _DESCRIPTIONS[template.later_system].format(template.original)
            + ", "
            + _generation_description(template.generation_prompts)
            for name, template in self.templates.items()
        )


# Startup log text for each outcome; {} names what the unmodified template
# does with a later system message.
_DESCRIPTIONS = {
    NATIVE: "renders later system messages in place",
    PATCHED: (
        "patched to render later system messages in place "
        "(the original template {} them)"
    ),
    UNSUPPORTED: (
        "requests with later system messages are rejected "
        "(the original template {} them)"
    ),
}


def _generation_description(prompts):
    """Startup log text for the generation prompts the probe found: their
    token counts across reasoning efforts."""
    counts = sorted({len(tokens) for _text, tokens in prompts.values()})
    if not counts:
        return "generation prompt unknown"
    if counts[0] != counts[-1]:
        return f"generation prompt {counts[0]}-{counts[-1]} tokens"
    return f"generation prompt {counts[0]} {'token' if counts[0] == 1 else 'tokens'}"


def _renderer(tokenizer):
    def render(source, messages, options):
        return render_chat_template(
            tokenizer, messages, {**options, "chat_template": source, "tokenize": False}
        )

    return render


def _encoder(tokenizer):
    def encode(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    return encode


def _prepare(render, encode, source):
    original = _original(render, source)
    later_system = UNSUPPORTED
    if original == RENDERS:
        later_system = NATIVE
    elif original in (REJECTS, DROPS):
        patched = _patch(render, source, original)
        if patched is not None and _verified(render, source, patched):
            source, later_system = patched, PATCHED

    @lru_cache(maxsize=_PROBED_OPTIONS)
    def probe(key):
        options = {**json.loads(key), "add_generation_prompt": True}
        return _generation_prompt(render, encode, source, options) or ("", ())

    return ChatTemplate(
        source,
        later_system,
        original,
        _generation_prompts(render, encode, source),
        probe,
    )


# Probe conversations. Leading system messages are already merged, as request
# preparation merges them; images use the canonical request part.
_MARKER = "Splash later-system canary 5d0c8e"
_SYSTEM = {"role": "system", "content": "Leading instructions"}
_ASK = {"role": "user", "content": "First question"}
_ANSWER = {"role": "assistant", "content": "First answer"}
_LATER = {"role": "system", "content": _MARKER}
_NEXT = {"role": "user", "content": "Next question"}
_THOUGHT = {"role": "assistant", "content": "First answer", "reasoning_content": "Hm"}
_CALL = {
    "role": "assistant",
    "content": "",
    "reasoning_content": "Look it up",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": {"key": "alpha"}},
        }
    ],
}
_RESULT = {"role": "tool", "tool_call_id": "call_1", "content": "beta"}
_IMAGE = {
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
        {"type": "text", "text": "Describe it"},
    ],
}
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a key.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    }
]
# Pairs of a canary and the same conversation without its later system message.
_CANARIES = (
    ([_SYSTEM, _ASK, _ANSWER, _LATER, _NEXT], [_SYSTEM, _ASK, _ANSWER, _NEXT]),
    ([_ASK, _ANSWER, _LATER, _NEXT], [_ASK, _ANSWER, _NEXT]),
)
_ORDINARY = (
    [_ASK],
    [_SYSTEM, _ASK],
    [_SYSTEM, _ASK, _THOUGHT, _NEXT],
    [_SYSTEM, _ASK, _CALL, _RESULT],
    [_SYSTEM, _ASK, _CALL, _RESULT, _THOUGHT, _NEXT],
    [_SYSTEM, _IMAGE],
)
_OPTIONS = (
    *(
        template_options(
            reasoning_effort=effort,
            preserve_thinking=preserve,
            tools=tools,
            add_generation_prompt=True,
        )
        for tools in (None, _TOOLS)
        for effort in (None, *REASONING_EFFORTS)
        for preserve in (None, True, False)
    ),
    {"add_generation_prompt": False},
)


# Conversations the generation prompt must be the same for: a question, a
# follow-up after a reasoned answer, and a tool result.
_FOLLOWED = ([_ASK], [_SYSTEM, _ASK, _THOUGHT, _NEXT], [_SYSTEM, _ASK, _CALL, _RESULT])


def _generation_key(options):
    """The template options that can shape the generation prompt."""
    return json.dumps(
        {
            name: value
            for name, value in options.items()
            if name not in _HISTORY_OPTIONS
        },
        sort_keys=True,
    )


def _generation_prompts(render, encode, source):
    """The generation prompt for each reasoning effort a request can pass."""
    prompts = {}
    for effort in (None, *REASONING_EFFORTS):
        options = template_options(
            reasoning_effort=effort,
            preserve_thinking=None,
            tools=None,
            add_generation_prompt=True,
        )
        if found := _generation_prompt(render, encode, source, options):
            prompts[_generation_key(options)] = found
    return prompts


def _generation_prompt(render, encode, source, options):
    """What add_generation_prompt appends under these options, with its
    tokens: kept where every probe conversation renders as a prefix of its
    prompt and gains the same non-empty text."""
    texts = set()
    for messages in _FOLLOWED:
        prompt = _outcome(render, source, messages, options)
        history = _outcome(
            render, source, messages, {**options, "add_generation_prompt": False}
        )
        if not (
            isinstance(prompt, str)
            and isinstance(history, str)
            and prompt.startswith(history)
        ):
            return None
        texts.add(prompt[len(history) :])
    if len(texts) == 1 and (text := texts.pop()):
        return text, tuple(encode(text))
    return None


def _outcome(render, source, messages, options):
    try:
        return render(source, messages, options)
    except Exception as error:  # templates reject some options themselves
        return type(error), str(error)


def _in_place(rendered):
    marker = rendered.find(_MARKER)
    before = rendered.find(_ANSWER["content"])
    after = rendered.find(_NEXT["content"])
    return (
        rendered.count(_MARKER) == 1
        and 0 <= before
        and before + len(_ANSWER["content"]) <= marker
        and marker + len(_MARKER) <= after
    )


def _original(render, source):
    """How the unmodified template treats the canaries' later system message."""
    for canary, _ in _CANARIES:
        try:
            rendered = render(source, canary, {"add_generation_prompt": True})
        except Exception:
            return REJECTS
        if _MARKER not in rendered:
            return DROPS
        if not _in_place(rendered):
            return MISPLACES
    return RENDERS


def _verified(render, original, patched):
    """Ordinary conversations render byte-identically, and wherever the
    original renders a canary's conversation, the patch renders the canary
    in place."""
    for options in _OPTIONS:
        for messages in _ORDINARY:
            if _outcome(render, original, messages, options) != _outcome(
                render, patched, messages, options
            ):
                return False
        for canary, without in _CANARIES:
            if not isinstance(_outcome(render, original, without, options), str):
                continue
            rendered = _outcome(render, patched, canary, options)
            if not isinstance(rendered, str) or not _in_place(rendered):
                return False
    return True


def _system_block(render, source):
    """The template's own block around a system message's content.

    It is what a leading system message adds to the front of a one-question
    conversation without tools or thinking instructions, so a template that
    renders anything before its system block, such as a BOS token, is not
    patched.
    """
    ask = [_ASK]
    options = {"add_generation_prompt": False, "enable_thinking": False}
    try:
        with_system = render(
            source, [{"role": "system", "content": _MARKER}, *ask], options
        )
        without = render(source, ask, options)
    except Exception:
        return None
    if not with_system.endswith(without):
        return None
    block = with_system[: len(with_system) - len(without)]
    if block.count(_MARKER) != 1:
        return None
    return block.split(_MARKER)


def _literal(text):
    """A Jinja string literal for text."""
    return "'" + text.encode("unicode_escape").decode("ascii").replace("'", "\\'") + "'"


def _patch(render, source, original):
    """The template with its rejecting or dropping construct replaced, or None."""
    # Jinja lexes CRLF and CR as LF and renders them so; tags are located in
    # the source as it lexes.
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    try:
        tree = _ENVIRONMENT.parse(source)
    except TemplateSyntaxError:
        return None
    spans = _tag_spans(source)
    find = _rejections if original == REJECTS else _skips
    # The one construct responsible, as a tag; none or several is no patch.
    constructs = {
        spans[node.lineno]: variable
        for loop, variable in _message_loops(tree)
        for node in find(loop, variable)
    }
    if len(constructs) != 1:
        return None
    block = _system_block(render, source)
    if block is None:
        return None
    (((start, end), variable),) = constructs.items()
    prefix, suffix = block
    # The later system message, rendered as the template renders a system
    # message: its own block around the trimmed content.
    later = (
        f"{{{{- {_literal(prefix)} + ({variable}.content|trim) + "
        f"{_literal(suffix)} -}}}}"
    )
    if original == REJECTS:
        # Render where the template raised; its own guard (not the first
        # message, or past the merged leading messages) still applies.
        replacement = later
    else:
        # Render ahead of the condition that skipped the message. Leading
        # system messages are merged into the first message, so every system
        # message after the first is a later one.
        text = source[start:end]
        head = _STATEMENT_HEAD.match(text).group()
        replacement = (
            f"{head}if not loop.first and {variable}.role == 'system' %}}"
            f"{later}{head}el{text[len(head) :]}"
        )
    return source[:start] + replacement + source[end:]


# Structural matching. Jinja parses the template to find the construct, and
# lexes it to locate the tag the construct was parsed from.


class _Generation(Extension):
    """transformers' ``{% generation %}`` block, parsed as a plain scope."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.Scope(body, lineno=lineno)


class _TagNumbers(Extension):
    """Numbers each parsed node by the tag it starts in, in place of its line,
    so ``_tag_spans(source)[node.lineno]`` is where it was written."""

    def filter_stream(self, stream):
        number = -1
        for token in stream:
            if token.type in ("variable_begin", "block_begin"):
                number += 1
            yield Token(number, token.type, token.value)


_ENVIRONMENT = Environment(
    extensions=["jinja2.ext.loopcontrols", _Generation, _TagNumbers]
)
_STATEMENT_HEAD = re.compile(r"\{%[-+]?\s*")


def _tag_spans(source):
    """Where each expression and statement tag is, in the order Jinja lexes
    them (so strings, comments and raw sections are Jinja's own)."""
    spans, cursor = [], 0
    for _, token, value in _ENVIRONMENT.lex(source):
        # Tokens follow each other, except for whitespace a "-" strips.
        position = source.index(value, cursor)
        cursor = position + len(value)
        if token in ("variable_begin", "block_begin"):
            start = position
        elif token in ("variable_end", "block_end"):
            # "-%}" and "-}}" take the whitespace they strip; the tag does not.
            spans.append((start, position + len(value.rstrip())))
    return spans


def _message_loops(tree):
    """(loop, variable) of each ``for`` over ``messages``."""
    for loop in tree.find_all(nodes.For):
        if (
            isinstance(loop.target, nodes.Name)
            and isinstance(loop.iter, nodes.Name)
            and loop.iter.name == "messages"
            and loop.test is None
        ):
            yield loop, loop.target.name


def _within(body, kind):
    """The nodes of a kind in a body, at any depth."""
    for node in body:
        if isinstance(node, kind):
            yield node
        yield from node.find_all(kind)


def _terms(node, operator):
    """The operands of a chain of ``and`` or ``or``."""
    if isinstance(node, operator):
        return [*_terms(node.left, operator), *_terms(node.right, operator)]
    return [node]


def _compares_role(node, variable, op):
    """Whether node is ``variable.role <op> 'system'`` (attribute or item)."""
    if not (
        isinstance(node, nodes.Compare)
        and len(node.ops) == 1
        and node.ops[0].op == op
        and node.ops[0].expr == nodes.Const("system")
    ):
        return False
    subject = node.expr
    if isinstance(subject, nodes.Getattr):
        name, target = subject.attr, subject.node
    elif isinstance(subject, nodes.Getitem) and isinstance(subject.arg, nodes.Const):
        name, target = subject.arg.value, subject.node
    else:
        return False
    return name == "role" and target == nodes.Name(variable, "load")


def _rejections(loop, variable):
    """The raise_exception calls in the loop's branches for system messages;
    each ``elif`` is an ``If`` node of its own."""
    for branch in _within(loop.body, nodes.If):
        if any(
            _compares_role(term, variable, "eq")
            for term in _terms(branch.test, nodes.Or)
        ):
            for output in _within(branch.body, nodes.Output):
                for call in output.nodes:
                    if isinstance(call, nodes.Call) and call.node == nodes.Name(
                        "raise_exception", "load"
                    ):
                        yield call


def _skips(loop, variable):
    """The condition directly in the loop that excludes system messages."""
    for node in loop.body:
        if (
            isinstance(node, nodes.If)
            and not node.elif_
            and not node.else_
            and any(
                _compares_role(term, variable, "ne")
                for term in _terms(node.test, nodes.And)
            )
        ):
            yield node
