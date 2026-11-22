"""The chat format of each model family Splash serves (install/families.py):
how its model frames reasoning and lays out tool calls, which the server
reads its output by and constrains it to, and the tokens it shares with the
engine and the tokenizer.

The vocabulary and stop tokens come from the engine's Ready; a format names
the texts its tokenizer encodes as those stop tokens, and the tokens it must
encode the template's tags as, which startup checks
(constraints.validate_tokenizer, check_capabilities). It also holds the
reasoning efforts that requests, templates and the CLI name. It imports only
the standard library, as the installer reads it through families.py and the
launcher through serve_options."""

import json
from dataclasses import dataclass

# The tags Qwen's and Flint's templates frame reasoning and calls with, each
# one token of their tokenizers: a think block, and a call's opening and
# close. Another format states its own (ChatFormat.think_open, think_close).
THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
# The reasoning efforts a request, --reasoning-effort or
# --default-reasoning-effort may name, least first.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True, kw_only=True)
class ToolCallSyntax:
    """How a family's chat template lays a tool call out between call_tag
    and call_close: name_open, the function's name and name_close, then its
    arguments. Arguments of the "tags" kind are each argument, then
    arguments_close; an argument is key_open, its name and key_close, then
    value_open, its value and value_close. Arguments of the "json" kind are
    one JSON object, {"name": NAME, "args": ARGUMENTS}, as the call's whole
    text before call_close. Grammars write calls that way (tool_schema.py),
    and the projector reads them that way (output.py)."""

    name_open: str
    name_close: str
    key_open: str
    key_close: str
    value_open: str
    value_close: str
    arguments_close: str
    # Whether each of its tags is a token of the family's tokenizer, as
    # call_tag and call_close are in every family's, which grammars then
    # write as that token; otherwise they write the tags as text.
    tags_are_tokens: bool
    # The tags a call opens and closes with.
    call_tag: str = TOOL_CALL_OPEN
    call_close: str = TOOL_CALL_CLOSE
    # How its arguments are laid out: "tags" or "json".
    arguments: str = "tags"

    @property
    def call_open(self):
        """The text a call opens with, up to its function's name."""
        return self.call_tag + self.name_open

    def function_opening(self, name):
        """What follows call_tag in a call of tool `name`, up to its
        arguments."""
        return f"{self.name_open}{name}{self.name_close}"

    def json_prefix(self, name):
        """What a call of the "json" kind writes of tool `name` before its
        arguments, as the template writes it: its object's name, compact, as
        the template's tojson writes it, its characters unescaped."""
        name = json.dumps(name, ensure_ascii=False, separators=(",", ":"))
        return '{"name":' + name + ',"args":'

    @property
    def name_end(self):
        """The character a function's name ends at: the first of name_close,
        or where it has none, of the tags that follow the name, each of which
        opens with it."""
        return (self.name_close or self.key_open)[0]

    @property
    def argument_tags(self):
        """The tags that may follow an argument: the next one's opening, the
        arguments' close without the space its layout sets around it, and
        the call's close a model may write without the arguments'."""
        return tuple(
            tag
            for tag in (self.key_open, self.arguments_close.strip(), self.call_close)
            if tag
        )

    @property
    def value_ends(self):
        """Where a value ends: at its close where a tag that may follow an
        argument follows it, so a value may hold the tags in any other
        order."""
        return tuple(self.value_close + tag for tag in self.argument_tags)


# Qwen3-Coder's XML, as Qwen's templates lay a call out: a tag around the
# function and around each parameter, on lines of their own.
QWEN_XML = ToolCallSyntax(
    name_open="\n<function=",
    name_close=">\n",
    key_open="<parameter=",
    key_close=">",
    value_open="\n",
    value_close="\n</parameter>\n",
    arguments_close="</function>\n",
    tags_are_tokens=False,
)
# GLM-4.7's argument tags, as Flint's template lays a call out: the function's
# name right after TOOL_CALL_OPEN, then a key tag and a value tag for each
# argument, with nothing between the tags, each tag one token.
GLM_ARGS = ToolCallSyntax(
    name_open="",
    name_close="",
    key_open="<arg_key>",
    key_close="</arg_key>",
    value_open="<arg_value>",
    value_close="</arg_value>",
    arguments_close="",
    tags_are_tokens=True,
)
# Inkling 2's, as its template lays a call out: the model's message named for
# the function, <|message_model|>NAME<|content_invoke_tool_json|>, then the
# call as one JSON object, {"name":NAME,"args":{...}} with its keys sorted,
# closed by <|end_message|>, each tag a token.
INKLING2_JSON = ToolCallSyntax(
    name_open="",
    name_close="<|content_invoke_tool_json|>",
    key_open="",
    key_close="",
    value_open="",
    value_close="",
    arguments_close="",
    tags_are_tokens=True,
    call_tag="<|message_model|>",
    call_close="<|end_message|>",
    arguments="json",
)


@dataclass(frozen=True, kw_only=True)
class ChatFormat:
    """What the server reads, writes and checks in a family's tokens: the
    blocks its model writes, each a kind's opening, then its text, then its
    close (the reasoning, the answer's text where the format frames it, a
    tool call in call_syntax's layout), and the tokens that end a turn."""

    # The texts its tokenizer encodes as the engine's stop tokens (Ready's
    # stop_token_ids), one token each; the tokenizer's EOS is one of them.
    stop_tokens: tuple[str, ...]
    # The text that opens a reasoning block and the one that closes it.
    think_open: str = THINK_START
    think_close: str = THINK_END
    # The token of think_close.
    think_end_id: int
    # The tokens of think_open where the model opens its reasoning itself, as
    # a template without a thinking switch leaves it to (Flint's); none where
    # the generation prompt opens it, as Qwen's do by the request's reasoning
    # effort.
    think_start_ids: tuple[int, ...]
    # Whether the model may open a reasoning block again after text, where
    # reasoning starts again; otherwise only the output's start reasons, and
    # think_open after text is text.
    reasoning_returns: bool = False
    # The texts that open a block of the answer's text, each closed by
    # text_close, which the answer's text leaves out; none for a format whose
    # answer is the text outside other blocks. The first opens an answer.
    text_frames: tuple[str, ...] = ()
    text_close: str | None = None
    # The token of call_syntax.call_tag.
    call_open_id: int
    # The layout of a call.
    call_syntax: ToolCallSyntax
    # Whether a request's reasoning effort or enable_thinking reaches the
    # model: the generation prompt opens the reasoning or leaves it out
    # (Qwen's), or the template tells the model the effort (Inkling 2's). A
    # model told nothing decides each turn, whatever the request says
    # (Flint's). Whether the effort turns the reasoning on and off is
    # prompt_switches_reasoning.
    reasoning_switch: bool
    # Whether a call's opening also ends the reasoning where a call may
    # follow. A Qwen model may call a tool in the think block its generation
    # prompt opened, without closing it; a model that opens its reasoning
    # itself closes it before it calls one, so there a call's opening is
    # reasoning, as Flint's own parser reads it.
    calls_end_reasoning: bool
    # Whether the model opens its own message: its template appends no
    # generation prompt, and a prompt ends with the last message's close.
    opens_own_message: bool = False
    # The added tokens the format's blocks are framed with where they are
    # special tokens, which decoding would leave out: the text holds each,
    # as it holds a tag that is not special. Never a stop token.
    visible_tags: tuple[str, ...] = ()
    # The template's special token after which prompt tokenization reuses a
    # cached prompt's tokens (tokenization.PromptTokenizer).
    message_boundary: str
    # The texts that open a message of a role outside a prompt's head, as the
    # template writes them, one for each role that follows it: the user's,
    # the assistant's and a tool result's. The head is the system or
    # developer messages and the tools a template renders at the prompt's
    # start, which other conversations of the same agent or app share,
    # however many blocks it renders them in; it ends where the first of
    # these begins (frontend._head_tokens).
    turn_openings: tuple[str, ...]
    # The template's placeholder for one image, one token, which the server
    # widens to the image's tokens; None for a template without one.
    image_pad: str | None

    @property
    def prompt_switches_reasoning(self):
        """Whether a request's effort turns the model's reasoning on and off:
        where it reaches the model through the generation prompt, which opens
        the reasoning or leaves it out (Qwen's). A model that opens its
        reasoning itself (think_start_ids) reasons as it decides, at most
        told the effort (Inkling 2's, which still reasons at none), and no
        generation starts in its reasoning, which a thinking budget caps."""
        return self.reasoning_switch and not self.think_start_ids


# The Qwen families' (Qwen3_8.hpp, Qwen3_6Moe.hpp, Qwen4Exp.hpp):
# <|endoftext|> and the turn end <|im_end|> as stop tokens, and a generation
# prompt that opens the think block when the request thinks.
QWEN = ChatFormat(
    stop_tokens=("<|endoftext|>", "<|im_end|>"),
    think_end_id=248069,
    think_start_ids=(),
    call_open_id=248058,
    call_syntax=QWEN_XML,
    reasoning_switch=True,
    calls_end_reasoning=True,
    message_boundary="<|im_end|>",
    # A tool result is a user turn.
    turn_openings=("<|im_start|>user", "<|im_start|>assistant"),
    image_pad="<|image_pad|>",
)
# Flint's: <|end_of_text|> and the role tokens <|user|>, <|observation|> and
# <|developer|> as stop tokens, and a generation prompt, <|assistant|> alone,
# that leaves reasoning to the model. Every chat prompt ends with the
# assistant's role token.
FLINT = ChatFormat(
    stop_tokens=("<|end_of_text|>", "<|user|>", "<|observation|>", "<|developer|>"),
    think_end_id=150007,
    think_start_ids=(150006,),
    call_open_id=150010,
    call_syntax=GLM_ARGS,
    reasoning_switch=False,
    calls_end_reasoning=False,
    message_boundary="<|assistant|>",
    turn_openings=("<|user|>", "<|assistant|>", "<|observation|>"),
    image_pad=None,
)
# Inkling 2's (Inkling2Loop.hpp), as its template and its tokenizer_config's
# response_template lay a turn out: the model opens each of its messages
# itself, <|message_model|> then a content kind, its text and
# <|end_message|>, every tag a special token; reasoning (<|content_thinking|>)
# may come again after text, the answer's text is <|content_text|>, and a
# tool error or a text-invoked tool's text is the answer's too, and a call is
# INKLING2_JSON. A turn ends with <|content_model_end_sampling|>, its EOS. The
# template tells the model the request's effort in a system message (none
# renders 0).
INKLING2 = ChatFormat(
    stop_tokens=("<|content_model_end_sampling|>",),
    think_open="<|message_model|><|content_thinking|>",
    think_close="<|end_message|>",
    think_end_id=200010,
    think_start_ids=(200001, 200008),
    reasoning_returns=True,
    text_frames=(
        "<|message_model|><|content_text|>",
        "<|message_model|><|content_tool_error|>",
        "<|message_model|><|content_invoke_tool_text|>",
    ),
    text_close="<|end_message|>",
    call_open_id=200001,
    call_syntax=INKLING2_JSON,
    reasoning_switch=True,
    calls_end_reasoning=False,
    opens_own_message=True,
    visible_tags=(
        "<|message_model|>",
        "<|content_thinking|>",
        "<|content_text|>",
        "<|content_tool_error|>",
        "<|content_invoke_tool_text|>",
        "<|content_invoke_tool_json|>",
        "<|end_message|>",
    ),
    message_boundary="<|end_message|>",
    turn_openings=("<|message_user|>", "<|message_model|>", "<|message_tool|>"),
    # Its vision tower is not served yet; its template writes an image as
    # <|content_image|><|unused_200054|>.
    image_pad=None,
)
