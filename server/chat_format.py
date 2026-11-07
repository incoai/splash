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

from dataclasses import dataclass

# The tags every family's template frames reasoning and calls with, each one
# token of its tokenizer: a think block, and a call's opening and close.
THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
# The reasoning efforts a request or --default-reasoning-effort may name,
# least first.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True, kw_only=True)
class ToolCallSyntax:
    """How a family's chat template lays a tool call out between
    TOOL_CALL_OPEN and TOOL_CALL_CLOSE: name_open, the function's name and
    name_close, then each argument, then arguments_close. An argument is
    key_open, its name and key_close, then value_open, its value and
    value_close. Grammars write calls that way (tool_schema.py), and the
    projector reads them that way (output.py)."""

    name_open: str
    name_close: str
    key_open: str
    key_close: str
    value_open: str
    value_close: str
    arguments_close: str
    # Whether each of its tags is a token of the family's tokenizer, as
    # TOOL_CALL_OPEN and TOOL_CALL_CLOSE are in every family's, which grammars
    # then write as that token; otherwise they write the tags as text.
    tags_are_tokens: bool

    @property
    def call_open(self):
        """The text a call opens with, up to its function's name."""
        return TOOL_CALL_OPEN + self.name_open

    def function_opening(self, name):
        """What follows TOOL_CALL_OPEN in a call of tool `name`, up to its
        arguments."""
        return f"{self.name_open}{name}{self.name_close}"

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
            for tag in (self.key_open, self.arguments_close.strip(), TOOL_CALL_CLOSE)
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


@dataclass(frozen=True, kw_only=True)
class ChatFormat:
    """What the server reads, writes and checks in a family's tokens."""

    # The texts its tokenizer encodes as the engine's stop tokens (Ready's
    # stop_token_ids), one token each; the tokenizer's EOS is one of them.
    stop_tokens: tuple[str, ...]
    # The token of THINK_END, which closes the reasoning.
    think_end_id: int
    # The token of THINK_START where the model opens its reasoning itself, as
    # its first token, as a template without a thinking switch leaves it to
    # (Flint's); None where the generation prompt opens it, as Qwen's do by
    # the request's reasoning effort.
    think_start_id: int | None
    # The token of TOOL_CALL_OPEN.
    call_open_id: int
    call_syntax: ToolCallSyntax
    # The template's special token after which prompt tokenization reuses a
    # cached prompt's tokens (tokenization.PromptTokenizer).
    message_boundary: str
    # The template's placeholder for one image, one token, which the server
    # widens to the image's tokens; None for a template without one.
    image_pad: str | None

    @property
    def reasoning_switch(self):
        """Whether a request turns the model's reasoning on and off, its
        effort or enable_thinking opening the generation prompt's think block
        or leaving it out; a model that opens its reasoning itself decides
        each turn, whatever the request says."""
        return self.think_start_id is None

    @property
    def calls_end_reasoning(self):
        """Whether a call's opening also ends the reasoning where a call may
        follow. A Qwen model may call a tool in the think block its
        generation prompt opened, without closing it; a model that opens its
        reasoning itself closes it before it calls one, so there a call's
        opening is reasoning, as Flint's own parser reads it."""
        return self.think_start_id is None


# The Qwen families' (Qwen3_8.hpp, Qwen3_6Moe.hpp, Qwen4Exp.hpp):
# <|endoftext|> and the turn end <|im_end|> as stop tokens, and a generation
# prompt that opens the think block when the request thinks.
QWEN = ChatFormat(
    stop_tokens=("<|endoftext|>", "<|im_end|>"),
    think_end_id=248069,
    think_start_id=None,
    call_open_id=248058,
    call_syntax=QWEN_XML,
    message_boundary="<|im_end|>",
    image_pad="<|image_pad|>",
)
# Flint's: <|end_of_text|> and the role tokens <|user|>, <|observation|> and
# <|developer|> as stop tokens, and a generation prompt, <|assistant|> alone,
# that leaves reasoning to the model. Every chat prompt ends with the
# assistant's role token.
FLINT = ChatFormat(
    stop_tokens=("<|end_of_text|>", "<|user|>", "<|observation|>", "<|developer|>"),
    think_end_id=150007,
    think_start_id=150006,
    call_open_id=150010,
    call_syntax=GLM_ARGS,
    message_boundary="<|assistant|>",
    image_pad=None,
)
