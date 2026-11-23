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

# The tags every family's template frames reasoning and calls with, each one
# token of its tokenizer: a think block, and a call's opening and close.
THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
# The reasoning efforts a request, --reasoning-effort or
# --default-reasoning-effort may name, least first.
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
class JsonCallSyntax:
    """How a family's chat template lays a tool call out as one JSON object in
    a frame of its own: call_open, the function's name, name_close, then
    {"name": NAME, "args": ARGUMENTS} and call_close, each tag a token of the
    family's tokenizer. Inkling's template writes the object compactly, the
    name first; grammars write calls that way (tool_schema.py), and the
    projector reads them that way (output.JsonCallProjector)."""

    call_open: str
    name_close: str
    call_close: str
    # The token of name_close, which only a call holds.
    name_close_id: int
    # The object's keys of the function's name and of its arguments.
    name_key: str = "name"
    arguments_key: str = "args"
    tags_are_tokens: bool = True

    def function_opening(self, name):
        """What follows call_open in a call of tool `name`, up to its
        arguments' value: the name, its close and the object's keys before
        the arguments, as the template writes them (tojson, compact and
        without escaping what is not ASCII)."""
        return (
            f"{name}{self.name_close}{{{json.dumps(self.name_key)}:"
            f"{json.dumps(name, ensure_ascii=False)},"
            f"{json.dumps(self.arguments_key)}:"
        )

    @property
    def name_end(self):
        """The character a function's name ends at: name_close's first."""
        return self.name_close[0]


@dataclass(frozen=True, kw_only=True)
class MessageFrames:
    """How a family's model frames each part of its turn as a block of its
    own, as Inkling's does: header, the token of the role the model writes
    itself, then a content token naming what the block holds, the block's
    text and close. A turn is its reasoning blocks (reasoning), its text
    blocks (text, or another of text_alike, which the template does not
    write but the model's own reader takes as text), its calls (the call
    syntax's frame, whose name follows header) and then the stop token. The
    model writes header itself: the template ends a prompt after its last
    block, with no generation prompt."""

    header: str
    reasoning: str
    text: str
    text_alike: tuple[str, ...]
    close: str
    header_id: int
    reasoning_id: int
    text_id: int
    close_id: int

    @property
    def content_tokens(self):
        """The content tokens after which a block's text starts, each with
        the field its text is read into."""
        return {self.reasoning: "reasoning_content"} | {
            token: "content" for token in (self.text, *self.text_alike)
        }

    @property
    def answer_opening(self):
        """What opens a block of the turn's text, the template's layout of a
        turn without reasoning, as its history renders one."""
        return self.header + self.text


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
    call_syntax: ToolCallSyntax | JsonCallSyntax
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
    # How the model frames its turn's parts as blocks, where it does
    # (Inkling's); None where reasoning and calls are tags in one text
    # (Qwen's and Flint's). A framed format's think_start_id, think_end_id and
    # call_open_id are the tokens of the frames' reasoning, close and header.
    frames: MessageFrames | None = None
    # The range of the numbers the template reads as a reasoning effort,
    # beside the named efforts, where it reads such a level (Inkling's
    # "Thinking effort level", from 0 to 0.99); None where it reads names
    # alone.
    effort_levels: tuple[float, float] | None = None

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

    @property
    def reasoning_effort(self):
        """Whether a request's reasoning effort reaches the model: where it
        switches the generation prompt's think block (reasoning_switch), or
        where the template renders it as a level the model reads
        (effort_levels), though the model opens its reasoning itself."""
        return self.reasoning_switch or self.effort_levels is not None

    def accepts_effort(self, effort):
        """Whether a request may name `effort`: one of REASONING_EFFORTS, or
        a number within effort_levels where the template reads one."""
        if isinstance(effort, str):
            return effort in REASONING_EFFORTS
        if self.effort_levels is None or isinstance(effort, bool):
            return False
        low, high = self.effort_levels
        return isinstance(effort, (int, float)) and low <= effort <= high

    @property
    def tags(self):
        """The tags the server reads the model's output by, each with the
        token the tokenizer must encode it as (constraints.validate_tokenizer):
        the frames' tokens and a call's name close where the model frames
        its turn, else the close of thinking, a call's opening and, where the
        model opens its reasoning itself, its opening."""
        if self.frames is not None:
            frames = self.frames
            return {
                frames.header: frames.header_id,
                frames.reasoning: frames.reasoning_id,
                frames.text: frames.text_id,
                frames.close: frames.close_id,
                self.call_syntax.name_close: self.call_syntax.name_close_id,
            }
        tags = {THINK_END: self.think_end_id, TOOL_CALL_OPEN: self.call_open_id}
        if self.think_start_id is not None:
            tags[THINK_START] = self.think_start_id
        return tags

    @property
    def decoded_tags(self):
        """The control tokens the server reads the output by, which decoding
        must keep as text: a framed format's frames, the content tokens it
        takes as text and its call's name close. None for the other
        formats, whose tags their tokenizers add as text already."""
        if self.frames is None:
            return ()
        frames = self.frames
        return (
            frames.header,
            frames.reasoning,
            frames.text,
            *frames.text_alike,
            frames.close,
            self.call_syntax.name_close,
        )

    @property
    def opens_turn(self):
        """Whether the model writes its turn's opening itself, so that a chat
        prompt ends with no generation prompt (Inkling's)."""
        return self.frames is not None

    @property
    def answer_opening(self):
        """What the server writes after a prompt rendered without reasoning so
        that the model's next token is its answer's first (scoring): where
        the model opens its reasoning itself, an empty think block, or a
        framed turn's text block opened (MessageFrames.answer_opening);
        nothing where the generation prompt of effort none closes the
        reasoning itself."""
        if self.frames is not None:
            return self.frames.answer_opening
        return THINK_START + THINK_END if self.think_start_id is not None else ""

    def call_opening(self, name):
        """The text a call of tool `name` opens with, up to its arguments."""
        if self.frames is not None:
            return self.call_syntax.call_open + self.call_syntax.function_opening(name)
        return TOOL_CALL_OPEN + self.call_syntax.function_opening(name)


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
    think_start_id=150006,
    call_open_id=150010,
    call_syntax=GLM_ARGS,
    message_boundary="<|assistant|>",
    turn_openings=("<|user|>", "<|assistant|>", "<|observation|>"),
    image_pad=None,
)
# Inkling 2's (Inkling.hpp): every part of a turn is a block the model frames
# itself, <|message_model|> then <|content_thinking|> or <|content_text|> and
# the block's text, or the function's name, <|content_invoke_tool_json|> and
# {"name":NAME,"args":{...}}, each closed by <|end_message|>; then
# <|content_model_end_sampling|>, its one stop token. The template ends every
# prompt after its last block, with no generation prompt, and renders the
# request's reasoning effort as a "Thinking effort level" system block (none
# 0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh and max 0.99, default
# 0.9, or a number from 0 to 0.99), which steers the reasoning the model
# opens itself but never turns it off. One placeholder, <|unused_200054|>,
# stands for each of an image's tokens.
INKLING_FRAMES = MessageFrames(
    header="<|message_model|>",
    reasoning="<|content_thinking|>",
    text="<|content_text|>",
    text_alike=("<|content_tool_error|>", "<|content_invoke_tool_text|>"),
    close="<|end_message|>",
    header_id=200001,
    reasoning_id=200008,
    text_id=200004,
    close_id=200010,
)
INKLING_JSON = JsonCallSyntax(
    call_open=INKLING_FRAMES.header,
    name_close="<|content_invoke_tool_json|>",
    call_close=INKLING_FRAMES.close,
    name_close_id=200049,
)
INKLING = ChatFormat(
    stop_tokens=("<|content_model_end_sampling|>",),
    think_end_id=INKLING_FRAMES.close_id,
    think_start_id=INKLING_FRAMES.reasoning_id,
    call_open_id=INKLING_FRAMES.header_id,
    call_syntax=INKLING_JSON,
    message_boundary=INKLING_FRAMES.close,
    # Each role's header. The template renders the request's effort as a
    # system block before the first message that is not a system message,
    # so a prompt's head ends after it: conversations share a head's state
    # at one effort.
    turn_openings=("<|message_user|>", INKLING_FRAMES.header, "<|message_tool|>"),
    image_pad="<|unused_200054|>",
    frames=INKLING_FRAMES,
    effort_levels=(0.0, 0.99),
)
