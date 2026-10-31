"""The chat format of each model family Splash serves (install/families.py):
how its model frames reasoning and lays out tool calls, which the server
reads its output by and constrains it to, and the tokens it shares with the
engine and the tokenizer.

A format's token IDs are the engine's: the vocabulary and stop tokens of the
family's target layout (runtime/model/), and the tokens its tokenizer must
encode the template's tags as, which startup checks
(constraints.validate_tokenizer). It imports only the standard library, as
the installer reads it through families.py."""

from dataclasses import dataclass

# The tags every family's template frames reasoning with, each one token of
# its tokenizer: a think block's opening and close.
THINK_START = "<think>"
THINK_END = "</think>"
# The tags Qwen's and Flint's templates open and close each call with, each one
# token of their tokenizers.
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
# The token DeepSeek's DSML opens each of its tags with, after their < or </.
DSML = "｜DSML｜"
# The placeholder DeepSeek's template writes for an image.
DEEPSEEK_IMAGE = "<｜deepseek_image｜>"


@dataclass(frozen=True, kw_only=True)
class ToolCallSyntax:
    """How a family's chat template lays a tool call out between its
    call_tags: name_open, the function's name and name_close, then each
    argument, then arguments_close. An argument is key_open, its name and
    key_close, then value_open, its value and value_close. A template that
    writes a turn's calls in one block (DSML's) sets them between
    calls_open and calls_close, each followed by call_end. Grammars write
    calls that way (tool_schema.py), and the projector reads them that way
    (output.py)."""

    name_open: str
    name_close: str
    key_open: str
    key_close: str
    value_open: str
    value_close: str
    arguments_close: str
    # Whether each of its tags is a token of the family's tokenizer, as
    # TOOL_CALL_OPEN and TOOL_CALL_CLOSE are in Qwen's and Flint's, which
    # grammars then write as that token; otherwise they write the tags as
    # text.
    tags_are_tokens: bool
    # The tags around each call, each one token of the family's tokenizer that
    # grammars write as that token; none where the tags around its function's
    # name and arguments open and close it, as DSML's invoke does.
    call_tags: tuple[str, ...] = (TOOL_CALL_OPEN, TOOL_CALL_CLOSE)
    # The token every call's opening holds, which text outside calls cannot
    # spell: the format's call_open_id.
    marker: str = TOOL_CALL_OPEN
    # The key close before a value the template writes as JSON where it marks
    # such values apart from text, as DSML's string="false" does; key_close
    # is then the close before a text. None where it does not mark them.
    json_key_close: str | None = None
    # What the template writes as the arguments of a call that has none,
    # where that is not nothing (DSML's empty line), which grammars let come
    # before any call's arguments' close.
    empty_arguments: str = ""
    calls_open: str = ""
    calls_close: str = ""
    call_end: str = ""
    # What sets a function's namespace apart from its name where the template
    # reads one from the name, as DeepSeek's reads the text before a name's
    # first "::" (refuses_name). None where it reads none.
    namespace_separator: str | None = None

    @property
    def call_open(self):
        """The text a call opens with, up to its function's name."""
        return "".join(self.call_tags[:1]) + self.name_open

    @property
    def call_close(self):
        """The tag that closes a call after its arguments' close, if any."""
        return "".join(self.call_tags[1:])

    def function_opening(self, name):
        """What follows a call's opening tag (call_tags) in a call of tool
        `name`, up to its arguments."""
        return f"{self.name_open}{name}{self.name_close}"

    def opening(self, name):
        """The text a turn's first call opens with when it calls tool `name`,
        up to its arguments."""
        return (
            self.calls_open + "".join(self.call_tags[:1]) + self.function_opening(name)
        )

    def refuses_name(self, name):
        """Whether the template refuses a function's `name`, which it writes
        in the function's declaration and in each of its calls: where it
        reads a namespace from the name, one that holds namespace_separator
        more than once, since what follows the namespace must not hold it."""
        separator = self.namespace_separator
        return separator is not None and name.count(separator) > 1

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
    def block_tags(self):
        """The tags around a block of calls as text outside calls holds them:
        its opening with the space the template sets before it, which frames
        the calls as the space after them does, or without it, and its
        close."""
        if not self.calls_open:
            return ()
        return (
            self.calls_open.rstrip(),
            self.calls_open.strip(),
            self.calls_close.strip(),
        )

    @property
    def key_closes(self):
        """Where a parameter's name ends: at its key close, before a text or,
        where the template marks values written as JSON, before one."""
        return tuple(close for close in (self.key_close, self.json_key_close) if close)

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
# DeepSeek-V4.1's DSML: a turn's calls in one block after its text, an invoke
# tag around each call, the name in its attribute, and a parameter tag around
# each argument whose string attribute says whether the value is text or JSON.
# Each tag opens with < or </ and DSML, one token; the rest is text. The
# template reads a name's text before its first :: as a namespace.
DEEPSEEK_DSML = ToolCallSyntax(
    name_open=f'<{DSML} invoke name="',
    name_close='">\n',
    key_open=f'<{DSML} parameter name="',
    key_close='" string="true">',
    value_open="",
    value_close=f"</{DSML} parameter>\n",
    arguments_close=f"</{DSML} invoke>",
    tags_are_tokens=False,
    call_tags=(),
    marker=DSML,
    json_key_close='" string="false">',
    empty_arguments="\n",
    calls_open=f"\n\n<{DSML} calls>\n",
    calls_close=f"</{DSML} calls>",
    call_end="\n",
    namespace_separator="::",
)


@dataclass(frozen=True, kw_only=True)
class ChatFormat:
    """What the server reads, writes and checks in a family's tokens."""

    # The model's vocabulary, the rows of its logits that a constraint's
    # masks cover: its target layout's vocabularySize, which may exceed its
    # tokenizer's.
    vocabulary: int
    # The engine's stop tokens (its target layout's stopTokens), each with the
    # text its tokenizer encodes as that token; the tokenizer's EOS is one.
    stop_tokens: tuple[tuple[str, int], ...]
    # The token of THINK_END, which closes the reasoning.
    think_end_id: int
    # The token of THINK_START where the model opens its reasoning itself, as
    # its first token, as a template without a thinking switch leaves it to
    # (Flint's); None where the generation prompt opens it, as Qwen's do by
    # the request's reasoning effort.
    think_start_id: int | None
    # The token of its call syntax's marker, which each call's opening holds.
    call_open_id: int
    call_syntax: ToolCallSyntax
    # Whether a call's opening also ends the reasoning where a call may
    # follow. A Qwen model may call a tool in the think block its generation
    # prompt opened, without closing it. Flint's and DeepSeek's models close
    # their reasoning before they call one, so there a call's opening is
    # reasoning, as their own parsers read it.
    calls_end_reasoning: bool
    # The template's special token after which prompt tokenization reuses a
    # cached prompt's tokens (tokenization.PromptTokenizer).
    message_boundary: str
    # The placeholder the template writes for an image where it refuses any
    # message whose text holds it, as DeepSeek's does: a page or file an agent
    # read that quotes it would fail every later request of the conversation.
    # Requests render the template with a marker of their own in its place
    # (frontend.py). None where the template refuses no such text.
    refused_placeholder: str | None = None

    @property
    def stop_ids(self):
        return tuple(token for _text, token in self.stop_tokens)

    @property
    def reasoning_switch(self):
        """Whether a request turns the model's reasoning on and off: its
        reasoning effort or enable_thinking opens the think block in the
        generation prompt, or leaves it out. A model that opens its reasoning
        itself decides each turn whether to reason, whatever the request
        says."""
        return self.think_start_id is None


# The Qwen families' (Qwen3_8.hpp, Qwen3_6Moe.hpp, Qwen4Exp.hpp): a padded
# vocabulary, <|endoftext|> and the turn end <|im_end|> as stop tokens, and a
# generation prompt that opens the think block when the request thinks.
QWEN = ChatFormat(
    vocabulary=248320,
    stop_tokens=(("<|endoftext|>", 248044), ("<|im_end|>", 248046)),
    think_end_id=248069,
    think_start_id=None,
    call_open_id=248058,
    call_syntax=QWEN_XML,
    calls_end_reasoning=True,
    message_boundary="<|im_end|>",
)
# Flint's: a vocabulary its tokenizer fills, <|end_of_text|> and the role
# tokens <|user|>, <|observation|> and <|developer|> as stop tokens, and a
# generation prompt, <|assistant|> alone, that leaves reasoning to the model.
# Every chat prompt ends with the assistant's role token.
FLINT = ChatFormat(
    vocabulary=150144,
    stop_tokens=(
        ("<|end_of_text|>", 150001),
        ("<|user|>", 150003),
        ("<|observation|>", 150005),
        ("<|developer|>", 150018),
    ),
    think_end_id=150007,
    think_start_id=150006,
    call_open_id=150010,
    call_syntax=GLM_ARGS,
    calls_end_reasoning=False,
    message_boundary="<|assistant|>",
)
# DeepSeek-V4.1-Flash's: a vocabulary its tokenizer fills, the EOS
# <｜end▁of▁sentence｜> as the stop token, a generation prompt that opens the
# think block when the request thinks and closes it when it does not, calls
# in DSML, the assistant's role token before each of its turns, and a
# template that refuses its image placeholder in a message's text.
DEEPSEEK = ChatFormat(
    vocabulary=129280,
    stop_tokens=(("<｜end▁of▁sentence｜>", 1),),
    think_end_id=128822,
    think_start_id=None,
    call_open_id=128825,
    call_syntax=DEEPSEEK_DSML,
    calls_end_reasoning=False,
    message_boundary="<｜Assistant｜>",
    refused_placeholder=DEEPSEEK_IMAGE,
)
