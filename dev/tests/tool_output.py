"""Tool-call grammars and output as a request produces them, for tests."""

from server import output, tool_schema
from server.chat_format import QWEN_XML


def argument_grammar(schema, syntax=QWEN_XML):
    """The argument grammar of a strict tool whose parameters are `schema`,
    framed within a request's budget as ToolPolicy frames it, in `syntax`'s
    tags (Qwen's unless a test passes another)."""
    policy = tool_schema.ToolPolicy(
        schemas={"tool": schema},
        required=False,
        parallel=True,
        strict=frozenset({"tool"}),
        constrained=True,
    )
    return tool_schema._argument_grammar(policy.argument_schemas["tool"], syntax)


def tool_policy(schemas):
    """The policy of a request that offers each tool named in `schemas` with
    its parameter schema."""
    tools = [
        {"type": "function", "function": {"name": name, "parameters": schema}}
        for name, schema in schemas.items()
    ]
    return tool_schema.normalize_tools(tools, "auto", True)[1]


def put(projector, text, size=None):
    """The events of putting `text` into `projector`, whole or in chunks of
    `size` characters."""
    size = size or len(text) or 1
    events = []
    for offset in range(0, len(text), size):
        events += projector.put(text[offset : offset + size])
    return events


def project(
    text, policy, request_id="test", incomplete=False, size=None, syntax=QWEN_XML
):
    """Project `text` as a request does, whole or in chunks of `size`
    characters, with its calls in `syntax`'s layout (Qwen's unless a test
    passes another), and finish it. Returns the content, the calls and the
    events, those the finish still owed last."""
    projector = output.StreamingToolCallProjector(policy, request_id, syntax=syntax)
    events = put(projector, text, size)
    content, calls, owed = projector.finish(incomplete)
    return content, calls, events + owed


def streamed_text(events):
    return "".join(value for kind, value in events if kind == "content")


def streamed_arguments(events):
    return "".join(
        value["function"].get("arguments", "")
        for kind, value in events
        if kind == "tool"
    )
