"""Tool and response-format schema normalization and llguidance grammars."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from urllib.parse import unquote

from jsonschema.exceptions import SchemaError
from referencing import Registry

if __package__:
    from .errors import APIError
    from .schema_validation import build_validator
else:  # ``python server/server.py`` from the repo root.
    from errors import APIError
    from schema_validation import build_validator

MAX_JSON_NESTING = 256

# The chat template's tool-call framing. The projector, the parser and the
# grammars must agree byte for byte, so every piece is spelled here once.
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
FUNCTION_OPEN = "\n<function="
FUNCTION_CLOSE = "</function>\n</tool_call>"
PARAMETER_OPEN = "<parameter="
PARAMETER_CLOSE = "\n</parameter>\n"
THINK_END_TOKEN_ID = 248069  # the chat template's think-close token


LOCAL_REGISTRY = Registry()


def _reject_json_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def strict_json_loads(value):
    return json.loads(value, parse_constant=_reject_json_constant)


@dataclass(frozen=True)
class ToolPolicy:
    validators: dict
    schemas: dict
    required: bool
    parallel: bool
    namespaces: dict = field(default_factory=dict)

    @cached_property
    def argument_schemas(self):
        return {
            name: tool_argument_schema(schema) for name, schema in self.schemas.items()
        }


def json_value(value):
    value = value.strip()
    try:
        parsed = json.loads(value, parse_constant=str)
        pending = [(parsed, 0)]
        while pending:
            item, depth = pending.pop()
            if isinstance(item, dict):
                children = item.values()
            elif isinstance(item, list):
                children = item
            else:
                continue
            depth += 1
            if depth > MAX_JSON_NESTING:
                return value
            pending.extend((child, depth) for child in children)
        json.dumps(parsed, allow_nan=False)
        return parsed
    except (ValueError, RecursionError):
        return value


# JSON Schema keywords whose values are schemas: maps from names to schemas,
# then single schemas or lists of schemas. ``dependencies`` holds a schema or
# a list of property names per entry and is told apart by shape.
SCHEMA_MAP_KEYWORDS = {
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
}
SUBSCHEMA_KEYWORDS = {
    "items",
    "prefixItems",
    "additionalItems",
    "contains",
    "additionalProperties",
    "unevaluatedItems",
    "unevaluatedProperties",
    "propertyNames",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "contentSchema",
}


def _schemas(schema):
    """Yield ``schema`` and, depth first, every schema nested under it.

    Only schema positions are visited, so property names and literal const,
    enum, default and examples data are never mistaken for schemas.
    """
    yield schema
    if not isinstance(schema, dict):
        return
    for key, item in schema.items():
        if key in SCHEMA_MAP_KEYWORDS and isinstance(item, dict):
            children = item.values()
        elif key == "dependencies" and isinstance(item, dict):
            children = (child for child in item.values() if not isinstance(child, list))
        elif key in SUBSCHEMA_KEYWORDS:
            children = item if isinstance(item, list) else (item,)
        else:
            continue
        for child in children:
            yield from _schemas(child)


def _remote_ref(schema):
    for node in _schemas(schema):
        if isinstance(node, dict):
            if "$schema" in node and not isinstance(node["$schema"], str):
                raise APIError(400, "$schema must be a string")
            for key in ("$ref", "$dynamicRef", "$recursiveRef"):
                ref = node.get(key)
                if isinstance(ref, str) and not ref.startswith("#"):
                    return ref
    return None


SCHEMA_ANNOTATIONS = {
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
}


# Raw string parameters are framed by the tool-call grammar and validated
# against their complete JSON Schema after parsing. Keeping these assertions
# out of the grammar preserves multiline string payloads.
STRING_SCHEMA_POST_VALIDATION_KEYWORDS = {
    "allOf",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
}


def _grammar_compatible_schema(schema):
    """Guide generation with supported constraints; validate the original."""
    output = copy.deepcopy(schema)
    for node in _schemas(output):
        if isinstance(node, dict):
            node.pop("propertyNames", None)
            node.pop("pattern", None)
    if isinstance(output, dict):
        output["x-guidance"] = {"lenient": True}
    return output


def _lookup_tool_reference(ref, root):
    if not isinstance(ref, str) or not ref.startswith("#"):
        raise APIError(400, "unsupported tool parameter reference")
    fragment = unquote(ref[1:])
    if not fragment:
        return root
    if fragment.startswith("/"):
        current = root
        for part in fragment[1:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, list) and re.fullmatch(r"0|[1-9][0-9]*", part):
                index = int(part)
                if index < len(current):
                    current = current[index]
                    continue
            elif isinstance(current, dict) and part in current:
                current = current[part]
                continue
            raise APIError(400, f"unresolved tool parameter reference: {ref}")
        return current
    for node in _schemas(root):
        if isinstance(node, dict) and fragment in (
            node.get("$anchor"),
            node.get("$dynamicAnchor"),
        ):
            return node
    raise APIError(400, f"unresolved tool parameter reference: {ref}")


def _resolve_tool_schema(schema, root):
    seen = set()
    while isinstance(schema, dict) and "$ref" in schema:
        ref = schema["$ref"]
        constrained = bool(
            set(schema) - {"$ref", "$defs", "definitions"} - SCHEMA_ANNOTATIONS
        )
        if ref in seen:
            raise APIError(400, "cyclic direct tool parameter reference")
        seen.add(ref)
        current = _lookup_tool_reference(ref, root)
        if constrained:
            # Keep the reference and its intersecting assertions together in
            # the JSON grammar instead of choosing a raw-string encoding.
            return None
        schema = current
    return schema


def _schema_with_root(schema, root):
    def local_refs(value):
        for node in _schemas(value):
            ref = node.get("$ref") if isinstance(node, dict) else None
            if isinstance(ref, str) and ref.startswith("#"):
                yield node

    if next(local_refs(schema), None) is None:
        return schema
    root_defs = root.get("$defs", {}) if isinstance(root, dict) else {}
    schema_defs = schema.get("$defs", {}) if isinstance(schema, dict) else {}
    name = "__splash_root"
    while name in root_defs or name in schema_defs:
        name += "_"
    prefix = f"#/$defs/{name}"

    def rebase(value):
        output = copy.deepcopy(value)
        for node in local_refs(output):
            if node["$ref"] == "#" or node["$ref"].startswith("#/"):
                node["$ref"] = prefix + node["$ref"][1:]
        return output

    output = rebase(schema)
    definitions = dict(output.get("$defs", {}))
    definitions[name] = rebase(root)
    output["$defs"] = definitions
    return output


def raw_string_schema(schema, root):
    return _raw_string_schema(schema, root, frozenset())


def _raw_string_schema(schema, root, ancestors):
    schema = _resolve_tool_schema(schema, root)
    if not isinstance(schema, dict):
        return None
    if id(schema) in ancestors:
        raise APIError(400, "cyclic tool parameter alternatives without a nested value")
    ancestors = ancestors | {id(schema)}
    union = schema.get("anyOf", schema.get("oneOf"))
    if union is not None:
        options = []
        has_other_type = False
        allows_null = False
        for option_schema in union:
            resolved = _resolve_tool_schema(option_schema, root)
            null_only = isinstance(resolved, dict) and (
                resolved.get("type") == "null"
                or resolved.get("const", object()) is None
                or (
                    isinstance(resolved.get("enum"), list)
                    and resolved["enum"]
                    and all(value is None for value in resolved["enum"])
                )
            )
            if null_only:
                allows_null = True
                continue
            option = _raw_string_schema(option_schema, root, ancestors)
            if option is None:
                has_other_type = True
            else:
                options.append(option)
        if not options:
            return None
        if has_other_type:
            return None
        if allows_null:
            return None
        if set(schema) - {"anyOf", "oneOf"} - SCHEMA_ANNOTATIONS:
            return None
        if any(option[0] == "raw" for option in options):
            return "raw", None
        values = sum((option[1] for option in options), [])
        return "literal", values

    schema_type = schema.get("type")
    if isinstance(schema_type, list) and "string" in schema_type:
        if set(schema_type) - {"string", "null"}:
            return None
        if "null" in schema_type:
            return None
        schema_type = "string"
    values = schema.get("enum")
    if "const" in schema:
        values = [schema["const"]]
    if schema_type == "null":
        return None
    if schema_type != "string" and not (
        isinstance(values, list) and any(isinstance(value, str) for value in values)
    ):
        return None
    unsupported = (
        set(schema)
        - {
            "type",
            "enum",
            "const",
            "$defs",
            "definitions",
        }
        - SCHEMA_ANNOTATIONS
        - STRING_SCHEMA_POST_VALIDATION_KEYWORDS
    )
    if unsupported:
        return None
    if values is None:
        return "raw", None
    if any(value is not None and not isinstance(value, str) for value in values):
        return None
    if None in values:
        return None
    if any(
        isinstance(value, str) and PARAMETER_CLOSE.rstrip("\n") in value
        for value in values
    ):
        raise APIError(400, "string tool parameter enum contains XML framing")
    return "literal", values


def _schema_combination(keyword, values):
    identity = keyword == "allOf"
    values = [value for value in values if value is not identity]
    if not values:
        return identity
    if any(value is not identity and isinstance(value, bool) for value in values):
        return not identity
    if len(values) == 1:
        return values[0]
    combined = {keyword: values}
    # Keep the raw-string transport when every union branch, or at least one
    # intersection, requires a string. Other unions use JSON-encoded values.
    strings = [value.get("type") == "string" for value in values]
    if all(strings) if keyword == "anyOf" else any(strings):
        combined["type"] = "string"
    return combined


def tool_argument_schema(root):
    """Project object fields for XML framing; validate the untouched schema.

    Cross-field assertions remain on ToolPolicy.validators. This projection
    preserves the set of possible field values rather than choosing a branch
    before the model has supplied the discriminator or dependent properties.
    """

    def combine(shapes, union=False):
        if union:
            shapes = [shape for shape in shapes if shape is not None]
            if not shapes:
                return None
        elif any(shape is None for shape in shapes):
            return None
        if not shapes:
            return {"properties": {}, "required": [], "additionalProperties": True}
        names = dict.fromkeys(name for shape in shapes for name in shape["properties"])
        required = set(shapes[0]["required"])
        for shape in shapes[1:]:
            if union:
                required.intersection_update(shape["required"])
            else:
                required.update(shape["required"])
        keyword = "anyOf" if union else "allOf"
        return {
            "properties": {
                name: _schema_combination(
                    keyword,
                    [
                        shape["properties"].get(name, shape["additionalProperties"])
                        for shape in shapes
                    ],
                )
                for name in names
            },
            "required": sorted(required),
            "additionalProperties": _schema_combination(
                keyword, [shape["additionalProperties"] for shape in shapes]
            ),
        }

    def project(node, visiting):
        if node is False:
            return None
        if node is True:
            node = {}
        if not isinstance(node, dict):
            raise APIError(400, "tool parameters must allow a JSON object")
        kind = node.get("type", "object")
        if kind != "object" and not (isinstance(kind, list) and "object" in kind):
            return None
        properties = dict(node.get("properties", {}))
        additional = node.get("additionalProperties", True)
        patterns = list(node.get("patternProperties", {}).values())
        if patterns:
            additional = _schema_combination("anyOf", [additional, *patterns])
        required = node.get("required", [])
        for name in required:
            properties.setdefault(name, additional)
        shape = {
            "properties": properties,
            "required": required,
            "additionalProperties": additional,
        }
        shapes = [shape]
        ref = node.get("$ref")
        if ref is not None:
            if ref in visiting:
                raise APIError(400, "cyclic direct tool argument reference")
            resolved = _lookup_tool_reference(ref, root)
            shapes.append(project(resolved, visiting | {ref}))
        for child in node.get("allOf", []):
            shapes.append(project(child, visiting))
        for keyword in ("anyOf", "oneOf"):
            if keyword in node:
                shapes.append(
                    combine([project(child, visiting) for child in node[keyword]], True)
                )
        if "if" in node:
            shapes.append(
                combine(
                    [
                        project(node.get("then", {}), visiting),
                        project(node.get("else", {}), visiting),
                    ],
                    True,
                )
            )
        for child in node.get("dependentSchemas", {}).values():
            shapes.append(
                combine([project(child, visiting), project({}, visiting)], True)
            )
        choices = [node["const"]] if "const" in node else node.get("enum")
        if choices is not None:
            objects = [value for value in choices if isinstance(value, dict)]
            if not objects:
                return None
            shapes.append(
                combine(
                    [
                        {
                            "properties": {
                                name: {"const": value} for name, value in obj.items()
                            },
                            "required": list(obj),
                            "additionalProperties": False,
                        }
                        for obj in objects
                    ],
                    True,
                )
            )
        return combine(shapes)

    shape = project(root, set())
    if shape is None:
        raise APIError(400, "tool parameters must allow a top-level JSON object")
    shape["type"] = "object"
    shape["properties"] = {
        name: _schema_with_root(value, root)
        for name, value in shape["properties"].items()
    }
    shape["additionalProperties"] = _schema_with_root(
        shape["additionalProperties"], root
    )
    return shape


def _extra_parameter_names(names, rules):
    # A trie expresses the complement of declared names without lookaround,
    # which the grammar engine's regular-expression dialect does not support.
    trie = {}
    for name in names:
        node = trie
        for char in name:
            node = node.setdefault(char, {})
        node[None] = True
    counter = 0

    def emit(node, depth):
        nonlocal counter
        rule = f"extra_name_{counter}"
        counter += 1
        children = [char for char in node if char is not None]
        excluded = "".join(re.escape(char).replace("/", r"\/") for char in children)
        options = [f"/[^<>\\n\\r{excluded}][^<>\\n\\r]*/"]
        for char in children:
            options.append(f"{json.dumps(char)} {emit(node[char], depth + 1)}")
        if depth and None not in node:
            options.append("")
        rules.append(f"{rule}: " + " | ".join(options))
        return rule

    return emit(trie, 0)


def _tool_arguments_grammar(schema):
    return _argument_grammar(tool_argument_schema(schema))


def _parameter_rules(rule, prefix, value_schema):
    rules = []
    closing = json.dumps(PARAMETER_CLOSE)
    string_schema = raw_string_schema(value_schema, value_schema)
    value_schema = _grammar_compatible_schema(value_schema)
    if string_schema is None:
        rules.append(
            f"{rule}: {prefix} "
            f"%json {json.dumps(value_schema, separators=(',', ':'))} "
            f"{closing}"
        )
    elif string_schema[0] == "raw":
        value_rule = f"{rule}_value"
        rules.append(f"{rule}: {prefix} {value_rule}")
        rules.append(f"{value_rule}[suffix={closing}]: /(?s:.*)/")
    else:
        choices = []
        for choice_index, value in enumerate(string_schema[1]):
            text = "null" if value is None else value
            if text:
                choices.append(json.dumps(text))
            else:
                empty_rule = f"{rule}_empty_{choice_index}"
                rules.append(f"{empty_rule}:")
                choices.append(empty_rule)
        rules.append(f"{rule}: {prefix} ({' | '.join(choices)}) {closing}")
    return rules


def _argument_grammar(schema):
    properties = schema["properties"]
    required = schema["required"]
    rules = []
    sequence = []
    for index, (name, value_schema) in enumerate(properties.items()):
        if value_schema is False:
            if name in required:
                raise APIError(
                    400, f"required tool parameter cannot have a value: {name}"
                )
            continue
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or any(character in name for character in "<>\n\r")
        ):
            raise APIError(400, "invalid tool parameter name")
        rule = f"parameter_{index}"
        sequence.append(rule + ("" if name in required else "?"))
        prefix = json.dumps(f"{PARAMETER_OPEN}{name}>\n")
        rules.extend(_parameter_rules(rule, prefix, value_schema))
    additional = schema["additionalProperties"]
    if additional is not False:
        name_rule = _extra_parameter_names(properties, rules)
        prefix = f"{json.dumps(PARAMETER_OPEN)} {name_rule} {json.dumps('>' + chr(10))}"
        rules.extend(_parameter_rules("extra", prefix, additional))
        sequence.append("extra*")
    start = " ".join(sequence)
    return (
        "%llguidance {}\nstart:"
        + (" " + start if start else "")
        + "\n"
        + "\n".join(rules)
        + "\n"
    )


def json_grammar(schema, thinking):
    start = "start: " + ("think " if thinking else "") + "WS %json "
    grammar = [
        "%llguidance {}",
        start + json.dumps(_grammar_compatible_schema(schema), separators=(",", ":")),
    ]
    if thinking:
        grammar.append(f"think: TEXT <[{THINK_END_TOKEN_ID}]>")
        grammar.append("TEXT: /(.|\\n)*/")
    grammar.append("WS: /[ \\n\\r\\t]*/")
    return "\n".join(grammar) + "\n"


def normalize_response_format(value):
    if value in (None, {"type": "text"}):
        return None, None
    if not isinstance(value, dict):
        raise APIError(400, "response_format must be an object")
    kind = value.get("type")
    if kind == "json_object":
        schema = {"type": "object"}
    elif kind == "json_schema":
        wrapper = value.get("json_schema")
        schema = wrapper.get("schema") if isinstance(wrapper, dict) else None
        if not isinstance(schema, (dict, bool)):
            raise APIError(400, "response_format.json_schema.schema is required")
    else:
        raise APIError(400, "unsupported response_format")
    if ref := _remote_ref(schema):
        raise APIError(400, f"remote schema reference is not allowed: {ref}")
    try:
        validator = build_validator(schema, _schemas, LOCAL_REGISTRY)
    except SchemaError as error:
        raise APIError(400, f"invalid response schema: {error.message}") from error
    return schema, validator


def normalize_tools(tools, tool_choice, parallel, namespaces=None):
    if parallel is None:
        parallel = True
    if not isinstance(parallel, bool):
        raise APIError(400, "parallel_tool_calls must be a boolean")
    if tools is None:
        if tool_choice not in (None, "none", "auto"):
            raise APIError(400, "tool_choice requires tools")
        return None, None
    if not isinstance(tools, list):
        raise APIError(400, "tools must be an array")
    validators = {}
    schemas = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise APIError(400, "only function tools are supported")
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name) is None
        ):
            raise APIError(400, "tool name must match [A-Za-z0-9_-]{1,128}")
        if name in validators:
            raise APIError(400, f"duplicate tool name: {name}")
        schema = function.get("parameters")
        if schema is None:
            schema = {}
        if not isinstance(schema, (dict, bool)):
            raise APIError(400, f"invalid tool schema for {name}")
        if ref := _remote_ref(schema):
            raise APIError(400, f"remote tool schema reference is not allowed: {ref}")
        try:
            validators[name] = build_validator(schema, _schemas, LOCAL_REGISTRY)
            schemas[name] = schema
        except SchemaError as error:
            raise APIError(
                400, f"invalid tool schema for {name}: {error.message}"
            ) from error
    choice = "auto" if tool_choice is None else tool_choice
    if not tools:
        if choice not in ("auto", "none"):
            raise APIError(400, "tool_choice requires at least one tool")
        return None, None
    if choice == "none":
        return None, None
    if isinstance(choice, dict):
        function = choice.get("function", {})
        name = function.get("name") if isinstance(function, dict) else None
        if (
            choice.get("type") != "function"
            or not isinstance(name, str)
            or name not in validators
        ):
            raise APIError(400, "invalid named tool_choice")
        # The prompt keeps every tool; the grammar and validators force the call.
        validators = {name: validators[name]}
        schemas = {name: schemas[name]}
    elif choice not in ("auto", "required"):
        raise APIError(400, "invalid tool_choice")
    policy = ToolPolicy(
        validators,
        schemas,
        choice == "required" or isinstance(choice, dict),
        parallel,
        namespaces or {},
    )
    return tools, policy


THINK_END = "</think>"


def tool_grammar(policy, thinking, response_schema=None):
    side_grammars = []
    tag_rules = []
    for index, (name, schema) in enumerate(policy.argument_schemas.items()):
        grammar_name = f"arguments_{index}"
        side_grammars.append(
            {"name": grammar_name, "lark_grammar": _argument_grammar(schema)}
        )
        tag_rules.append(
            f"tool_{index}: {'WS' if response_schema is not None else 'TEXT'} {TOOL_CALL_OPEN} "
            f"{json.dumps(FUNCTION_OPEN + name + '>' + chr(10))} "
            f"@{grammar_name} {json.dumps(FUNCTION_CLOSE.removesuffix(TOOL_CALL_CLOSE))} "
            f"{TOOL_CALL_CLOSE}"
        )
    tool_choice = (
        "(" + " | ".join(f"tool_{index}" for index in range(len(tag_rules))) + ")"
    )
    thinking_prefix = "think " if thinking else ""
    if response_schema is not None:
        calls = tool_choice + ("+" if policy.parallel else "") + " WS"
        body = calls if policy.required else f"({calls} | answer)"
        start = f"start: {thinking_prefix}{body}"
    elif policy.required:
        body = tool_choice + ("+" if policy.parallel else "")
        start = f"start: {thinking_prefix}{body}"
    else:
        body = tool_choice + ("*" if policy.parallel else "?")
        start = f"start: {thinking_prefix}{body} tail"
    main = ["%llguidance {}", start]
    if response_schema is not None:
        main.extend(
            [
                "answer: WS %json "
                + json.dumps(
                    _grammar_compatible_schema(response_schema), separators=(",", ":")
                )
                + " WS",
                r"WS: /[ \n\r\t]*/",
            ]
        )
    if thinking:
        main.append(f"think: TEXT <[{THINK_END_TOKEN_ID}]>")
    main.extend(
        [
            "tail: TEXT",
            *tag_rules,
            r"TEXT: /(?s:.*)/ & ~/(?s:.*)(<tool_call>|<\/think>)(?s:.*)/",
        ]
    )
    side_grammars.insert(
        0, {"name": "tool_output", "lark_grammar": "\n".join(main) + "\n"}
    )
    return json.dumps({"grammars": side_grammars}, separators=(",", ":"))
