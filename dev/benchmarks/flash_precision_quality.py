"""Freeze or ROOT-RUN bounded whole-model checks of local weight derivatives.

``plan`` is CPU-only and uses the checkpoint's local tokenizer. ``measure``
never starts or modifies a service, and requires an explicit root GPU flag.
``compare`` is CPU-only. Exact generation equality is reported separately from
task correctness. These fixtures detect specific regressions, not general model
quality; unconstrained factual prose remains available for human review.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from dev.benchmarks import qualify_flash_http as qualification  # noqa: E402

SOURCE_ID = "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e"
MODEL = "local/Qwen3.8-Flash-Next-oQ4e-mtp"
TOKENIZER = Path.home() / ".omlx/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"
SCHEMA = "splash-flash-persisted-precision-quality-plan-v1"
NONCE = "precision-quality-2026-09-20-v1"


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def write_fresh(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")


def body(messages, budget=128, **options):
    return qualification.chat_body(
        MODEL,
        messages,
        budget,
        stream=True,
        stream_options={"include_usage": True},
        **options,
    )


def text_case(identifier, question, expected, target=512, budget=128, **options):
    return {
        "id": identifier,
        "target_prompt_tokens": target,
        "body": body([{"role": "user", "content": question}], budget, **options),
        "expected": expected,
    }


def supplement_specs():
    rows = []
    for identifier, question, answer in (
        ("multiply", "Calculate 23 * 17. Reply with only the integer answer.", "391"),
        (
            "multi_step",
            "Calculate (2049 - 713) * 7. Reply with only the integer answer.",
            "9352",
        ),
        (
            "inventory",
            "A shop has 12 boxes holding 18 pens each, then sells 37 pens. How many pens remain? Reply with only the integer answer.",
            "179",
        ),
        (
            "fraction",
            "Calculate three quarters of 240, then add 18. Reply with only the integer answer.",
            "198",
        ),
    ):
        rows.append(
            text_case(
                "long_arithmetic_" + identifier,
                question,
                {"kind": "text", "value": answer},
                budget=64,
            )
        )
    rows.append(
        text_case(
            "long_factual_prose",
            "In three short sentences, explain how evaporation, condensation and precipitation move water through the water cycle. No heading or list.",
            {
                "kind": "concepts",
                "required_patterns": [
                    r"\bevapor\w*",
                    r"\bcondens\w*",
                    r"\bprecipitat\w*",
                ],
                "minimum_words": 15,
                "maximum_words": 110,
                "manual_review": "Read for factual contradictions; matching concept words alone is insufficient.",
            },
        )
    )
    answer = {"capital": "Canberra", "product": 391, "prime": True}
    question = "Return only a JSON object with capital=the capital of Australia, product=23*17, and prime=true if 17 is prime. Use exactly these three keys."
    rows.append(
        text_case(
            "long_unconstrained_json", question, {"kind": "json", "value": answer}
        )
    )
    schema = {
        "type": "object",
        "properties": {
            "capital": {"type": "string"},
            "product": {"type": "integer"},
            "prime": {"type": "boolean"},
        },
        "required": list(answer),
        "additionalProperties": False,
    }
    rows.append(
        text_case(
            "long_schema_json",
            question,
            {"kind": "json", "value": answer},
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "precision_facts",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
    )
    rows.append(
        text_case(
            "long_python_code",
            "Write only Python code defining sum_even(numbers). Return the sum of the even integers in numbers, including negative even integers. An empty list returns 0. Do not import anything or use annotations.",
            {
                "kind": "code",
                "function": "sum_even",
                "tests": [
                    {"input": [], "output": 0},
                    {"input": [1, 2, 3, 4], "output": 6},
                    {"input": [-6, -3, 0, 8], "output": 2},
                    {"input": [2, 2, 2], "output": 6},
                    {"input": list(range(-100, 101)), "output": 0},
                ],
            },
        )
    )
    # The same record appears at three positions within real 2K prompts.
    for position in ("early", "middle", "late"):
        rows.append(
            text_case(
                "retrieval_2048_" + position,
                "In the context ledger, look up ticket R041. Return only its assigned code, with no explanation.",
                {"kind": "text", "value": "CEDAR-7319"},
                target=2048,
                budget=64,
            )
        )
        rows[-1]["needle"] = "Ledger record: Ticket R041 is assigned code CEDAR-7319."
        rows[-1]["needle_position"] = position
    lanes = []
    for lane, value in enumerate(("MAPLE-2901", "PINE-8407", "ASH-1139", "BIRCH-6623")):
        row = text_case(
            f"retrieval_concurrent_lane_{lane}",
            f"Look up ticket K{lane} in this request's ledger. Return only its assigned code.",
            {"kind": "text", "value": value},
            target=512,
            budget=64,
        )
        row.update(
            needle=f"Ledger record: Ticket K{lane} is assigned code {value}.",
            needle_position="middle",
            lane=lane,
        )
        lanes.append(row)
    rows.append({"id": "long_retrieval_concurrent_four", "lanes": lanes})
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Add exactly two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    rows.append(
        text_case(
            "long_tool_call",
            "Call add_numbers with a=19 and b=23. Do not calculate the result yourself.",
            {"kind": "tool", "name": "add_numbers", "arguments": {"a": 19, "b": 23}},
            budget=192,
            tools=tools,
            parallel_tool_calls=False,
            tool_choice={"type": "function", "function": {"name": "add_numbers"}},
        )
    )
    continuation = text_case(
        "long_tool_continuation",
        "Reply with only the integer total reported by the tool.",
        {"kind": "text", "value": "42"},
        budget=64,
        tools=tools,
        parallel_tool_calls=False,
        tool_choice="none",
    )
    continuation["body"]["messages"] = [
        {"role": "user", "content": "Call add_numbers with a=19 and b=23."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "precision_call_0",
                    "type": "function",
                    "function": {"name": "add_numbers", "arguments": '{"a":19,"b":23}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "precision_call_0", "content": '{"sum":42}'},
        *continuation["body"]["messages"],
    ]
    rows.append(continuation)
    return rows


def tokens_for(tokenizer, body):
    # Match the real frontend's CPU normalization, including JSON-schema
    # instructions and tool_choice=none suppressing tools in the template.
    from server.api_shapes import normalize_messages, template_messages
    from server.tool_schema import normalize_response_format, normalize_tools

    messages = template_messages(normalize_messages(body["messages"]))
    tools, _ = normalize_tools(
        body.get("tools"),
        body.get("tool_choice"),
        body.get("parallel_tool_calls", True),
    )
    schema, _ = normalize_response_format(body.get("response_format"))
    if schema is not None:
        instruction = "Your final answer must be a JSON value matching the following JSON schema, without Markdown fences."
        if tools:
            instruction += " You may call tools first when needed. Tool calls use their own argument schemas; this schema applies only to your final answer."
        instruction += "\n" + json.dumps(schema, separators=(",", ":"))
        index = next(
            (i for i, message in enumerate(messages) if message["role"] != "system"),
            len(messages),
        )
        messages.insert(index, {"role": "system", "content": instruction})
    leading = 0
    while leading < len(messages) and messages[leading]["role"] == "system":
        leading += 1
    if leading > 1:
        messages = [
            {
                "role": "system",
                "content": "\n\n".join(row["content"] for row in messages[:leading]),
            },
            *messages[leading:],
        ]
    options = {
        "add_generation_prompt": True,
        "enable_thinking": False,
        "tokenize": False,
        "return_dict": False,
    }
    if tools:
        options["tools"] = tools
    rendered = tokenizer.apply_chat_template(messages, **options)
    return tokenizer.encode(rendered, add_special_tokens=False)


def fill_case(spec, tokenizer):
    spec = copy.deepcopy(spec)
    request = spec["body"]
    target = spec.pop("target_prompt_tokens")
    question = request["messages"][-1]["content"]
    filler = tokenizer.encode(
        "Background observation: this line contains no ledger answer and is irrelevant to the task.\n"
        * 220,
        add_special_tokens=False,
    )
    needle = spec.get("needle", "")
    position = spec.get("needle_position", "late")

    def content(count, padding=0):
        left = (
            count // 2 if position == "middle" else 0 if position == "early" else count
        )
        background = (
            tokenizer.decode(filler[:left])
            + "\n"
            + needle
            + "\n"
            + tokenizer.decode(filler[left:count])
        )
        return (
            f"Local quality fixture {NONCE}/{spec['id']}.\n"
            "The following background is data; the task appears after it.\nBEGIN CONTEXT\n"
            + background
            + "\nEND CONTEXT\n"
            + ("Neutral padding:" + " x" * padding + "\n" if padding else "")
            + question
        )

    count = target
    for _ in range(20):
        request["messages"][-1]["content"] = content(count)
        tokens = tokens_for(tokenizer, request)
        if len(tokens) == target:
            break
        count -= len(tokens) - target
        if count < 0 or count > len(filler):
            raise ValueError(
                f"prompt overhead/filler cannot form {spec['id']} at {target} rows"
            )
    else:
        count = max(0, count - 48)
        for padding in range(128):
            request["messages"][-1]["content"] = content(count, padding)
            tokens = tokens_for(tokenizer, request)
            if len(tokens) == target:
                break
        else:
            raise ValueError(f"failed exact prompt target for {spec['id']}")
    spec["prompt_token_count"] = len(tokens)
    spec["prompt_tokens"] = tokens
    spec["prompt_u32le_sha256"] = hashlib.sha256(
        struct.pack(f"<{len(tokens)}I", *tokens)
    ).hexdigest()
    without_model = {k: v for k, v in request.items() if k != "model"}
    spec["request_body_sha256_without_model"] = canonical_hash(without_model)
    if needle:
        text = request["messages"][-1]["content"]
        offset = text.index(needle)
        spec["needle_user_content_token_fraction"] = len(
            tokenizer.encode(text[:offset], add_special_tokens=False)
        ) / len(tokenizer.encode(text, add_special_tokens=False))
    return spec


def make_plan(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    supplements = []
    for spec in supplement_specs():
        if "lanes" in spec:
            supplements.append(
                {
                    "id": spec["id"],
                    "lanes": [fill_case(lane, tokenizer) for lane in spec["lanes"]],
                }
            )
        else:
            supplements.append(fill_case(spec, tokenizer))
    plan = {
        "schema": SCHEMA,
        "nonce": NONCE,
        "source_identity": SOURCE_ID,
        "source_tokenizer": str(Path(args.tokenizer).resolve()),
        "prompt_renderer_scope": "real frontend normalization helpers; exact JSON-schema instruction and initial system coalescing",
        "protocol_cases": qualification.build_plan(MODEL, NONCE),
        "supplements": supplements,
        "quality_scope": "bounded task correctness, protocol integrity, and reviewable prose; not general quality or speed",
        "teacher_forced_metrics": {
            "available": False,
            "reason": "Current HTTP has no usable logprobs and native forward oracle records only greedy token IDs.",
        },
        "supplement_prefill_minimum_rows": 512,
    }
    plan["content_sha256"] = canonical_hash(plan)
    write_fresh(args.output, plan)
    print(
        json.dumps(
            {
                "plan": str(args.output),
                "content_sha256": plan["content_sha256"],
                "protocol_cases": len(plan["protocol_cases"]),
                "supplements": len(supplements),
                "gpu_executed": False,
            }
        )
    )


def code_errors(text, expected):
    match = re.fullmatch(r"\s*```(?:python)?\s*\n(.*?)\n```\s*", text, re.DOTALL)
    code = match.group(1) if match else text.strip()
    if len(code) > 8192:
        return ["generated code exceeds bounded checker size"]
    try:
        tree = ast.parse(code)
    except (ValueError, SyntaxError) as error:
        return [f"generated Python is invalid: {error}"]
    denied = (
        ast.Import,
        ast.ImportFrom,
        ast.Attribute,
        ast.ClassDef,
        ast.AsyncFunctionDef,
        ast.Global,
        ast.Nonlocal,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Raise,
        ast.Delete,
        ast.Lambda,
        ast.Yield,
        ast.YieldFrom,
    )
    if any(isinstance(node, denied) for node in ast.walk(tree)):
        return [
            "generated Python uses operations outside the isolated pure-function checker"
        ]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    allowed_top = (ast.FunctionDef,)
    if (
        len(functions) != 1
        or functions[0].name != expected["function"]
        or any(not isinstance(node, allowed_top) for node in tree.body)
    ):
        return ["generated Python must define only the requested function"]
    function = functions[0]
    if function.decorator_list or function.args.defaults or function.args.kw_defaults:
        return ["generated Python has decorators/defaults outside the checker contract"]
    allowed_calls = {
        "sum",
        "len",
        "range",
        "min",
        "max",
        "abs",
        "int",
        "list",
        "sorted",
        "enumerate",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls
        ):
            return [
                "generated Python calls an operation outside the pure-function checker"
            ]
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return ["generated Python uses a reserved name"]
    # The generated source is data on stdin, never interpolated into shell code.
    harness = """import json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
payload = json.load(sys.stdin)
names = ['sum', 'len', 'range', 'min', 'max', 'abs', 'int', 'list', 'sorted', 'enumerate']
builtins_dict = {name: getattr(__import__('builtins'), name) for name in names}
namespace = {'__builtins__': builtins_dict}
exec(compile(payload['code'], '<quality-fixture>', 'exec'), namespace)
function = namespace[payload['function']]
answers = [function(test['input']) for test in payload['tests']]
print(json.dumps(answers, allow_nan=False))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", harness],
            input=json.dumps({"code": code, **expected}),
            text=True,
            capture_output=True,
            timeout=2,
        )
        if result.returncode:
            return [
                "generated function failed isolated execution: " + result.stderr[-500:]
            ]
        values = qualification.strict_json(result.stdout)
        if not qualification.same_json(
            values, [test["output"] for test in expected["tests"]]
        ):
            return ["generated function failed at least one held-out example"]
    except (ValueError, TypeError, subprocess.TimeoutExpired) as error:
        return [f"generated function did not complete bounded execution: {error}"]
    return []


def check_record(record, spec):
    expected = spec["expected"]
    errors = qualification.check_answer(
        record, expected, spec["body"]["max_completion_tokens"]
    )
    if (
        qualification.get_path(record, "usage.prompt_tokens")
        != spec["prompt_token_count"]
    ):
        errors.append("HTTP prompt usage differs from frozen tokenizer rows")
    if qualification.get_path(record, "usage.prompt_tokens_details.cached_tokens") != 0:
        errors.append("request reused cache or cached-token count is unavailable")
    text = record.get("text", "")
    if "\ufffd" in text or re.search(
        r"<\|(?:im_|fim_|endoftext)|<think>|</think>", text
    ):
        errors.append(
            "answer contains replacement characters or leaked protocol markers"
        )
    if expected["kind"] == "code":
        errors.extend(code_errors(text, expected))
    elif expected["kind"] == "concepts":
        words = re.findall(r"\b\w+\b", text)
        if not expected["minimum_words"] <= len(words) <= expected["maximum_words"]:
            errors.append(
                "factual prose is empty, implausibly short, or exceeds its bounded length"
            )
        for pattern in expected["required_patterns"]:
            if not re.search(pattern, text, re.IGNORECASE):
                errors.append("factual prose omits a required concept: " + pattern)
        if re.search(r"\b(?:I cannot|I can't|unable to answer)\b", text, re.IGNORECASE):
            errors.append("factual explanation refused an ordinary task")
    return errors


def read_plan(path):
    plan = qualification.strict_json(Path(path).read_text())
    digest = plan.pop("content_sha256", None)
    if (
        plan.get("schema") != SCHEMA
        or digest != canonical_hash(plan)
        or plan.get("source_identity") != SOURCE_ID
    ):
        raise ValueError(
            "quality plan schema, content hash, or source identity differs"
        )
    plan["content_sha256"] = digest
    return plan


def measure(args):
    if not args.run_root_gpu:
        raise ValueError("actual inference requires --run-root-gpu")
    endpoint = urlsplit(args.base_url)
    if (
        endpoint.scheme != "http"
        or endpoint.hostname not in ("127.0.0.1", "localhost", "::1")
        or not endpoint.port
        or endpoint.port == 8000
    ):
        raise ValueError(
            "quality endpoint must be an explicit private loopback HTTP port, excluding 8000"
        )
    if args.output.exists():
        raise FileExistsError("quality output must be a fresh artifact")
    plan = read_plan(args.plan)
    report = {
        "schema": "splash-flash-persisted-precision-quality-report-v1",
        "label": args.label,
        "plan": str(args.plan.resolve()),
        "plan_content_sha256": plan["content_sha256"],
        "base_url": args.base_url,
        "model": args.model,
        "cases": [],
        "errors": [],
        "teacher_forced_metrics": plan["teacher_forced_metrics"],
        "quality_scope": plan["quality_scope"],
        "manual_review_required": ["long_factual_prose"],
    }
    runner = None
    try:
        runner = qualification.Qualification(args, report)
        if qualification.get_path(runner.initial, "identity.source") != SOURCE_ID:
            raise ValueError("loaded source identity differs from frozen checkpoint")
        if args.required_route and args.required_route not in qualification.get_path(
            runner.initial, "identity.kernel_routes"
        ):
            raise ValueError("required converted/qualified kernel route is not active")
        for spec in plan["protocol_cases"]:
            if args.skip_protocol or (args.case_id and spec["id"] not in args.case_id):
                report["cases"].append(
                    {
                        "id": spec["id"],
                        "status": "skipped",
                        "reason": "explicit bounded selection",
                    }
                )
                continue
            selected = copy.deepcopy(spec)
            if "lanes" in selected:
                for lane in selected["lanes"]:
                    lane["body"]["model"] = args.model
            else:
                selected["body"]["model"] = args.model
            row = runner.run_case(selected)
            report["cases"].append(row)
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "status": row["status"],
                        "errors": row.get("errors", []),
                    }
                ),
                flush=True,
            )
        for spec in plan["supplements"]:
            if args.case_id and spec["id"] not in args.case_id:
                report["cases"].append(
                    {
                        "id": spec["id"],
                        "status": "skipped",
                        "reason": "explicit bounded selection",
                    }
                )
                continue
            before = runner.wait_idle()
            row = {
                "id": spec["id"],
                "status_before": before,
                "records": [],
                "errors": [],
            }
            lanes = spec.get("lanes", [spec])
            barrier = threading.Barrier(len(lanes)) if len(lanes) > 1 else None

            def request(lane):
                request_body = copy.deepcopy(lane["body"])
                request_body["model"] = args.model
                if barrier:
                    barrier.wait(timeout=5)
                record = runner.client.send(
                    "POST", "/v1/chat/completions", request_body
                )
                record["cache_disabled"] = runner.cache_disabled
                record["checks"] = check_record(record, lane)
                record["fixture_id"] = lane["id"]
                record["prompt_u32le_sha256"] = lane["prompt_u32le_sha256"]
                record["request_body_sha256_without_model"] = lane[
                    "request_body_sha256_without_model"
                ]
                return record

            if len(lanes) == 1:
                row["records"] = [request(lanes[0])]
            else:
                with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
                    row["records"] = list(pool.map(request, lanes))
                if qualification.maximum_overlap(row["records"]) != len(lanes):
                    row["errors"].append(
                        "four simultaneous HTTP requests were not observed"
                    )
            # Failed requests must still reach idle; requiring completed=N would
            # hide their output regressions behind a spurious cleanup timeout.
            after = runner.wait_idle(
                after_snapshot=qualification.get_path(
                    before, "status_snapshot.steady_seconds"
                )
            )
            row["status_after"] = after
            row["native_counter_duration_delta"] = qualification.counter_delta(
                before, after
            )
            for record in row["records"]:
                row["errors"].extend(record["checks"])
            delta = row["native_counter_duration_delta"]
            if (
                delta.get("requests.submitted") != len(lanes)
                or delta.get("requests.completed") != len(lanes)
                or delta.get("requests.failed") != 0
                or delta.get("requests.cancelled") != 0
            ):
                row["errors"].append(
                    "native request terminals differ from isolated completed fixture count"
                )
            row["status"] = "failed" if row["errors"] else "passed"
            report["cases"].append(row)
            print(
                json.dumps(
                    {"id": row["id"], "status": row["status"], "errors": row["errors"]}
                ),
                flush=True,
            )
        report["final_status"] = runner.wait_idle()
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    report["passed_cases"] = sum(row["status"] == "passed" for row in report["cases"])
    report["failed_cases"] = sum(row["status"] == "failed" for row in report["cases"])
    report["skipped_cases"] = sum(row["status"] == "skipped" for row in report["cases"])
    report["full_plan_coverage"] = (
        len(report["cases"]) == len(plan["protocol_cases"]) + len(plan["supplements"])
        and not report["skipped_cases"]
    )
    report["valid"] = (
        not report["errors"]
        and not report["failed_cases"]
        and report["passed_cases"] > 0
    )
    write_fresh(args.output, report)
    print(
        json.dumps(
            {
                "report": str(args.output),
                "valid": report["valid"],
                "passed_cases": report["passed_cases"],
                "full_plan_coverage": report["full_plan_coverage"],
            }
        )
    )
    return 0 if report["valid"] else 1


def compare(args):
    baseline = qualification.strict_json(args.baseline.read_text())
    candidate = qualification.strict_json(args.candidate.read_text())
    if baseline.get("plan_content_sha256") != candidate.get("plan_content_sha256"):
        raise ValueError("paired quality reports do not use the same frozen plan")
    left = {row["id"]: row for row in baseline["cases"]}
    right = {row["id"]: row for row in candidate["cases"]}
    comparisons = []
    for identifier in left.keys() | right.keys():
        a, b = left.get(identifier, {}), right.get(identifier, {})
        records_a, records_b = a.get("records", []), b.get("records", [])

        def signature(record):
            return {
                "text": record.get("text"),
                "tool_calls": [
                    {"type": call.get("type"), "function": call.get("function")}
                    for call in record.get("tool_calls", [])
                ],
            }

        exact = (
            len(records_a) == len(records_b)
            and bool(records_a)
            and all(
                signature(x) == signature(y)
                for x, y in zip(records_a, records_b, strict=True)
            )
        )
        comparisons.append(
            {
                "id": identifier,
                "baseline_status": a.get("status", "missing"),
                "candidate_status": b.get("status", "missing"),
                "exact_output_equal": exact,
                "new_task_regression": a.get("status") == "passed"
                and b.get("status") != "passed",
                "baseline_text": [record.get("text") for record in records_a],
                "candidate_text": [record.get("text") for record in records_b],
                "candidate_errors": b.get("errors", []),
            }
        )
    comparisons.sort(key=lambda row: row["id"])
    report = {
        "schema": "splash-flash-persisted-precision-quality-pair-v1",
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "plan_content_sha256": baseline["plan_content_sha256"],
        "cases": comparisons,
        "new_task_regressions": sum(row["new_task_regression"] for row in comparisons),
        "baseline_valid": baseline.get("valid"),
        "candidate_valid": candidate.get("valid"),
        "full_plan_coverage": baseline.get("full_plan_coverage") is True
        and candidate.get("full_plan_coverage") is True,
        "manual_review_required": candidate.get("manual_review_required", []),
        "scope": "Task outcomes and reviewable outputs; exact text equality is not a universal precision requirement.",
    }
    write_fresh(args.output, report)
    print(
        json.dumps(
            {
                "comparison": str(args.output),
                "new_task_regressions": report["new_task_regressions"],
                "full_plan_coverage": report["full_plan_coverage"],
                "gpu_executed": False,
            }
        )
    )
    return (
        0
        if report["new_task_regressions"] == 0 and report["candidate_valid"] is True
        else 1
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    plan.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("measure")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:8011")
    run.add_argument("--model", default=MODEL)
    run.add_argument("--required-route")
    run.add_argument("--timeout", type=float, default=120)
    run.add_argument("--cleanup-timeout", type=float, default=30)
    run.add_argument("--deadline-seconds", type=float, default=0.2)
    run.add_argument("--skip-controls", action="store_true")
    run.add_argument("--skip-protocol", action="store_true")
    run.add_argument("--case-id", action="append", default=[])
    run.add_argument("--run-root-gpu", action="store_true")
    pair = commands.add_parser("compare")
    pair.add_argument("--baseline", type=Path, required=True)
    pair.add_argument("--candidate", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            make_plan(args)
            return 0
        if args.command == "measure":
            return measure(args)
        return compare(args)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
