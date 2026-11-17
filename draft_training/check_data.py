"""Check local tokenized data without loading teacher or draft weights."""

from .config import TrainingError
from .data import conversations, disjoint, examples, split_manifest


def check_dataset(tokenizer, path, recipe, supervision, block_size=8):
    from jinja2.exceptions import TemplateError

    if supervision == "target-generated" and recipe.max_new_tokens < block_size - 1:
        raise TrainingError(
            "max_new_tokens must allow at least seven assistant proposals"
        )
    supplied = supervision == "supplied-answers"
    report = {
        "examples": 0,
        "max_prompt_tokens": 0,
        "max_sequence_tokens": 0,
        "short_answers": 0 if supplied else None,
        "full_assistant_blocks": 0 if supplied else None,
    }
    errors, invalid = [], 0
    for i, (messages, answer) in enumerate(
        examples(conversations(path), supervision), 1
    ):
        report["examples"] += 1
        try:
            prefix = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                **recipe.chat_template_kwargs(),
            )
            length = len(prefix) + recipe.max_new_tokens
            if answer is not None:
                complete = tokenizer.apply_chat_template(
                    [*messages, {"role": "assistant", "content": answer}],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=False,
                    **recipe.chat_template_kwargs(),
                )
                length = len(complete)
                if complete[: len(prefix)] != prefix:
                    raise TrainingError("chat template supplied-answer prefix differs")
                if length <= len(prefix):
                    raise TrainingError("supplied answer has no assistant tokens")
                blocks = max(0, length - block_size - max(1, len(prefix) - 1) + 1)
                report["full_assistant_blocks"] += blocks
                report["short_answers"] += blocks == 0
            if length > recipe.sequence_length:
                raise TrainingError(
                    f"needs {length} tokens but sequence_length is {recipe.sequence_length}; shorten the example or increase the recipe limit"
                )
            report["max_prompt_tokens"] = max(report["max_prompt_tokens"], len(prefix))
            report["max_sequence_tokens"] = max(report["max_sequence_tokens"], length)
        except (ValueError, TypeError, RuntimeError, TemplateError) as error:
            invalid += 1
            if len(errors) < 5:
                errors.append(f"{path}: example {i}: {error}")
    if invalid:
        raise TrainingError(f"{invalid} invalid examples; " + "; ".join(errors))
    if not report["examples"]:
        raise TrainingError(
            f"{path}: no usable examples for {supervision}; supplied-answers needs a user turn immediately followed by an assistant answer"
        )
    if supplied and not report["full_assistant_blocks"]:
        raise TrainingError(
            f"{path}: no full assistant query blocks; supplied answers need at least seven scorable tokens"
        )
    return report


def load_tokenizer(target):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(target), local_files_only=True, trust_remote_code=False
    )


def check_datasets(target, data, validation_data, recipe, supervision, block_size=8):
    disjoint(split_manifest(data), split_manifest(validation_data))
    tokenizer = load_tokenizer(target)
    return {
        "mode": "tokenizer-only",
        "teacher_weights_loaded": False,
        "supervision": supervision,
        "chat_template": recipe.chat_template_kwargs(),
        "generated_lengths_include_max_new_tokens": supervision == "target-generated",
        "training": check_dataset(tokenizer, data, recipe, supervision, block_size),
        "validation": check_dataset(
            tokenizer, validation_data, recipe, supervision, block_size
        ),
    }
