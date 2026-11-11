from src.models import FunctionDefinition


def build_prompt(functions: list[FunctionDefinition], request: str) -> str:
    lines: list[str] = []
    for fn in functions:
        params = ", ".join(
            f"{name}: {spec.type}" for name, spec in fn.parameters.items()
        )
        lines.append(f"- {fn.name}({params}): {fn.description}")
    system = (
        "You turn a user request into exactly one function call. "
        'Answer with JSON only, in the form {"name": "<function>", '
        '"parameters": {...}}, using double quotes. '
        "Choose the function that does what the request asks. "
        "Copy string values exactly as they appear in the request, "
        "and keep each value as short as possible."
    )
    user = (
        "Available functions:\n"
        + "\n".join(lines)
        + f"\nRequest: {request}"
    )
    return (
        "<|im_start|>system\n" + system + "<|im_end|>\n"
        + "<|im_start|>user\n" + user + "<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        + '{"name": "'
    )
