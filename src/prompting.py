from src.models import FunctionDefinition


def build_prompt(functions: list[FunctionDefinition], request: str) -> str:
    """Build the chat prompt for one request.

    The prompt lists the available functions and ends with the start of
    the JSON answer (``{"name": "``), so the model continues with the
    function name.

    Args:
        functions: The available function definitions.
        request: The user's natural-language request.

    Returns:
        The full prompt text in the model's chat format.
    """
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
        "and keep each value as short as possible. "
        r'Inside a string, write a backslash as \\ and a double quote '
        r'as \", for example "C:\\temp\\a.txt". '
        "If a parameter is a regular expression, write a pattern "
        "that matches the described text, not a word: [0-9]+ matches "
        "numbers, [a-zA-Z]+ matches words and [aeiouAEIOU] matches "
        "vowels. If the request names a character in words, such as "
        "asterisk, dash or underscore, write the character itself: "
        "*, - or _."
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
