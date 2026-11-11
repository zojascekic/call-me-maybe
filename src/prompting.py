from src.models import FunctionDefinition


def build_prompt(functions: list[FunctionDefinition], request: str) -> str:
    lines: list[str] = []
    for fn in functions:
        params = ", ".join(
            f"{name}: {spec.type}" for name, spec in fn.parameters.items()
        )
        lines.append(f"- {fn.name}({params}): {fn.description}")
    return (
        "Available functions:\n"
        + "\n".join(lines)
        + f"\nRequest: {request}"
        + '\nCall: {"name": "'
    )
