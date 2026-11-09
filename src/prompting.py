from src.models import FunctionDefinition


def build_prompt(functions: list[FunctionDefinition], request: str) -> str:
    fn_list = [f"{fn.name} - {fn.description}" for fn in functions]
    return (
        "Available functions:\n"
        + "\n".join(fn_list)
        + f"\nRequest: {request}"
        + '\nCall: {"name": "')
