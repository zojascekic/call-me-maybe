# from src.models import FunctionDefinition

# # print(ParameterSpec(type="number"))
# # print(ParameterSpec.model_validate({"type": "string"}))
# # print(ParameterSpec.model_validate({"wrong": "x"}))

# import json
# with open("data/input/functions_definition.json") as f:
# 	data = json.load(f)

# print(type(data), len(data))
# print(data[0])
# functions = [FunctionDefinition.model_validate(item) for item in data]
# for fn in functions:
#     print(fn.name, {name: spec.type for name, spec in fn.parameters.items()})

# print("------------")
# broken = dict(data[0])
# del broken["name"]
# FunctionDefinition.model_validate(broken)


# import json

# from src.models import PromptItem

# with open("data/input/function_calling_tests.json") as f:
#     data = json.load(f)

# print(type(data), len(data))

# prompts = [PromptItem.model_validate(item) for item in data]
# for item in prompts:
#     print(repr(item.prompt))

"""Scratch tests for src/io_utils.py: real files plus broken ones.

Run from the project root:
    uv run python -m scratch.test_io
"""

import json
from pathlib import Path
from typing import Any, Callable

from src.io_utils import load_functions, load_json, load_prompts

# Adjust these two if your input files have different names.
REAL_FUNCTIONS = "data/input/functions_definition.json"
REAL_PROMPTS = "data/input/function_calling_tests.json"

FIXTURES = Path("scratch/fixtures")


def write_fixture(name: str, content: str) -> str:
    """Write a test file into scratch/fixtures and return its path."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def check(label: str, func: Callable[[str], Any], path: str,
          expect_error: bool) -> None:
    """Run one loader on one path and report whether it behaved as expected."""
    try:
        result = func(path)
        outcome = f"OK, loaded {len(result) if isinstance(result, list) else type(result).__name__}"
        passed = not expect_error
    except ValueError as error:
        outcome = f"ERROR: {error}".splitlines()[0]
        passed = expect_error
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}\n        {outcome}")


def main() -> None:
    """Create broken test files and run every loader against them."""
    broken = write_fixture("broken.json", '[{"prompt": "hi",')
    dict_file = write_fixture("dict.json", "{}")
    empty = write_fixture("empty.json", "[]")
    wrong_key = write_fixture("wrong_key.json", '[{"text": "hi"}]')

    with open(REAL_FUNCTIONS, encoding="utf-8") as f:
        functions = json.load(f)
    del functions[0]["name"]
    no_name = write_fixture("no_name.json", json.dumps(functions))

    print("--- load_json ---")
    check("real prompts file", load_json, REAL_PROMPTS, False)
    check("missing file", load_json, "does_not_exist.json", True)
    check("broken JSON", load_json, broken, True)
    check("folder instead of file", load_json, "data/input", True)

    print("\n--- load_functions ---")
    check("real functions file", load_functions, REAL_FUNCTIONS, False)
    check("dict instead of list", load_functions, dict_file, True)
    check("empty list", load_functions, empty, True)
    check("function without name", load_functions, no_name, True)

    print("\n--- load_prompts ---")
    check("real prompts file", load_prompts, REAL_PROMPTS, False)
    check("broken JSON", load_prompts, broken, True)
    check("dict instead of list", load_prompts, dict_file, True)
    check("wrong key", load_prompts, wrong_key, True)
    check("empty list (error if you chose to reject it)",
          load_prompts, empty, True)


main()
