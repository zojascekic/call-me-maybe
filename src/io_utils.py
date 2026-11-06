import json
from typing import Any

from pydantic import ValidationError
from pathlib import Path

from src.models import FunctionDefinition, PromptItem, FunctionCallResult


def load_json(path: str) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path} (line {error.lineno})")
    except OSError:
        raise ValueError(f"Cannot read {path}")


def load_functions(path: str) -> list[FunctionDefinition]:
    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    if not data:
        raise ValueError(f"No functions are defined in {path}")

    try:
        return [FunctionDefinition.model_validate(item) for item in data]
    except ValidationError as error:
        raise ValueError(f"Invalid function definition "
                         f"in {path}: {error}") from None


def load_prompts(path: str) -> list[PromptItem]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    if not data:
        raise ValueError(f"No prompts are defined in {path}")
    try:
        return [PromptItem.model_validate(item) for item in data]
    except ValidationError as error:
        raise ValueError(f"Invalid prompt "
                         f"in {path}: {error}") from None


def save_results(path: str, results: list[FunctionCallResult]) -> None:
    data = [result.model_dump() for result in results]
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        raise ValueError(f"Cannot write output file: {path}") from None
