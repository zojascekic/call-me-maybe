import json
from typing import Any
from pydantic import ValidationError
from src.models import FunctionDefinition


def load_json(path: str) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in {path}")
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
