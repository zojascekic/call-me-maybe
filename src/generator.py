import numpy as np
import numpy.typing as npt
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition
from src.constraints import (allowed_name_tokens, allowed_number_tokens,
                             is_complete_number)


def next_token_scores(
        model: Small_LLM_Model,
        input_ids: list[int]
        ) -> npt.NDArray[np.float64]:
    logits = model.get_logits_from_input_ids(input_ids)
    scores = np.array(logits, dtype=np.float64)
    return scores


def pick_allowed_token(
        scores: npt.NDArray[np.float64],
        allowed_ids: list[int]
        ) -> int:
    if not allowed_ids:
        raise ValueError("No allowed tokens.")
    masked = np.full_like(scores, -np.inf)
    masked[allowed_ids] = scores[allowed_ids]
    return int(np.argmax(masked))


def choose_function_name(
        model: Small_LLM_Model,
        prompt_ids: list[int],
        names: list[str],
        token_table: list[str]
        ) -> str:
    targets = [name + '"' for name in names]
    input_ids = list(prompt_ids)
    written = ""

    while not written.endswith('"'):
        scores = next_token_scores(model, input_ids)
        allowed = allowed_name_tokens(written, targets, token_table)
        token_id = pick_allowed_token(scores, allowed)
        input_ids.append(token_id)
        written += token_table[token_id]

    return written[:-1]


def generate_number(
        model: Small_LLM_Model,
        prefix_ids: list[int],
        is_integer: bool,
        terminator: str,
        token_table: list[str],
        max_tokens: int = 20
        ) -> str:
    input_ids = list(prefix_ids)
    written = ""

    for _ in range(max_tokens):
        scores = next_token_scores(model, input_ids)
        allowed = allowed_number_tokens(written, is_integer,
                                        terminator, token_table)
        token_id = pick_allowed_token(scores, allowed)
        input_ids.append(token_id)
        written += token_table[token_id]
        if written.endswith(terminator):
            return written[:-1]
    if is_complete_number(written, is_integer):
        return written
    raise ValueError(f"No complete number after "
                     f"{max_tokens} tokens: {written!r}")


def generate_parameters(
        model: Small_LLM_Model,
        prompt_text: str,
        function: FunctionDefinition,
        token_table: list[str]
        ) -> dict[str, float | int | str | bool]:
    params: dict[str, float | int | str | bool] = {}
    so_far = prompt_text + function.name + '", "parameters": {'
    names = list(function.parameters)

    for index, param in enumerate(names):
        spec = function.parameters[param]
        is_last = index == len(names) - 1
        terminator = "}" if is_last else ","
        so_far += f'"{param}": '

        if spec.type in ("number", "integer"):
            is_integer = True if spec.type == "integer" else False
            ids = model.encode(so_far)[0].tolist()
            text = generate_number(model, ids, is_integer,
                                   terminator, token_table)
            params[param] = int(text) if is_integer else float(text)
            so_far += text + terminator
        else:
            params[param] = ""
            so_far += '""' + terminator
        if not is_last:
            so_far += " "
    return params
