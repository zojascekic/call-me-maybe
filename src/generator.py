import numpy as np
import numpy.typing as npt
from llm_sdk import Small_LLM_Model
from src.constraints import allowed_name_tokens


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
