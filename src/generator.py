import numpy as np
import numpy.typing as npt
from llm_sdk import Small_LLM_Model


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
