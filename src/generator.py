import numpy as np
import numpy.typing as npt
import json
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition
from src.constraints import (allowed_name_tokens, allowed_number_tokens,
                             is_complete_number, allowed_string_tokens,
                             scan_token)


def next_token_scores(
        model: Small_LLM_Model,
        input_ids: list[int]
        ) -> npt.NDArray[np.float64]:
    """Get the model's scores for the next token.

    Args:
        model: The loaded language model.
        input_ids: The token IDs seen so far.

    Returns:
        One logit per token in the vocabulary.
    """
    logits = model.get_logits_from_input_ids(input_ids)
    scores = np.array(logits, dtype=np.float64)
    return scores


def pick_allowed_token(
        scores: npt.NDArray[np.float64],
        allowed_ids: list[int]
        ) -> int:
    """Pick the highest-scoring token among the allowed ones.

    Args:
        scores: One logit per token in the vocabulary.
        allowed_ids: IDs of the tokens that may be picked.

    Returns:
        The ID of the allowed token with the highest score.

    Raises:
        ValueError: If ``allowed_ids`` is empty.
    """
    if not allowed_ids:
        raise ValueError("No allowed tokens.")
    masked = np.full_like(scores, -np.inf)
    masked[allowed_ids] = scores[allowed_ids]
    return int(np.argmax(masked))


def choose_from(
        model: Small_LLM_Model,
        prompt_ids: list[int],
        targets: list[str],
        token_table: list[str]
        ) -> str:
    """Generate text that is forced to be one of the targets.

    Args:
        model: The loaded language model.
        prompt_ids: Token IDs of the prompt to continue.
        targets: The strings the output may become. None of them may be
            a prefix of another.
        token_table: Text of each token, indexed by token ID.

    Returns:
        The target the model generated.

    Raises:
        ValueError: If no token can continue toward a target.
    """
    input_ids = list(prompt_ids)
    written = ""

    while written not in targets:
        scores = next_token_scores(model, input_ids)
        allowed = allowed_name_tokens(written, targets, token_table)
        token_id = pick_allowed_token(scores, allowed)
        input_ids.append(token_id)
        written += token_table[token_id]

    return written


def choose_function_name(
        model: Small_LLM_Model,
        prompt_ids: list[int],
        names: list[str],
        token_table: list[str]
        ) -> str:
    """Let the model choose one of the function names.

    Each name is matched with its closing quote, so a name that is a
    prefix of another is still chosen correctly.

    Args:
        model: The loaded language model.
        prompt_ids: Token IDs of the prompt, ending just before the name.
        names: The available function names.
        token_table: Text of each token, indexed by token ID.

    Returns:
        The chosen function name.
    """
    targets = [name + '"' for name in names]
    return choose_from(model, prompt_ids, targets, token_table)[:-1]


def generate_number(
        model: Small_LLM_Model,
        prefix_ids: list[int],
        is_integer: bool,
        terminator: str,
        token_table: list[str],
        max_tokens: int = 50
        ) -> str:
    """Generate a number value with constrained decoding.

    Args:
        model: The loaded language model.
        prefix_ids: Token IDs of the text before the number.
        is_integer: If True, only an integer may be generated.
        terminator: The character that ends the number, such as ``,``
            or ``}``.
        token_table: Text of each token, indexed by token ID.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        The number as text, without the terminator. If no complete
        number is produced within ``max_tokens``, returns ``"0"``.
    """
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
            return written[:-1].strip()
    written = written.strip().rstrip(".")
    if is_complete_number(written, is_integer):
        return written
    return "0"


def generate_string(
        model: Small_LLM_Model,
        prefix_ids: list[int],
        token_table: list[str],
        allowed_normal: list[int],
        allowed_pending: list[int],
        max_tokens: int = 60
        ) -> str:
    """Generate the body of a JSON string with constrained decoding.

    Generation stops at the first unescaped closing quote.

    Args:
        model: The loaded language model.
        prefix_ids: Token IDs of the text up to the opening quote.
        token_table: Text of each token, indexed by token ID.
        allowed_normal: Tokens allowed when no escape is pending.
        allowed_pending: Tokens allowed right after a backslash.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        The raw string body, still escaped and without the quotes. If
        the limit is reached in the middle of an escape, the trailing
        backslash is dropped.
    """
    input_ids = list(prefix_ids)
    written = ""
    pending = False
    for _ in range(max_tokens):
        scores = next_token_scores(model, input_ids)
        allowed = allowed_pending if pending else allowed_normal
        token_id = pick_allowed_token(scores, allowed)
        text = token_table[token_id]
        _, pending, index = scan_token(text, pending)
        if index != -1:
            return written + text[:index]
        written += text
        input_ids.append(token_id)

    if pending:
        written = written[:-1]
    return written


def generate_parameters(
        model: Small_LLM_Model,
        prompt_text: str,
        function: FunctionDefinition,
        token_table: list[str]
        ) -> dict[str, float | int | str | bool]:
    """Generate a value for every parameter of a function.

    The parameters are generated one at a time, in order. Each value is
    added to the JSON text that the next parameter is generated from.

    Args:
        model: The loaded language model.
        prompt_text: The prompt, ending just before the function name.
        function: The chosen function.
        token_table: Text of each token, indexed by token ID.

    Returns:
        A mapping from each parameter name to its generated value.

    Raises:
        ValueError: If decoding reaches a state where no token is
            allowed.
    """
    params: dict[str, float | int | str | bool] = {}
    so_far = prompt_text + function.name + '", "parameters": {'
    names = list(function.parameters)

    allowed_normal = allowed_string_tokens(token_table, False, prompt_text)
    allowed_pending = allowed_string_tokens(token_table, True, prompt_text)
    for index, param in enumerate(names):
        spec = function.parameters[param]
        is_last = index == len(names) - 1
        terminator = "}" if is_last else ","
        so_far += f'"{param}":'

        if spec.type in ("number", "integer"):
            is_integer = True if spec.type == "integer" else False
            ids = model.encode(so_far)[0].tolist()
            text = generate_number(model, ids, is_integer,
                                   terminator, token_table)
            params[param] = int(text) if is_integer else float(text)
            so_far += " " + text + terminator
        elif spec.type == "boolean":
            so_far += " "
            ids = model.encode(so_far)[0].tolist()
            targets = ["true" + terminator, "false" + terminator]
            text = choose_from(model, ids, targets, token_table)[:-1]
            params[param] = text == "true"
            so_far += text + terminator
        else:
            so_far += ' "'
            ids = model.encode(so_far)[0].tolist()
            text = generate_string(model, ids, token_table,
                                   allowed_normal, allowed_pending)
            try:
                params[param] = json.loads('"' + text + '"')
            except json.JSONDecodeError:
                params[param] = text
            so_far += text + '"' + terminator
        if not is_last:
            so_far += " "
    return params


def default_parameters(
        function: FunctionDefinition
        ) -> dict[str, float | int | str | bool]:
    """Build placeholder values for every parameter of a function.

    Used when generation fails for a prompt.

    Args:
        function: The function to build values for.

    Returns:
        A mapping from each parameter name to a default for its type:
        ``0.0``, ``0``, ``False`` or ``""``. Unknown types get ``""``.
    """
    defaults: dict[str, float | int | str | bool] = {
        "number": 0.0, "integer": 0, "boolean": False, "string": ""
    }
    return {name: defaults.get(spec.type, "")
            for name, spec in function.parameters.items()}
