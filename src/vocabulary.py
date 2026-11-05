from llm_sdk import Small_LLM_Model


def build_token_table(model: Small_LLM_Model, size: int) -> list[str]:
    """Build a table mapping each token ID to its text.

    Args:
        model: The loaded language model.
        size: Number of token IDs to cover (the logits length).

    Returns:
        A list where position i holds the text of token i.
    """
    output: list[str] = []
    for token_id in range(size):
        output.append(model.decode([token_id]))
    return output
