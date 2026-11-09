
def allowed_name_tokens(
        written: str,
        targets: list[str],
        token_table: list[str]
        ) -> list[int]:
    allowed_ids: list[int] = []
    for token_id, text in enumerate(token_table):
        if not text:
            continue
        candidate = written + text
        if any(target.startswith(candidate) for target in targets):
            allowed_ids.append(token_id)
    return allowed_ids
