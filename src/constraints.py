import re


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


def is_number_prefix(text: str, is_integer: bool) -> bool:
    if is_integer:
        pattern = r"-?\d*"
    else:
        pattern = r"-?(\d+(\.\d*)?)?"
    return re.fullmatch(pattern, text) is not None


def is_complete_number(text: str, is_integer: bool) -> bool:
    if is_integer:
        pattern = r"-?\d+"
    else:
        pattern = r"-?\d+(\.\d+)?"
    return re.fullmatch(pattern, text) is not None


def allowed_number_tokens(
        written: str,
        is_integer: bool,
        terminator: str,
        token_table: list[str]
        ) -> list[int]:
    allowed_ids: list[int] = []
    for token_id, text in enumerate(token_table):
        if not text:
            continue
        candidate = written + text
        if is_number_prefix(candidate, is_integer):
            allowed_ids.append(token_id)
        elif (is_complete_number(candidate[:-1], is_integer)
              and candidate.endswith(terminator)):
            allowed_ids.append(token_id)

    return allowed_ids
