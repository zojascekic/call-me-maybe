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
        pattern = r"-?[0-9]*"
    else:
        pattern = r"-?([0-9]+(\.[0-9]*)?)?"
    return re.fullmatch(pattern, text) is not None


def is_complete_number(text: str, is_integer: bool) -> bool:
    if is_integer:
        pattern = r"-?[0-9]+"
    else:
        pattern = r"-?[0-9]+(\.[0-9]+)?"
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
        candidate = (written + text).removeprefix(" ")
        if is_number_prefix(candidate, is_integer):
            allowed_ids.append(token_id)
        elif (is_complete_number(candidate[:-1], is_integer)
              and candidate.endswith(terminator)):
            allowed_ids.append(token_id)

    return allowed_ids


def scan_token(text: str, pending: bool) -> tuple[bool, bool, int]:
    for i, char in enumerate(text):
        if pending:
            if char == "\\" or char == '"':
                pending = False
            else:
                return False, False, -1
        elif char == "\\":
            pending = True
        elif char == '"':
            return True, False, i
    return True, pending, -1


def allowed_string_tokens(
        token_table: list[str],
        pending: bool,
        prompt_text: str
        ) -> list[int]:
    allowed_ids: list[int] = []
    for token_id, text in enumerate(token_table):
        if not text:
            continue
        ok, _, index = scan_token(text, pending)
        if not ok:
            continue
        kept = text if index == -1 else text[:index]
        if all(char.isprintable()
               and (char.isascii() or char in prompt_text)
               for char in kept):
            allowed_ids.append(token_id)
    return allowed_ids
