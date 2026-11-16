import re


def allowed_name_tokens(
        written: str,
        targets: list[str],
        token_table: list[str]
        ) -> list[int]:
    """List the tokens that keep the output a prefix of some target.

    Args:
        written: The text generated so far.
        targets: The strings the output is allowed to become.
        token_table: Text of each token, indexed by token ID.

    Returns:
        IDs of the tokens that, appended to ``written``, still give a
        prefix of at least one target.
    """
    allowed_ids: list[int] = []
    for token_id, text in enumerate(token_table):
        if not text:
            continue
        candidate = written + text
        if any(target.startswith(candidate) for target in targets):
            allowed_ids.append(token_id)
    return allowed_ids


def is_number_prefix(text: str, is_integer: bool) -> bool:
    """Check whether text could be the start of a valid number.

    Args:
        text: The text to check. It may be empty.
        is_integer: If True, only integers are accepted. If False,
            decimals such as ``3.`` or ``-1.5`` are accepted too.

    Returns:
        True if more characters could turn ``text`` into a valid number.
    """
    if is_integer:
        pattern = r"-?[0-9]*"
    else:
        pattern = r"-?([0-9]+(\.[0-9]*)?)?"
    return re.fullmatch(pattern, text) is not None


def is_complete_number(text: str, is_integer: bool) -> bool:
    """Check whether text is a complete, valid number.

    Args:
        text: The text to check.
        is_integer: If True, only integers are accepted. If False,
            decimals with digits after the point are accepted too.

    Returns:
        True if ``text`` is a full number, such as ``-12`` or ``3.5``.
    """
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
    """List the tokens that keep the output a valid number.

    A token is allowed if it extends the number, or if it completes the
    number and ends with the terminator.

    Args:
        written: The number text generated so far.
        is_integer: If True, only integer tokens are allowed.
        terminator: The character that must follow the number, such as
            ``,`` or ``}``.
        token_table: Text of each token, indexed by token ID.

    Returns:
        IDs of the allowed tokens.
    """
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
    """Scan a token's text as part of a JSON string body.

    Only ``\\\\`` and ``\\"`` are accepted as escape sequences.

    Args:
        text: The token text.
        pending: True if the previous text ended with an unfinished
            backslash escape.

    Returns:
        A tuple ``(ok, pending, index)``. ``ok`` is False if the token
        contains an invalid escape. ``pending`` is True if the token ends
        with an unfinished backslash. ``index`` is the position of the
        unescaped closing quote, or -1 if there is none.
    """
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
    """List the tokens allowed inside a JSON string value.

    A token is allowed if its escapes are valid and every character
    before the closing quote is printable and either ASCII or present
    in the prompt.

    Args:
        token_table: Text of each token, indexed by token ID.
        pending: True if the string so far ends with an unfinished
            backslash escape.
        prompt_text: The full prompt. Non-ASCII characters are allowed
            only if they appear in it.

    Returns:
        IDs of the allowed tokens.
    """
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
