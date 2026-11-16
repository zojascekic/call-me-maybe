from src.constraints import scan_token

# No quote, no backslash
assert scan_token("abc", False) == (True, False, -1)

# Backslash handling
assert scan_token("\\", False) == (True, True, -1)
assert scan_token("\\\\", False) == (True, False, -1)
assert scan_token('\\"', False) == (True, False, -1)
assert scan_token('"', True) == (True, False, -1)
assert scan_token('",', True) == (True, False, -1)

# Closing quote: third value is its position
assert scan_token('"', False) == (True, False, 0)
assert scan_token('",', False) == (True, False, 0)
assert scan_token('abc"}', False) == (True, False, 3)

# Invalid escapes
assert scan_token("\\d", False)[0] is False
assert scan_token("d", True)[0] is False

print("all passed")
