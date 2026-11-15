from src.constraints import scan_token

assert scan_token("abc", False) == (True, False, False)
assert scan_token("\\", False) == (True, True, False)
assert scan_token("\\\\", False) == (True, False, False)
assert scan_token('\\"', False) == (True, False, False)
assert scan_token('"', False) == (True, False, True)
assert scan_token('"', True) == (True, False, False)
assert scan_token("\\d", False)[0] is False
assert scan_token("d", True)[0] is False
assert scan_token('",', False)[0] is False
assert scan_token('",', True) == (True, False, False)
print("all passed")
