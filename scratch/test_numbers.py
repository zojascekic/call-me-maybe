from src.constraints import is_complete_number, is_number_prefix

texts = ["", "-", "2", "2.", "2.0", "-42", ".", "2..", "2a"]

for is_integer in [False, True]:
    print("integer" if is_integer else "number")
    for text in texts:
        prefix = is_number_prefix(text, is_integer)
        complete = is_complete_number(text, is_integer)
        print(f"  {text!r:8} prefix={prefix!s:5} complete={complete}")
