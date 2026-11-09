# from src.constraints import is_complete_number, is_number_prefix

# texts = ["", "-", "2", "2.", "2.0", "-42", ".", "2..", "2a"]

# for is_integer in [False, True]:
#     print("integer" if is_integer else "number")
#     for text in texts:
#         prefix = is_number_prefix(text, is_integer)
#         complete = is_complete_number(text, is_integer)
#         print(f"  {text!r:8} prefix={prefix!s:5} complete={complete}")

# from llm_sdk import Small_LLM_Model

# from src.constraints import allowed_number_tokens
# from src.vocabulary import build_token_table

# model = Small_LLM_Model()
# size = len(model.get_logits_from_input_ids([0]))
# table = build_token_table(model, size)

# cases = [
#     ("", False),
#     ("2", False),
#     ("2.", False),
#     ("2", True),
# ]
# for written, is_integer in cases:
#     allowed = allowed_number_tokens(written, is_integer, ",", table)
#     texts = sorted({table[i] for i in allowed})
#     kind = "integer" if is_integer else "number"
#     print(f"written={written!r} ({kind}): {len(allowed)} allowed")
#     print("   ", texts[:30])

from llm_sdk import Small_LLM_Model

from src.generator import generate_number
from src.io_utils import load_functions
from src.prompting import build_prompt
from src.vocabulary import build_token_table

model = Small_LLM_Model()
size = len(model.get_logits_from_input_ids([0]))
table = build_token_table(model, size)
functions = load_functions("data/input/functions_definition.json")

prompt = build_prompt(functions, "What is the sum of 265 and 345?")
so_far = prompt + 'fn_add_numbers", "parameters": {"a": '
ids = model.encode(so_far)[0].tolist()
a = generate_number(model, ids, False, ",", table)
print("a =", repr(a))

so_far += a + ', "b": '
ids = model.encode(so_far)[0].tolist()
b = generate_number(model, ids, False, "}", table)
print("b =", repr(b))
