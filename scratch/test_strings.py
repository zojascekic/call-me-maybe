from llm_sdk import Small_LLM_Model

from src.constraints import allowed_string_tokens
from src.io_utils import load_functions
from src.vocabulary import build_token_table

model = Small_LLM_Model()
size = len(model.get_logits_from_input_ids([0]))
table = build_token_table(model, size)
functions = load_functions("data/input/functions_definition.json")

allowed = allowed_string_tokens(table)
allowed_set = set(allowed)
print("allowed:", len(allowed), "of", len(table))
for text in ['"', '",', 'shrek', ' hello', '\n']:
    if text in table:
        print(repr(text), table.index(text) in allowed_set)
