import time

from llm_sdk import Small_LLM_Model

from src.constraints import allowed_name_tokens
from src.io_utils import load_functions
from src.vocabulary import build_token_table

model = Small_LLM_Model()
size = len(model.get_logits_from_input_ids([0]))
table = build_token_table(model, size)

functions = load_functions("data/input/functions_definition.json")
targets = [fn.name + '"' for fn in functions]

for written in ["", "fn_", "fn_greet"]:
    start = time.perf_counter()
    allowed = allowed_name_tokens(written, targets, table)
    took = time.perf_counter() - start
    print(f"written={written!r}: {len(allowed)} allowed, {took:.3f}s")
    print("   ", [table[i] for i in allowed[:20]])

