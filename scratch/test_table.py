from src.vocabulary import build_token_table
from llm_sdk import Small_LLM_Model
import time

model = Small_LLM_Model()
ids = model.encode("Answer only with JSON. What is the sum of 2 and 3?")
input_ids = ids[0].tolist()
logits = model.get_logits_from_input_ids(input_ids)
size = len(logits)

start = time.perf_counter()
table = build_token_table(model, size)
end = time.perf_counter()
print(f"Built in {end - start:.2f} seconds")

print(len(table))
print(repr(table[220]))
print(repr(table[14990]))
print(repr(table[12095]))
print(repr(table[151643]))
