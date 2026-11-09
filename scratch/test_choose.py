import time

from llm_sdk import Small_LLM_Model

from src.generator import choose_function_name
from src.io_utils import load_functions, load_prompts
from src.prompting import build_prompt
from src.vocabulary import build_token_table

model = Small_LLM_Model()
size = len(model.get_logits_from_input_ids([0]))
table = build_token_table(model, size)

functions = load_functions("data/input/functions_definition.json")
prompts = load_prompts("data/input/function_calling_tests.json")
names = [fn.name for fn in functions]

for item in prompts:
    start = time.perf_counter()
    prompt_ids = model.encode(build_prompt(functions, item.prompt))[0].tolist()
    chosen = choose_function_name(model, prompt_ids, names, table)
    took = time.perf_counter() - start
    print(f"{took:5.2f}s  {chosen:35} {item.prompt}")
