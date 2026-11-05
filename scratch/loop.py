import numpy as np
from llm_sdk import Small_LLM_Model
model = Small_LLM_Model()


ids = model.encode("Answer only with JSON. What is the sum of 2 and 3?")
input_ids = ids[0].tolist()
prompt_len = len(input_ids)
stop_ids = (151643, 151645)

for _ in range(60):
    logits = model.get_logits_from_input_ids(input_ids)
    scores = np.array(logits)
    next_id = int(np.argmax(scores))
    if next_id in stop_ids:
        break
    print(model.decode([next_id]), end="", flush=True)
    input_ids.append(next_id)
print("\n")
