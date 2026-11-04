import json
import numpy as np
from llm_sdk import Small_LLM_Model
model = Small_LLM_Model()

ids = model.encode('{"a": 2.0}')
print(ids)
print(type(ids))

for toke_id in ids[0].tolist():
    print(f"|{model.decode([toke_id])}|")

path = model.get_path_to_vocab_file()
print(path)
with open(path) as f:
    vocab = json.load(f)
print(type(vocab), len(vocab))
print(vocab["hello"])

for token, token_id in vocab.items():
    if token_id == 220:
        print(repr(token))

with_quote = [token for token in vocab if '"' in token]
print(len(with_quote))
print(with_quote[:10])

print("<|endoftext|>" in vocab)

for text in ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]:
    print(text, model.encode(text)[0].tolist())

ids = model.encode("The capital of France is")
input_ids = ids[0].tolist()
logits = model.get_logits_from_input_ids(input_ids)
print(type(logits))
print(len(logits))   # or len(...), depending on the type

scores = np.array(logits)
next_id = int(np.argmax(scores))
print(next_id, repr(model.decode([next_id])))
