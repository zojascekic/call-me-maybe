from llm_sdk import Small_LLM_Model

from src.generator import next_token_scores, pick_allowed_token

model = Small_LLM_Model()
input_ids = model.encode("The capital of France is")[0].tolist()
scores = next_token_scores(model, input_ids)

everything = list(range(len(scores)))
print("all allowed:", pick_allowed_token(scores, everything))
print("only 14990 and 220:", pick_allowed_token(scores, [14990, 220]))

try:
    pick_allowed_token(scores, [])
except ValueError as error:
    print("empty list:", error)
