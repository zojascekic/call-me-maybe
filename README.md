*This project has been created as part of the 42 curriculum by zscekic.*

# call me maybe

## Description

**call me maybe** translates natural-language requests into function calls,
using the small language model [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).

Given a list of function definitions such as:

```json
{"name": "fn_add_numbers", "description": "Add two numbers together",
 "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
 "returns": {"type": "number"}}
```

and a prompt such as `"What is the sum of 2 and 3?"`, the program outputs:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers",
 "parameters": {"a": 2.0, "b": 3.0}}
```

A 0.6B model is too small to produce valid JSON reliably on its own. This
project uses **constrained decoding**: at every step, the model may only
choose tokens that keep the output valid. The output is therefore always
parseable JSON, the function name always exists, and every parameter has the
type the schema asks for.

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/)

### Installation

```sh
make install        # same as: uv sync
```

The first run downloads the Qwen3-0.6B weights from Hugging Face.

### Running

```sh
make run            # same as: uv run python -m src
```

By default, the program reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, and writes
`data/output/function_calling_results.json`. You can choose other files:

```sh
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

### Other Makefile rules

| Rule               | What it does                                        |
| ------------------ | --------------------------------------------------- |
| `make debug`       | Runs the program under `pdb`                        |
| `make lint`        | Runs `flake8` and `mypy` with the required flags    |
| `make lint-strict` | Runs `flake8` and `mypy --strict`                   |
| `make clean`       | Removes `__pycache__`, `.mypy_cache` and other caches |

On 42 machines, the Makefile keeps the virtual environment and the caches in
`/goinfre` to stay under the home-directory quota. Use `make install` and
`make run` there: plain `uv sync` / `uv run` would put them in the home
directory instead.

## Example usage

```sh
$ make run
Loaded 5 functions with 11 prompts.
What is the sum of 2 and 3? -> fn_add_numbers
Greet shrek -> fn_greet
Reverse the string 'hello' -> fn_reverse_string
...
Wrote 11 results to data/output/function_calling_results.json
```

```json
[
  {
    "prompt": "What is the sum of 265 and 345?",
    "name": "fn_add_numbers",
    "parameters": {"a": 265.0, "b": 345.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  }
]
```

Errors are reported without a traceback:

```sh
$ uv run python -m src --input missing.json
Error: File not found: missing.json
```

## Algorithm explanation

### 1. Prompt

`src/prompting.py` builds a chat prompt in Qwen's format. It lists every
function with its parameter types and description, gives a few rules (copy
strings exactly, how to escape `\` and `"`, how to write regexes), and adds
the user's request. The assistant turn starts with an empty `<think>` block
(to skip Qwen3's reasoning mode) followed by `{"name": "`. The model only has
to continue from there.

### 2. Token table

`src/vocabulary.py` decodes every token ID once and stores its text in a list
(`token_table[i]` is the text of token `i`). All constraints work on this
table, so they never need to call the tokenizer during generation.

### 3. Constrained decoding loop

Each generation step does the same thing:

1. Get the logits for the next token with `get_logits_from_input_ids`.
2. Compute the set of **allowed** token IDs for the current state.
3. Set every other logit to `-inf` and take the `argmax` (greedy decoding).
4. Append the token and update the state.

The rules for step 2 depend on what is being generated
(`src/constraints.py`):

- **Function name:** a token is allowed if `written + token` is still a
  prefix of some `name"`. Adding the closing quote to each target means a
  name that is the prefix of another name is still chosen correctly.
  Generation stops when the text equals one target.
- **Numbers:** a token is allowed if `written + token` still matches
  `-?[0-9]*` (integers) or `-?([0-9]+(\.[0-9]*)?)?` (decimals), or if it
  completes the number and ends with the terminator (`,` before the next
  parameter, `}` after the last one).
- **Booleans:** the same mechanism as function names, with the targets
  `true,`/`false,` (or `true}`/`false}`).
- **Strings:** each token is scanned character by character. Only the escapes
  `\\` and `\"` are accepted, and an unescaped `"` ends the string. A
  `pending` flag remembers a backslash at the end of a token, so an escape
  split across two tokens is still handled. Characters must be printable, and
  non-ASCII characters are only allowed if they appear in the prompt. This
  blocks curly quotes and other languages, which the model otherwise drifts
  into. The raw string is then unescaped with `json.loads`.

### 4. Parameters

`generate_parameters` in `src/generator.py` fills the parameters one by one,
in schema order. The keys, `:`, `,` and quotes are written by the program,
not generated. After each value, the JSON written so far is added to the
context, so the model sees `"a": 2, "b":` when it generates `b`.

## Design decisions

- **The structure is written by code, only values are generated.** Keys,
  punctuation and quotes never come from the model, so the output cannot be
  malformed. The model only decides the function name and the values.
- **Greedy decoding.** For function calling, the most likely answer is the
  wanted one, and greedy decoding makes runs reproducible.
- **The token table is built once.** Decoding the full vocabulary takes a
  moment, but afterwards each step only filters a Python list.
- **The allowed sets for strings are computed once per prompt** (one for the
  normal state, one after a backslash), instead of once per token.
- **Tokens with a leading space are accepted in numbers** (`" 2"`), because
  Qwen often tokenizes numbers that way after `:`.
- **Pydantic models** (`src/models.py`) validate the input files and describe
  the output, as the subject requires.
- **Graceful fallback.** If generation fails for one prompt, the program
  prints a warning and writes default values (`0`, `0.0`, `false`, `""`) for
  it, so the other prompts are still processed and the output stays valid.
- **No heuristics choose the function.** The name is always picked by the
  model, among the names allowed by the constraint.

## Performance analysis

- **JSON validity:** 100%. The output is written with `json.dump` from
  values that were already checked against the schema types.
- **Accuracy:** 10/11 on the public set and 10/11 on the
  private moulinette set.
- **Speed:** about 2 minutes for the 11 public prompts on a 42 school PC
  (CPU only, including model loading), under the 300 s limit. Each generated token is one forward pass of the
  model, and string values are capped at 60 tokens and numbers at 50.
- **Reliability:** generation is deterministic (greedy decoding), so the same
  input gives the same output on every run.

## Challenges faced

- **Strings that never ended.** The model kept writing after the value (for
  example `Hello {user}'s profile!}}. замените...`) or closed it with a curly
  `”`. Fixed by limiting string characters to printable ASCII plus characters
  that appear in the prompt.
- **Escaped characters.** Windows paths (`C:\Users\...`) and quotes inside
  strings need `\\` and `\"`. A token can end in the middle of an escape, so
  the scanner carries a `pending` flag from one token to the next.
- **Negative numbers.** The first number pattern did not accept a leading
  `-`. Fixed by adding `-?` to both number patterns.
- **Booleans.** Added as a separate type that chooses between `true` and
  `false` with the same prefix mechanism as function names.
- **Regex parameters.** The model wrote a word (`vowels`) instead of a
  pattern. Fixed with prompt guidance and examples such as `[0-9]+` and
  `[aeiouAEIOU]`.
- **Function names that are prefixes of other names.** Solved by matching
  each name together with its closing quote.

## Testing strategy

- **Moulinette:** the public and private sets were run through the provided
  moulinette after every change, to track the score and catch regressions.
- **Custom inputs:** extra test files covered edge cases: an empty prompt, a
  request no function matches, a function with no parameters, boolean
  parameters, negative and very large numbers, Windows paths, quotes inside
  strings, and non-ASCII text.
- **Error handling:** missing files, invalid JSON, a non-array top level, and
  invalid function definitions were all checked to print a clear error
  instead of a traceback.
- **Static checks:** `make lint` and `make lint-strict` (flake8 and mypy) pass
  on the whole project.

## Resources


- [How to generate text: using different decoding methods (Hugging Face)](https://huggingface.co/blog/how-to-generate)
- [Byte-Pair Encoding tokenization (Hugging Face LLM course)](https://huggingface.co/learn/llm-course/en/chapter6/5)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Python regular expressions (W3Schools)](https://www.w3schools.com/python/python_regex.asp)
- [JSON specification](https://www.json.org/json-en.html)
- [NumPy `argmax`](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html)
- [uv documentation](https://docs.astral.sh/uv/)

### Use of AI

AI (Claude) was used for:

- Writing the Google-style docstrings in `src/` and the project description
  in `pyproject.toml`.
- Drafting this README from the code, the subject and the commit history.
- Explaining concepts (tokenization, logits, constrained decoding) and
  reviewing my code.
- Preparing the evaluation: a test checklist, extra test prompts and the
  steps to run the moulinette.
