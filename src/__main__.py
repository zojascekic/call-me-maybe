import argparse
import sys

from llm_sdk import Small_LLM_Model
from src.io_utils import load_functions, load_prompts, save_results
from src.vocabulary import build_token_table
from src.prompting import build_prompt
from src.generator import choose_function_name, generate_parameters
from src.models import FunctionCallResult


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        The parsed arguments: functions_definition, input and output paths.
    """
    parser = argparse.ArgumentParser(
        description="Translate prompt into function calls."
        )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions file."
    )
    parser.add_argument(
            "--input",
            default="data/input/function_calling_tests.json",
            help="Path to the prompts file."
        )
    parser.add_argument(
            "--output",
            default="data/output/function_calls.json",
            help="Path to the function calling results."
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(functions)} functions with {len(prompts)} prompts.")

    model = Small_LLM_Model()
    size = len(model.get_logits_from_input_ids([0]))
    token_table = build_token_table(model, size)
    names = [fn.name for fn in functions]
    by_name = {fn.name: fn for fn in functions}

    results: list[FunctionCallResult] = []
    for item in prompts:
        prompt_text = build_prompt(functions, item.prompt)
        prompt_ids = model.encode(prompt_text)[0].tolist()
        fn_name = choose_function_name(model, prompt_ids, names, token_table)
        print(f"{item.prompt} -> {fn_name}")

        chosen = by_name[fn_name]
        results.append(
            FunctionCallResult(
                prompt=item.prompt,
                name=fn_name,
                parameters=generate_parameters(model, prompt_text,
                                               chosen, token_table),
            )
        )

    try:
        save_results(args.output, results)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {len(results)} results to {args.output}")


main()
