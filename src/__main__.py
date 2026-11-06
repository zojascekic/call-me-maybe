import argparse
import sys

from src.io_utils import load_functions, load_prompts, save_results
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
            default="data/output/function_calling_results.json",
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

    first = functions[0]
    results: list[FunctionCallResult] = []
    for item in prompts:
        results.append(
            FunctionCallResult(
                prompt=item.prompt,
                name=first.name,
                parameters={name: 0.0 for name in first.parameters},
            )
        )

    try:
        save_results(args.output, results)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {len(results)} results to {args.output}")


main()
