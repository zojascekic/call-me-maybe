import argparse


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


args = parse_args()
print(args.function_definition)
print(args.input)
print(args.output)
