#!/bin/bash
m=/home/zscekic/goinfre/zscekic/moulinette
uv run python -m src --functions_definition $m/data/input/functions_definition.json --input $m/data/input/function_calling_tests.json --output $m/student_output.json || exit 1
cd $m && uv run python -m moulinette grade_student_answers --student_answer_path student_output.json --set public