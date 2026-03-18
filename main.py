#!/usr/bin/env python3
"""
Morf interpreter CLI.

Usage:
    python main.py <file.morf>
"""

import sys
from morf.parser import parse, ParseError
from morf.lexer import LexError
from morf.eval import eval as morf_eval, value_of_term, EvalError
from morf.inference import build_ctx, infer_program, TypeError as IsoTypeError
from morf.pretty import show_value, show_iso_type, show_base_type
from morf.ast import Generator


BOLD_RED    = "\x1b[1;31m"
BOLD_PURPLE = "\x1b[1;35m"
BOLD_CYAN   = "\x1b[1;36m"
RESET       = "\x1b[0m"


def run_file(path: str) -> None:
    # Read source
    try:
        with open(path) as f:
            src = f.read()
    except OSError as e:
        print(f"{BOLD_RED}Error{RESET}: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse
    try:
        program = parse(src)
    except (LexError, ParseError) as e:
        print(f"{BOLD_RED}Syntax error{RESET}: {e}", file=sys.stderr)
        sys.exit(1)

    # Type inference
    try:
        result_type, _, program_t = infer_program(program)
    except IsoTypeError as e:
        print(f"{BOLD_RED}Type error{RESET}: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine result type display
    from morf.inference import _base_of_any, _iso_of_any
    try:
        bt = _base_of_any(result_type)
        type_str = show_base_type(bt)
    except Exception:
        try:
            it = _iso_of_any(result_type)
            type_str = show_iso_type(it)
        except Exception:
            type_str = repr(result_type)

    # Evaluate
    try:
        result_term = morf_eval(program_t.term)
        result_val  = value_of_term(result_term)
    except EvalError as e:
        print(f"{BOLD_RED}Runtime error{RESET}: {e}", file=sys.stderr)
        sys.exit(1)

    val_str = show_value(result_val)
    print(f"{BOLD_CYAN}{val_str}{RESET} : {type_str}")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file.morf>", file=sys.stderr)
        sys.exit(1)
    run_file(sys.argv[1])


if __name__ == "__main__":
    main()
