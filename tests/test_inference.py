"""Tests for Morf type inference — check_pair linearity and infer_program lines."""
import pytest
from morf.parser import parse
from morf.inference import check_pair, infer_program, TypeError as IsoTypeError
from morf.ast import (
    VVar, VUnit, VTuple,
    EValue, ELetVal,
)


NAT_DEFS = "type nat = Z | S of nat\n"


# ---------------------------------------------------------------------------
# ③ check_pair linearity — behavioral regression tests
# ---------------------------------------------------------------------------

def test_check_pair_simple_linear():
    """| x <-> x  — linear use of a single variable."""
    check_pair(
        VVar("x"), EValue(VVar("x")), {}
    )


def test_check_pair_output_duplication_allowed():
    """| x <-> (x, x)  — variable appears twice in output value (allowed)."""
    check_pair(VVar("x"), EValue(VTuple((VVar("x"), VVar("x")))), {})


def test_check_pair_double_consume_raises():
    """Same variable used as iso input twice (two ELet) raises TypeError."""
    from morf.ast import ELet, IVar
    # | (x, y) <-> let a = f x in let b = g x in (a, b)
    # x is consumed by f, then consumed again by g → linearity violation
    with pytest.raises(IsoTypeError, match="already consumed"):
        check_pair(
            VTuple((VVar("x"), VVar("y"))),
            ELet(
                VVar("a"), IVar("f"), VVar("x"),
                ELet(
                    VVar("b"), IVar("g"), VVar("x"),
                    EValue(VTuple((VVar("a"), VVar("b"))))
                ),
            ),
            {},
        )


def test_check_pair_unknown_var_in_output_raises():
    """Referencing an unbound variable in the output raises TypeError."""
    # | x <-> y  — y never introduced
    with pytest.raises(IsoTypeError, match="not in context"):
        check_pair(VVar("x"), EValue(VVar("y")), {})


def test_check_pair_tuple_pattern_linear():
    """| (x, y) <-> (y, x)  — both variables used once."""
    check_pair(
        VTuple((VVar("x"), VVar("y"))),
        EValue(VTuple((VVar("y"), VVar("x")))),
        {},
    )


# ---------------------------------------------------------------------------
# ④ infer_program lines collection
# ---------------------------------------------------------------------------

def test_infer_program_collects_lines_for_iso():
    """infer_program must return non-empty lines when iso definitions exist."""
    src = NAT_DEFS + """
    let rec add = match with
    | (m, Z)   <-> (m, Z)
    | (m, S n) <-> let (m, n) = add (S m, n) in (m, S n)
    in
    add (2, 3)
    """
    _, lines, _ = infer_program(parse(src))
    assert len(lines) > 0, "Expected at least one collected iso type line"


def test_infer_program_line_contains_iso_name():
    """Each collected line should mention the iso name."""
    src = NAT_DEFS + """
    let rec add = match with
    | (m, Z)   <-> (m, Z)
    | (m, S n) <-> let (m, n) = add (S m, n) in (m, S n)
    in
    add (2, 3)
    """
    _, lines, _ = infer_program(parse(src))
    assert any("add" in line for line in lines)


def test_infer_program_line_format():
    """Collected lines must have the format '| <name> : <type>'."""
    src = NAT_DEFS + """
    let rec add = match with
    | (m, Z)   <-> (m, Z)
    | (m, S n) <-> let (m, n) = add (S m, n) in (m, S n)
    in
    add (2, 3)
    """
    _, lines, _ = infer_program(parse(src))
    for line in lines:
        assert line.startswith("| "), f"Expected line to start with '| ': {line!r}"
        assert " : " in line, f"Expected ' : ' in line: {line!r}"


def test_infer_program_multiple_isos_multiple_lines():
    """Multiple iso definitions produce multiple collected lines."""
    src = NAT_DEFS + """
    type parity = Even | Odd
    let rec parity_iso = match with
    | 0       <-> (Even, 0)
    | 1       <-> (Odd, 1)
    | S (S x) <-> let (b, x) = parity_iso x in (b, S (S x))
    in
    let rec double = match with
    | Z   <-> (Z, Z)
    | S n <-> let (n, n') = (n, n) in (S n, S n')
    in
    parity_iso 4
    """
    _, lines, _ = infer_program(parse(src))
    assert len(lines) >= 2, f"Expected >=2 lines, got {lines}"


def test_iso_to_let_simple():
    src = NAT_DEFS + """
    let rec add = match with
    | (m, Z)   <-> (m, Z)
    | (m, S n) <-> let (m, n) = add (S m, n) in (m, S n)
    in
    let f = {add} in
    f (2, 3)
    """
    program = parse(src)
    _, _, program_t = infer_program(program)
    from morf.eval import eval as morf_eval, value_of_term
    from morf.pretty import show_value
    result_term = morf_eval(program_t.term)
    val = value_of_term(result_term)
    assert show_value(val) == "5"


def test_iso_to_let_nested():
    src = NAT_DEFS + """
    let rec add = match with
    | (m, Z)   <-> (m, Z)
    | (m, S n) <-> let (m, n) = add (S m, n) in (m, S n)
    in
    let f = {add} in
    f (f (1, 1), 2)
    """
    program = parse(src)
    _, _, program_t = infer_program(program)
    from morf.eval import eval as morf_eval, value_of_term
    from morf.pretty import show_value
    result_term = morf_eval(program_t.term)
    val = value_of_term(result_term)
    assert show_value(val) == "4"


def test_iso_to_let_recursive():
    src = NAT_DEFS + """
    let rec add = match with
    | (m, Z)   <-> (m, Z)
    | (m, S n) <-> let (m, n) = add (S m, n) in (m, S n)
    in
    add (2, 3)
    """
    program = parse(src)
    _, _, program_t = infer_program(program)
    from morf.eval import eval as morf_eval, value_of_term
    from morf.pretty import show_value
    result_term = morf_eval(program_t.term)
    val = value_of_term(result_term)
    assert show_value(val) == "5"
