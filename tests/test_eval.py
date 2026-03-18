"""Tests for the Morf evaluator."""
import pytest
from morf.parser import parse
from morf.eval import eval as morf_eval, value_of_term, EvalError
from morf.pretty import show_value


def run(src: str) -> str:
    """Parse, evaluate, and display the result of an Morf program."""
    program = parse(src)
    result = morf_eval(program.term)
    val = value_of_term(result)
    return show_value(val)


# ---------------------------------------------------------------------------
# Basic value tests
# ---------------------------------------------------------------------------

def test_unit():
    assert run("()") == "()"


def test_nat_zero():
    assert run("0") == "0"


def test_nat_literal():
    assert run("5") == "5"


def test_list_empty():
    assert run("[]") == "[]"


def test_list_literal():
    assert run("[1; 2; 3]") == "[1; 2; 3]"


def test_tuple():
    assert run("(1, 2)") == "(1, 2)"


# ---------------------------------------------------------------------------
# Simple iso evaluation
# ---------------------------------------------------------------------------

NAT_DEFS = """
type nat = Z | S of nat
"""

def test_add():
    src = NAT_DEFS + """
    let rec add = match with
    | (m, Z)   <-> (m, Z)
    | (m, S n) <-> let (m, n) = add (S m, n) in (m, S n)
    in
    add (3, 2)
    """
    assert run(src) == "(5, 2)"


def test_inv_basic():
    """inv on a simple iso."""
    src = NAT_DEFS + """
    let rec add = match with
    | (m, Z)   <-> (m, Z)
    | (m, S n) <-> let (m, n) = add (S m, n) in (m, S n)
    in
    inv add (5, 2)
    """
    assert run(src) == "(3, 2)"


def test_parity():
    src = NAT_DEFS + """
    type parity = Even | Odd
    let rec parity = match with
    | 0       <-> (Even, 0)
    | 1       <-> (Odd, 1)
    | S (S x) <-> let (b, x) = parity x in (b, S (S x))
    in
    parity 4
    """
    assert run(src) == "(Even, 4)"


def test_inv_parity():
    src = NAT_DEFS + """
    type parity = Even | Odd
    let rec parity = match with
    | 0       <-> (Even, 0)
    | 1       <-> (Odd, 1)
    | S (S x) <-> let (b, x) = parity x in (b, S (S x))
    in
    inv parity (Even, 6)
    """
    assert run(src) == "6"


# ---------------------------------------------------------------------------
# Higher-order isos
# ---------------------------------------------------------------------------

LIST_DEFS = """
type nat = Z | S of nat
type 'a list = Nil | Cons of 'a * 'a list
"""

def test_map_double():
    src = LIST_DEFS + """
    let rec double = match with
    | Z   <-> (Z, Z)
    | S n <-> let (n, n') = (n, n) in (S n, S n')
    in
    let rec map f = match with
    | []      <-> []
    | h :: t  <-> f h :: map f t
    in
    map double [1; 2; 3]
    """
    # map double [1;2;3] = [(1,1);(2,2);(3,3)]
    assert run(src) == "[(1, 1); (2, 2); (3, 3)]"


def test_inv_map():
    src = LIST_DEFS + """
    let rec double = match with
    | Z   <-> (Z, Z)
    | S n <-> let (n, n') = (n, n) in (S n, S n')
    in
    let rec map f = match with
    | []      <-> []
    | h :: t  <-> f h :: map f t
    in
    inv {map double} [(1, 1); (2, 2)]
    """
    assert run(src) == "[1; 2]"


# ---------------------------------------------------------------------------
# End-to-end example files
# ---------------------------------------------------------------------------

def test_example_test(tmp_path):
    """test.morf → ([2; 0; 1], [[True; False]; []; [False]])"""
    result = run(open("examples/test.morf").read())
    assert result == "([2; 0; 1], [[True; False]; []; [False]])"


def test_example_nat():
    """nat.morf → ((55, 89), 5)"""
    result = run(open("examples/nat.morf").read())
    assert result == "((55, 89), 5)"


def test_example_list():
    """list.morf → [0; 0; 1; 1; 2; 2; 3; 3; 4; 4]"""
    result = run(open("examples/list.morf").read())
    assert result == "[0; 0; 1; 1; 2; 2; 3; 3; 4; 4]"


def test_example_run_length():
    """run_length.morf → [2; 2; 2; 2; 1; 5; 5; 5]"""
    result = run(open("examples/run_length.morf").read())
    assert result == "[2; 2; 2; 2; 1; 5; 5; 5]"


def test_example_isort():
    """isort.morf sorted part → [0; 1; 2; 4; 5; 5; 5; 6; 6; 7; 8; 9; 10]"""
    result = run(open("examples/isort.morf").read())
    # result is a pair (sorted, original)
    assert result.startswith("([0; 1; 2; 4; 5; 5; 5; 6; 6; 7; 8; 9; 10]")


# ---------------------------------------------------------------------------
# Reversibility property: eval(inv f (eval(f x))) == x
# ---------------------------------------------------------------------------

def test_inversion_property():
    """Inversion is a true inverse: applying f then inv f gives identity."""
    src = NAT_DEFS + """
    let rec add = match with
    | (m, Z)   <-> (m, Z)
    | (m, S n) <-> let (m, n) = add (S m, n) in (m, S n)
    in
    let r = add (2, 3) in inv add r
    """
    assert run(src) == "(2, 3)"
