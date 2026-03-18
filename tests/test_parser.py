"""Tests for the Morf parser."""
import pytest
from morf.parser import parse, ParseError
from morf.ast import (
    TVar, TCtor, TApp, TLetIso, TTuple, TCted, TUnit,
    VVar, VCtor, VTuple, VCted, VUnit,
    IVar, IApp, IFix, ILambda, IPairs, IInvert,
    EValue, ELet,
    BtNamed, BtUnit,
    VarValue, VarIso,
    nat_of_int,
)


def test_trivial_unit():
    p = parse("()")
    assert p.term == TUnit()


def test_trivial_ctor():
    p = parse("True")
    assert p.term == TCtor("True")


def test_nat_literal():
    p = parse("3")
    # 3 = S(S(S(Z)))
    assert p.term == TCted("S", TCted("S", TCted("S", TCtor("Z"))))


def test_simple_typedef():
    p = parse("type bool = False | True\nFalse")
    assert len(p.typedefs) == 1
    td = p.typedefs[0]
    assert td.type_name == "bool"
    assert len(td.variants) == 2
    assert td.variants[0] == VarValue("False")
    assert td.variants[1] == VarValue("True")


def test_typedef_with_payload():
    p = parse("type nat = Z | S of nat\nZ")
    td = p.typedefs[0]
    assert td.variants[1] == VarIso("S", BtNamed("nat"))


def test_typedef_with_type_param():
    p = parse("type 'a list = Nil | Cons of 'a * 'a list\nNil")
    td = p.typedefs[0]
    assert td.vars == ("'a",)
    assert td.type_name == "list"


def test_var_application():
    """f x → TApp(IVar("f"), TVar("x"))"""
    p = parse("f x")
    assert p.term == TApp(IVar("f"), TVar("x"))


def test_two_arg_application():
    """map f xs → TApp(IApp(IVar("map"), IVar("f")), TVar("xs"))"""
    p = parse("map f xs")
    assert p.term == TApp(IApp(IVar("map"), IVar("f")), TVar("xs"))


def test_inv():
    """inv f x → TApp(IInvert(IVar("f")), TVar("x"))"""
    p = parse("inv f x")
    assert p.term == TApp(IInvert(IVar("f")), TVar("x"))


def test_list_literal():
    """[1; 2] → Cons(1, Cons(2, Nil))"""
    p = parse("[1; 2]")
    expected = TCted("Cons", TTuple((
        TCted("S", TCtor("Z")),
        TCted("Cons", TTuple((
            TCted("S", TCted("S", TCtor("Z"))),
            TCtor("Nil"),
        ))),
    )))
    assert p.term == expected


def test_iso_rec():
    src = """
    type nat = Z | S of nat
    let rec add = match with
    | (m, Z) <-> (m, Z)
    | (m, S n) <-> let (m, n) = add (S m, n) in (m, S n)
    in
    add (3, 2)
    """
    p = parse(src)
    # Top level: TLetIso(phi="add", omega=IFix(...), term=TApp(...))
    from morf.ast import TLetIso
    assert isinstance(p.term, TLetIso)
    assert p.term.phi == "add"
    assert isinstance(p.term.omega, IFix)
    # Inner body: add (3, 2) → TApp(IVar("add"), TTuple(...))
    body = p.term.term
    assert isinstance(body, TApp)
    assert isinstance(body.omega, IVar)
    assert body.omega.name == "add"


def test_parse_all_examples():
    """All example files should parse without error."""
    import os
    examples_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
    for name in ("nat.morf", "list.morf", "test.morf", "isort.morf", "run_length.morf"):
        path = os.path.join(examples_dir, name)
        with open(path) as f:
            src = f.read()
        parse(src)  # should not raise
