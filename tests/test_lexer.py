"""Tests for the Morf lexer."""
import pytest
from morf.lexer import tokenize, TK, LexError


def tks(src: str) -> list[TK]:
    return [t.kind for t in tokenize(src)]


def test_keywords():
    src = "let in fix type inv rec of fun match with unit"
    expected = [
        TK.LET, TK.IN, TK.FIX, TK.TYPE, TK.INVERT,
        TK.REC, TK.OF, TK.FUN, TK.MATCH, TK.WITH, TK.UNIT,
        TK.EOF,
    ]
    assert tks(src) == expected


def test_operators():
    src = "* | , ; :: -> <-> ="
    expected = [
        TK.TIMES, TK.PIPE, TK.COMMA, TK.SEMICOLON, TK.CONS,
        TK.ARROW, TK.BIARROW, TK.EQUAL, TK.EOF,
    ]
    assert tks(src) == expected


def test_brackets():
    assert tks("( ) { } [ ]") == [
        TK.LPAREN, TK.RPAREN, TK.LBRACE, TK.RBRACE,
        TK.LBRACKET, TK.RBRACKET, TK.EOF,
    ]


def test_identifiers():
    tokens = tokenize("hello World 'a 42")
    assert tokens[0].kind == TK.VAR and tokens[0].value == "hello"
    assert tokens[1].kind == TK.CTOR and tokens[1].value == "World"
    assert tokens[2].kind == TK.TVAR and tokens[2].value == "'a"
    assert tokens[3].kind == TK.NAT and tokens[3].value == 42


def test_comment_stripped():
    assert tks("(* this is a comment *) 42") == [TK.NAT, TK.EOF]


def test_comment_inline():
    assert tks("x (* comment *) y") == [TK.VAR, TK.VAR, TK.EOF]


def test_unknown_char():
    with pytest.raises(LexError):
        tokenize("@")


def test_var_with_prime():
    tokens = tokenize("x'")
    assert tokens[0].kind == TK.VAR and tokens[0].value == "x'"


def test_line_tracking():
    tokens = tokenize("x\ny")
    assert tokens[0].line == 1
    assert tokens[1].line == 2
