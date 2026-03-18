"""
Morf lexer (tokenizer).

Port of lexer.mll from the OCaml implementation.
Produces a flat list of tokens from source text.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

class TK(Enum):
    EOF       = auto()
    LPAREN    = auto()  # (
    RPAREN    = auto()  # )
    LBRACE    = auto()  # {
    RBRACE    = auto()  # }
    LBRACKET  = auto()  # [
    RBRACKET  = auto()  # ]
    TIMES     = auto()  # *
    PIPE      = auto()  # |
    COMMA     = auto()  # ,
    SEMICOLON = auto()  # ;
    CONS      = auto()  # ::
    ARROW     = auto()  # ->
    BIARROW   = auto()  # <->
    EQUAL     = auto()  # =
    UNIT      = auto()  # unit  (keyword)
    LET       = auto()  # let
    IN        = auto()  # in
    FIX       = auto()  # fix
    TYPE      = auto()  # type
    INVERT    = auto()  # inv
    REC       = auto()  # rec
    OF        = auto()  # of
    FUN       = auto()  # fun
    MATCH     = auto()  # match
    WITH      = auto()  # with
    NAT       = auto()  # integer literal
    TVAR      = auto()  # type variable 'a
    VAR       = auto()  # variable (lowercase)
    CTOR      = auto()  # constructor (uppercase)


KEYWORDS: dict[str, TK] = {
    "unit":  TK.UNIT,
    "let":   TK.LET,
    "in":    TK.IN,
    "fix":   TK.FIX,
    "type":  TK.TYPE,
    "inv":   TK.INVERT,
    "rec":   TK.REC,
    "of":    TK.OF,
    "fun":   TK.FUN,
    "match": TK.MATCH,
    "with":  TK.WITH,
}


@dataclass(frozen=True)
class Token:
    kind: TK
    value: str | int | None  # str for VAR/TVAR/CTOR, int for NAT, None otherwise
    line: int
    col: int

    def __repr__(self) -> str:
        if self.value is not None:
            return f"Token({self.kind.name}, {self.value!r}, {self.line}:{self.col})"
        return f"Token({self.kind.name}, {self.line}:{self.col})"


# ---------------------------------------------------------------------------
# Lexer errors
# ---------------------------------------------------------------------------

class LexError(Exception):
    def __init__(self, msg: str, line: int, col: int) -> None:
        super().__init__(f"Lex error at {line}:{col}: {msg}")
        self.line = line
        self.col = col


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# We strip comments first (non-nested, as in the OCaml original).
# The OCaml regex: "(*" ([^'*'] | '*' [^')'])* "*)"
# This matches (* ... *) where * not followed by ) is allowed inside.
_COMMENT_RE = re.compile(r'\(\*(?:[^*]|\*(?!\)))*\*\)')


def _strip_comments(src: str) -> str:
    """Remove (* ... *) comments from source (non-nested, like the OCaml version)."""
    return _COMMENT_RE.sub('', src)


# Token specification: ordered list of (pattern, handler).
# We process the source left-to-right; first match wins.
_TOKEN_SPEC: list[tuple[re.Pattern[str], ...]] = []

# Build the master regex with named groups.
_RAW_SPEC: list[tuple[str, str]] = [
    ("WHITESPACE",  r"[ \t\r\n]+"),
    ("BIARROW",     r"<->"),           # must come before ARROW
    ("CONS",        r"::"),            # must come before COLON (not present but safe)
    ("ARROW",       r"->"),
    ("LPAREN",      r"\("),
    ("RPAREN",      r"\)"),
    ("LBRACE",      r"\{"),
    ("RBRACE",      r"\}"),
    ("LBRACKET",    r"\["),
    ("RBRACKET",    r"\]"),
    ("TIMES",       r"\*"),
    ("PIPE",        r"\|"),
    ("COMMA",       r","),
    ("SEMICOLON",   r";"),
    ("EQUAL",       r"="),
    ("NAT",         r"[0-9]+"),
    ("TVAR",        r"'[a-z][a-z0-9_]*"),
    ("CTOR",        r"[A-Z][a-zA-Z0-9]*"),
    ("VAR",         r"[a-z][a-z0-9_']*"),
    ("UNKNOWN",     r"."),
]

_MASTER_RE = re.compile(
    "|".join(f"(?P<{name}>{pat})" for name, pat in _RAW_SPEC)
)


def tokenize(src: str) -> list[Token]:
    """
    Tokenize Morf source code.
    Strips comments first, then produces a token list ending with EOF.
    """
    src = _strip_comments(src)

    tokens: list[Token] = []
    line = 1
    line_start = 0

    for m in _MASTER_RE.finditer(src):
        kind_name = m.lastgroup
        text = m.group()
        col = m.start() - line_start + 1

        # Track line numbers
        newlines = text.count('\n')
        if newlines:
            line += newlines
            line_start = m.start() + text.rfind('\n') + 1

        if kind_name == "WHITESPACE":
            continue
        elif kind_name == "UNKNOWN":
            raise LexError(f"unexpected character {text!r}", line, col)
        elif kind_name == "NAT":
            tokens.append(Token(TK.NAT, int(text), line, col))
        elif kind_name == "TVAR":
            tokens.append(Token(TK.TVAR, text, line, col))
        elif kind_name == "CTOR":
            tokens.append(Token(TK.CTOR, text, line, col))
        elif kind_name == "VAR":
            tk = KEYWORDS.get(text, TK.VAR)
            tokens.append(Token(tk, text if tk == TK.VAR else None, line, col))
        else:
            tk = TK[kind_name]
            tokens.append(Token(tk, None, line, col))

    tokens.append(Token(TK.EOF, None, line, 0))
    return tokens
