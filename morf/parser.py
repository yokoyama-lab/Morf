"""
Morf recursive-descent parser.

Port of parser.mly from the OCaml implementation.
Consumes a token list and produces a Program AST.
"""

from __future__ import annotations
from .ast import (
    # BaseType
    BaseType, BtUnit, BtProduct, BtNamed, BtVar, BtCtor,
    # IsoType (not directly constructed in parser)
    # Value
    Value, VUnit, VVar, VCtor, VCted, VTuple,
    # Expr
    Expr, EValue, ELet, ELetVal,
    # ExprIntermediate
    ExprIntermediate, EIValue, EILet,
    # Iso
    Iso, IPairs, IFix, ILambda, IVar, IApp, IInvert,
    # Term
    Term, TUnit, TVar, TCtor, TCted, TTuple, TApp, TLet, TLetIso,
    # Helpers
    Typedef, VarValue, VarIso, Variant, Program,
    nat_of_int, lambdas_of_params, expand_expr, Generator,
)
from .lexer import Token, TK, tokenize


# ---------------------------------------------------------------------------
# Parse errors
# ---------------------------------------------------------------------------

class ParseError(Exception):
    def __init__(self, msg: str, tok: Token) -> None:
        super().__init__(f"Parse error at {tok.line}:{tok.col}: {msg} (got {tok.kind.name}"
                         + (f" {tok.value!r}" if tok.value is not None else "") + ")")
        self.tok = tok


# ---------------------------------------------------------------------------
# Token stream helper
# ---------------------------------------------------------------------------

class TokenStream:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def peek(self) -> Token:
        return self._tokens[self._pos]

    def peek2(self) -> Token:
        """Look two ahead (for lookahead)."""
        if self._pos + 1 < len(self._tokens):
            return self._tokens[self._pos + 1]
        return self._tokens[-1]  # EOF

    def advance(self) -> Token:
        tok = self._tokens[self._pos]
        if self._pos < len(self._tokens) - 1:
            self._pos += 1
        return tok

    def expect(self, kind: TK) -> Token:
        tok = self.peek()
        if tok.kind != kind:
            raise ParseError(f"expected {kind.name}", tok)
        return self.advance()

    def consume(self, kind: TK) -> bool:
        """Consume token if it matches kind, return True if consumed."""
        if self.peek().kind == kind:
            self.advance()
            return True
        return False

    def at(self, kind: TK) -> bool:
        return self.peek().kind == kind

    def at_any(self, *kinds: TK) -> bool:
        return self.peek().kind in kinds


# ---------------------------------------------------------------------------
# wtf(separator, X): parses at least 2 X separated by separator
# In practice used for: (A, B, ...) type lists and A * B * ... products
# ---------------------------------------------------------------------------

def parse_wtf(stream: TokenStream, sep: TK, parse_item):
    """Parse 2+ items separated by `sep` (like Menhir's wtf rule)."""
    first = parse_item(stream)
    stream.expect(sep)
    second = parse_item(stream)
    items = [first, second]
    while stream.consume(sep):
        items.append(parse_item(stream))
    return items


# ---------------------------------------------------------------------------
# base_type
# ---------------------------------------------------------------------------

def parse_base_type_grouped(stream: TokenStream) -> BaseType:
    """
    base_type_grouped:
      | LPAREN base_type RPAREN
      | UNIT
      | VAR (named)
      | TVAR (type var)
      | base_type_grouped VAR  (1-arg type constructor)
      | LPAREN wtf(COMMA, base_type) RPAREN VAR  (multi-arg type constructor)
    """
    tok = stream.peek()

    if tok.kind == TK.UNIT:
        stream.advance()
        bt: BaseType = BtUnit()
    elif tok.kind == TK.LPAREN:
        stream.advance()
        # Check if this is (t1, t2, ...) VAR  or  (t)
        # We need to try parsing a base_type, then look at what follows.
        inner = parse_base_type(stream)
        if stream.consume(TK.COMMA):
            # Multi-arg: (t1, t2, ...) VAR
            items = [inner]
            items.append(parse_base_type(stream))
            while stream.consume(TK.COMMA):
                items.append(parse_base_type(stream))
            stream.expect(TK.RPAREN)
            name_tok = stream.expect(TK.VAR)
            bt = BtCtor(tuple(items), str(name_tok.value))
        else:
            stream.expect(TK.RPAREN)
            bt = inner
    elif tok.kind == TK.VAR:
        stream.advance()
        bt = BtNamed(str(tok.value))
    elif tok.kind == TK.TVAR:
        stream.advance()
        bt = BtVar(str(tok.value))
    else:
        raise ParseError("expected base type", tok)

    # Postfix: base_type_grouped VAR  →  BtCtor([bt], name)
    while stream.at(TK.VAR):
        name_tok = stream.advance()
        bt = BtCtor((bt,), str(name_tok.value))

    return bt


def parse_base_type(stream: TokenStream) -> BaseType:
    """
    base_type:
      | wtf(TIMES, base_type_grouped)   (product)
      | base_type_grouped
    """
    first = parse_base_type_grouped(stream)
    if stream.at(TK.TIMES):
        items = [first]
        while stream.consume(TK.TIMES):
            items.append(parse_base_type_grouped(stream))
        return BtProduct(tuple(items))
    return first


# ---------------------------------------------------------------------------
# typedef
# ---------------------------------------------------------------------------

def parse_variant(stream: TokenStream) -> Variant:
    """
    variant:
      | CTOR OF base_type
      | CTOR
    """
    tok = stream.expect(TK.CTOR)
    name = str(tok.value)
    if stream.consume(TK.OF):
        arg = parse_base_type(stream)
        return VarIso(name, arg)
    return VarValue(name)


def parse_typedef(stream: TokenStream) -> Typedef:
    """
    typedef:
      | TYPE VAR EQUAL PIPE? separated_nonempty_list(PIPE, variant)
      | TYPE TVAR VAR EQUAL ...
      | TYPE LPAREN wtf(COMMA, TVAR) RPAREN VAR EQUAL ...
    """
    stream.expect(TK.TYPE)

    # Parse type parameters
    vars_: list[str] = []
    if stream.at(TK.TVAR):
        tok = stream.advance()
        vars_ = [str(tok.value)]
    elif stream.at(TK.LPAREN):
        stream.advance()
        tok = stream.expect(TK.TVAR)
        vars_ = [str(tok.value)]
        while stream.consume(TK.COMMA):
            tok = stream.expect(TK.TVAR)
            vars_.append(str(tok.value))
        stream.expect(TK.RPAREN)

    # Type name
    name_tok = stream.expect(TK.VAR)
    type_name = str(name_tok.value)

    stream.expect(TK.EQUAL)
    stream.consume(TK.PIPE)  # optional leading |

    variants: list[Variant] = [parse_variant(stream)]
    while stream.consume(TK.PIPE):
        variants.append(parse_variant(stream))

    return Typedef(tuple(vars_), type_name, tuple(variants))


# ---------------------------------------------------------------------------
# value (patterns)
# ---------------------------------------------------------------------------

def parse_value_grouped(stream: TokenStream) -> Value:
    """
    value_grouped:
      | LPAREN value RPAREN
      | LPAREN RPAREN              → VUnit
      | LPAREN wtf(COMMA, value) RPAREN → VTuple
      | VAR                        → VVar
      | CTOR                       → VCtor
      | NAT                        → nat_of_int
      | LBRACKET RBRACKET          → VCtor "Nil"
      | LBRACKET separated_nonempty_list(SEMICOLON, value) RBRACKET
    """
    tok = stream.peek()

    if tok.kind == TK.LPAREN:
        stream.advance()
        if stream.consume(TK.RPAREN):
            return VUnit()
        inner = parse_value(stream)
        if stream.consume(TK.COMMA):
            items = [inner]
            items.append(parse_value(stream))
            while stream.consume(TK.COMMA):
                items.append(parse_value(stream))
            stream.expect(TK.RPAREN)
            return VTuple(tuple(items))
        stream.expect(TK.RPAREN)
        return inner
    elif tok.kind == TK.VAR:
        stream.advance()
        return VVar(str(tok.value))
    elif tok.kind == TK.CTOR:
        stream.advance()
        return VCtor(str(tok.value))
    elif tok.kind == TK.NAT:
        stream.advance()
        return nat_of_int(int(tok.value))
    elif tok.kind == TK.LBRACKET:
        stream.advance()
        if stream.consume(TK.RBRACKET):
            return VCtor("Nil")
        # [v1; v2; ...]
        items: list[Value] = [parse_value(stream)]
        while stream.consume(TK.SEMICOLON):
            items.append(parse_value(stream))
        stream.expect(TK.RBRACKET)
        # fold_right: Cons(v1, Cons(v2, Nil))
        result: Value = VCtor("Nil")
        for v in reversed(items):
            result = VCted("Cons", VTuple((v, result)))
        return result
    elif tok.kind == TK.UNIT:
        stream.advance()
        return VUnit()
    else:
        raise ParseError("expected value pattern", tok)


def parse_value_almost(stream: TokenStream) -> Value:
    """
    value_almost:
      | value_grouped
      | CTOR value_grouped   → VCted
    """
    tok = stream.peek()
    if tok.kind == TK.CTOR:
        # Check if constructor is followed by a value_grouped starter
        next_tok = stream.peek2()
        if next_tok.kind in (TK.LPAREN, TK.VAR, TK.CTOR, TK.NAT,
                              TK.LBRACKET, TK.UNIT):
            stream.advance()
            arg = parse_value_grouped(stream)
            return VCted(str(tok.value), arg)
    return parse_value_grouped(stream)


def parse_value(stream: TokenStream) -> Value:
    """
    value:
      | value_almost
      | value_almost CONS value   → VCted("Cons", VTuple([v1, v2]))
    """
    v = parse_value_almost(stream)
    if stream.consume(TK.CONS):
        v2 = parse_value(stream)
        return VCted("Cons", VTuple((v, v2)))
    return v


# ---------------------------------------------------------------------------
# iso
# ---------------------------------------------------------------------------

def parse_biarrowed(stream: TokenStream) -> tuple[Value, Expr]:
    """value BIARROW expr"""
    v = parse_value(stream)
    stream.expect(TK.BIARROW)
    e = parse_expr(stream)
    return v, e


def parse_iso_grouped(stream: TokenStream) -> Iso:
    """
    iso_grouped:
      | LBRACE iso RBRACE
      | VAR
    """
    tok = stream.peek()
    if tok.kind == TK.LBRACE:
        stream.advance()
        omega = parse_iso(stream)
        stream.expect(TK.RBRACE)
        return omega
    elif tok.kind == TK.VAR:
        stream.advance()
        return IVar(str(tok.value))
    else:
        raise ParseError("expected iso (variable or {iso})", tok)


# Tokens that can start either an iso_grouped or a term_grouped.
# Used for 2-token lookahead in term_ctx mode.
_TERM_OR_ISO_GROUPED_STARTERS = frozenset({
    TK.VAR, TK.LBRACE,                          # iso_grouped starters
    TK.LPAREN, TK.CTOR, TK.NAT, TK.LBRACKET, TK.UNIT,  # term_grouped starters
})


def parse_iso_almost(stream: TokenStream, term_ctx: bool = False) -> Iso:
    """
    iso_almost:
      | iso_grouped
      | INVERT iso_grouped            → IInvert
      | iso_almost iso_grouped        → IApp (left-assoc)

    When term_ctx=True (called from parse_term_almost), the loop uses 2-token
    lookahead for VAR tokens: only consume a VAR as iso_grouped when the token
    *after* it is also a valid starter (meaning something is still left for
    term_grouped).  LBRACE is always safe to consume.
    """
    tok = stream.peek()
    if tok.kind == TK.INVERT:
        stream.advance()
        omega = parse_iso_grouped(stream)
        omega = IInvert(omega)
    else:
        omega = parse_iso_grouped(stream)

    # Left-associative application loop
    while True:
        if stream.at(TK.LBRACE):
            # LBRACE is unambiguously an iso_grouped starter; always consume.
            arg = parse_iso_grouped(stream)
            omega = IApp(omega, arg)
        elif stream.at(TK.VAR):
            if term_ctx:
                # Only consume this VAR if what comes after is also a valid
                # starter — guaranteeing there's still something for term_grouped.
                if stream.peek2().kind not in _TERM_OR_ISO_GROUPED_STARTERS:
                    break
            arg = parse_iso_grouped(stream)
            omega = IApp(omega, arg)
        else:
            break

    return omega


def parse_iso(stream: TokenStream) -> Iso:
    """
    iso:
      | iso_almost
      | MATCH WITH PIPE? separated_nonempty_list(PIPE, biarrowed)
      | FIX VAR ARROW iso
      | FUN VAR+ ARROW iso
    """
    tok = stream.peek()

    if tok.kind == TK.MATCH:
        # Check if it's "match with" or "match term with"
        if stream.peek2().kind == TK.WITH:
            stream.advance() # MATCH
            stream.advance() # WITH
            stream.consume(TK.PIPE)  # optional leading |
            pairs: list[tuple[Value, Expr]] = [parse_biarrowed(stream)]
            while stream.consume(TK.PIPE):
                pairs.append(parse_biarrowed(stream))
            return IPairs(tuple(pairs))
        else:
            # Not an iso, but an application? No, parse_iso should only handle isos.
            return parse_iso_almost(stream)

    elif tok.kind == TK.FIX:
        stream.advance()
        phi_tok = stream.expect(TK.VAR)
        stream.expect(TK.ARROW)
        omega = parse_iso(stream)
        return IFix(str(phi_tok.value), omega)

    elif tok.kind == TK.FUN:
        stream.advance()
        params: list[str] = []
        while stream.at(TK.VAR):
            params.append(str(stream.advance().value))
        if not params:
            raise ParseError("fun requires at least one parameter", stream.peek())
        stream.expect(TK.ARROW)
        omega = parse_iso(stream)
        return lambdas_of_params(params, omega)

    else:
        return parse_iso_almost(stream)


# ---------------------------------------------------------------------------
# expr_intermediate and expr
# ---------------------------------------------------------------------------

def parse_expr_intermediate(stream: TokenStream) -> ExprIntermediate:
    """
    expr_intermediate:
      | term_nonlet                                           → EIValue
      | LET value EQUAL term_nonlet IN expr_intermediate     → EILet
      | LET value EQUAL MATCH term_nonlet WITH PIPE? biarrowed+ IN expr_i
    """
    if stream.at(TK.LET):
        stream.advance()
        p1 = parse_value(stream)
        stream.expect(TK.EQUAL)

        if stream.at(TK.MATCH):
            # LET p = MATCH t WITH pairs IN e
            stream.advance()
            t = parse_term_nonlet(stream)
            stream.expect(TK.WITH)
            stream.consume(TK.PIPE)
            pairs: list[tuple[Value, Expr]] = [parse_biarrowed(stream)]
            while stream.consume(TK.PIPE):
                pairs.append(parse_biarrowed(stream))
            p2: Term = TApp(IPairs(tuple(pairs)), t)
        else:
            p2 = parse_term_nonlet(stream)

        stream.expect(TK.IN)
        inner = parse_expr_intermediate(stream)
        return EILet(p1, p2, inner)

    else:
        t = parse_term_nonlet(stream)
        return EIValue(t)


def parse_expr(stream: TokenStream) -> Expr:
    """
    expr:
      | expr_intermediate   (then call expand_expr)
    """
    gen = Generator()
    ei = parse_expr_intermediate(stream)
    return expand_expr(gen, ei)


# ---------------------------------------------------------------------------
# term
# ---------------------------------------------------------------------------

def _is_term_grouped_starter(tok: Token) -> bool:
    return tok.kind in (TK.LPAREN, TK.VAR, TK.CTOR, TK.NAT, TK.LBRACKET, TK.UNIT)


def parse_term_grouped(stream: TokenStream) -> Term:
    """
    term_grouped:
      | LPAREN term RPAREN
      | LPAREN RPAREN               → TUnit
      | LPAREN wtf(COMMA, term) RPAREN → TTuple
      | VAR                         → TVar
      | CTOR                        → TCtor
      | NAT                         → nat term
      | LBRACKET RBRACKET           → TCtor "Nil"
      | LBRACKET separated_nonempty_list(SEMICOLON, term) RBRACKET
    """
    tok = stream.peek()

    if tok.kind == TK.LPAREN:
        stream.advance()
        if stream.consume(TK.RPAREN):
            return TUnit()
        inner = parse_term(stream)
        if stream.consume(TK.COMMA):
            items = [inner]
            items.append(parse_term(stream))
            while stream.consume(TK.COMMA):
                items.append(parse_term(stream))
            stream.expect(TK.RPAREN)
            return TTuple(tuple(items))
        stream.expect(TK.RPAREN)
        return inner
    elif tok.kind == TK.VAR:
        stream.advance()
        return TVar(str(tok.value))
    elif tok.kind == TK.CTOR:
        stream.advance()
        return TCtor(str(tok.value))
    elif tok.kind == TK.NAT:
        stream.advance()
        return term_of_value_fn(nat_of_int(int(tok.value)))
    elif tok.kind == TK.LBRACKET:
        stream.advance()
        if stream.consume(TK.RBRACKET):
            return TCtor("Nil")
        items_t: list[Term] = [parse_term(stream)]
        while stream.consume(TK.SEMICOLON):
            items_t.append(parse_term(stream))
        stream.expect(TK.RBRACKET)
        result_t: Term = TCtor("Nil")
        for t in reversed(items_t):
            result_t = TCted("Cons", TTuple((t, result_t)))
        return result_t
    elif tok.kind == TK.UNIT:
        stream.advance()
        return TUnit()
    else:
        raise ParseError("expected term", tok)


def term_of_value_fn(v) -> Term:
    """Local import shim for term_of_value."""
    from .ast import term_of_value
    return term_of_value(v)


def parse_term_almost(stream: TokenStream) -> Term:
    """
    term_almost:
      | term_grouped
      | CTOR term_grouped              → TCted
      | iso_almost term_grouped        → TApp

    Tricky: iso_almost starts with VAR or LBRACE or INVERT.
    term_grouped also starts with VAR.
    When we see a VAR, we must decide: is it an iso application or just a var?

    Rule: if VAR is followed by something that starts a term_grouped (LPAREN, CTOR,
    NAT, LBRACKET, UNIT, or another VAR when that VAR is not at the 'top'), we treat
    the VAR as an iso being applied. Otherwise it's just a TVar.

    More precisely, in the OCaml grammar:
      term_almost: iso_almost term_grouped   (if the iso_almost is just a VAR,
                                              it looks like: VAR term_grouped)
    But term_grouped also accepts VAR as TVar.

    The key: if the next token is VAR and the token *after* that starts a
    term_grouped, we interpret the first VAR as an iso.
    """
    tok = stream.peek()

    if tok.kind == TK.CTOR:
        # CTOR term_grouped  or  just CTOR (nullary)
        next_tok = stream.peek2()
        if _is_term_grouped_starter(next_tok) and next_tok.line == tok.line:
            stream.advance()
            arg = parse_term_grouped(stream)
            return TCted(str(tok.value), arg)
        else:
            return parse_term_grouped(stream)

    elif tok.kind == TK.INVERT:
        # iso_almost (which starts with INVERT) term_grouped
        omega = parse_iso_almost(stream, term_ctx=True)
        arg = parse_term_grouped(stream)
        return TApp(omega, arg)

    elif tok.kind == TK.LBRACE:
        # iso_almost (which starts with LBRACE) term_grouped
        omega = parse_iso_almost(stream, term_ctx=True)
        arg = parse_term_grouped(stream)
        return TApp(omega, arg)

    elif tok.kind == TK.VAR:
        # Could be:
        #   - TVar(x)               when not followed by a term_grouped starter
        #   - TApp(IVar(x), arg)    when followed by a term_grouped starter
        next_tok = stream.peek2()
        if _is_term_grouped_starter(next_tok) and next_tok.line == tok.line:
            # iso_almost term_grouped (with 2-token lookahead in the loop)
            omega = parse_iso_almost(stream, term_ctx=True)
            arg = parse_term_grouped(stream)
            return TApp(omega, arg)
        else:
            return parse_term_grouped(stream)

    else:
        return parse_term_grouped(stream)


def parse_term_nonlet(stream: TokenStream) -> Term:
    """
    term_nonlet:
      | term_almost
      | term_almost CONS term_nonlet   → TCted("Cons", ...)
      | MATCH term_nonlet WITH PIPE? biarrowed+
    """
    tok = stream.peek()

    if tok.kind == TK.MATCH:
        stream.advance()
        t = parse_term_nonlet(stream)
        stream.expect(TK.WITH)
        stream.consume(TK.PIPE)
        pairs: list[tuple[Value, Expr]] = [parse_biarrowed(stream)]
        while stream.consume(TK.PIPE):
            pairs.append(parse_biarrowed(stream))
        return TApp(IPairs(tuple(pairs)), t)

    t = parse_term_almost(stream)
    if stream.consume(TK.CONS):
        t2 = parse_term_nonlet(stream)
        return TCted("Cons", TTuple((t, t2)))
    return t


def parse_term(stream: TokenStream) -> Term:
    """
    term:
      | term_almost
      | term_almost CONS term                       → TCted("Cons", ...)
      | LET [REC] p = rhs [IN term]                 → TLet/TLetIso
      | ISO phi params* EQUAL iso IN term           → TLetIso
      | MATCH term WITH PIPE? biarrowed+            → TApp(IPairs, term)
    """
    tok = stream.peek()

    if tok.kind == TK.LET:
        stream.advance()
        is_rec = stream.consume(TK.REC)
        p = parse_value(stream)
        
        # In Morf, ISO name is usually just a VAR. If it's a TLetIso, p must be a VVar.
        params: list[str] = []
        if isinstance(p, VVar):
            while stream.at(TK.VAR):
                params.append(str(stream.advance().value))
                
        stream.expect(TK.EQUAL)
        
        # Detection logic
        # If RHS starts with MATCH WITH, FUN, FIX, INVERT, LBRACE, it's an Iso.
        # Also if there are parameters, it's an Iso.
        next_tok = stream.peek()
        is_iso = False
        if params:
            is_iso = True
        elif next_tok.kind in (TK.FUN, TK.FIX, TK.INVERT, TK.LBRACE):
            is_iso = True
        elif next_tok.kind == TK.MATCH and stream.peek2().kind == TK.WITH:
            is_iso = True
        
        if is_iso:
            omega = parse_iso(stream)
            omega = lambdas_of_params(params, omega)
            if is_rec:
                if not isinstance(p, VVar):
                    raise ParseError("recursive let must bind a single name", tok)
                omega = IFix(p.name, omega)
            
            if stream.consume(TK.IN):
                t = parse_term(stream)
                if not isinstance(p, VVar):
                    raise ParseError("iso let must bind a single name", tok)
                return TLetIso(p.name, omega, t)
            else:
                # Top-level or trailing definition
                # We return a special marker that parse_program will handle
                return ('def_iso', p, omega) # type: ignore
        else:
            t1 = parse_term(stream)
            if stream.consume(TK.IN):
                t2 = parse_term(stream)
                return TLet(p, t1, t2)
            else:
                return ('def_term', p, t1) # type: ignore

    elif tok.kind == TK.MATCH:
        # This is match t with ...
        stream.advance()
        if stream.at(TK.WITH):
            # Bare match with in term position? 
            # Usually should be match t with. 
            # But maybe it's just an IPairs being used as a term (not allowed yet).
            raise ParseError("expected term before 'with'", stream.peek())
        t = parse_term(stream)
        stream.expect(TK.WITH)
        stream.consume(TK.PIPE)
        pairs: list[tuple[Value, Expr]] = [parse_biarrowed(stream)]
        while stream.consume(TK.PIPE):
            pairs.append(parse_biarrowed(stream))
        return TApp(IPairs(tuple(pairs)), t)

    else:
        t = parse_term_almost(stream)
        if stream.consume(TK.CONS):
            t2 = parse_term(stream)
            return TCted("Cons", TTuple((t, t2)))
        return t


# ---------------------------------------------------------------------------
# Top-level: program
# ---------------------------------------------------------------------------

def parse_program(stream: TokenStream) -> Program:
    """
    program:
      | typedef* definition* term EOF
    """
    typedefs: list[Typedef] = []
    while stream.at(TK.TYPE):
        typedefs.append(parse_typedef(stream))

    # Support multiple top-level let definitions
    definitions: list[tuple] = []
    
    # We need to try parsing a term, and if it's a definition, we keep going.
    # But parse_term already consumes LET.
    
    while True:
        # We need a way to peek if it's a definition without consuming.
        # But parse_term is recursive.
        # Let's just call parse_term and check its return value.
        res = parse_term(stream)
        if isinstance(res, tuple) and res[0] in ('def_iso', 'def_term'):
            definitions.append(res)
        else:
            final_term = res
            break

    stream.expect(TK.EOF)

    # Fold definitions back into the final term
    term = final_term
    for def_type, p, rhs in reversed(definitions):
        if def_type == 'def_iso':
            if not isinstance(p, VVar):
                raise ValueError("Top-level iso let must bind a single name")
            term = TLetIso(p.name, rhs, term)
        else:
            term = TLet(p, rhs, term)

    return Program(tuple(typedefs), term)


def parse(src: str) -> Program:
    """Parse Morf source code into a Program AST."""
    tokens = tokenize(src)
    stream = TokenStream(tokens)
    return parse_program(stream)
