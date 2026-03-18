"""
Morf AST node definitions.

Port of types.ml from the OCaml implementation.
Each OCaml variant type becomes a Python ABC with @dataclass subclasses.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Generator (fresh variable names)
# ---------------------------------------------------------------------------

class Generator:
    """Generates fresh integer IDs for type variables."""

    def __init__(self) -> None:
        self._i = 0

    def fresh(self) -> int:
        n = self._i
        self._i += 1
        return n


def chars_of_int(n: int) -> str:
    """Convert non-negative integer to a letter sequence: 0→'a', 25→'z', 26→'aa', ..."""
    if n < 0:
        return "wtf"
    if n < 26:
        return chr(ord('a') + n)
    return chars_of_int((n // 26) - 1) + chars_of_int(n % 26)


# ---------------------------------------------------------------------------
# BaseType
# ---------------------------------------------------------------------------

class BaseType:
    """Abstract base for base types."""
    __slots__ = ()


@dataclass(frozen=True)
class BtUnit(BaseType):
    """unit"""

    def __repr__(self) -> str:
        return "BtUnit()"


@dataclass(frozen=True)
class BtProduct(BaseType):
    """t1 * t2 * ..."""
    types: tuple[BaseType, ...]

    def __repr__(self) -> str:
        return f"BtProduct({self.types!r})"


@dataclass(frozen=True)
class BtNamed(BaseType):
    """A named type (e.g. 'nat', 'bool')."""
    name: str


@dataclass(frozen=True)
class BtVar(BaseType):
    """A type variable (e.g. 'a)."""
    name: str


@dataclass(frozen=True)
class BtCtor(BaseType):
    """A parameterized type constructor (e.g. 'a list, ('l, 'r) either)."""
    args: tuple[BaseType, ...]
    name: str


# ---------------------------------------------------------------------------
# IsoType
# ---------------------------------------------------------------------------

class IsoType:
    """Abstract base for isomorphism types."""
    __slots__ = ()


@dataclass(frozen=True)
class ItBiArrow(IsoType):
    """A <-> B"""
    a: BaseType
    b: BaseType


@dataclass(frozen=True)
class ItArrow(IsoType):
    """(A <-> B) -> (C <-> D)  or  tau1 -> tau2"""
    t1: IsoType
    t2: IsoType


@dataclass(frozen=True)
class ItVar(IsoType):
    """A type variable for isomorphism types."""
    name: str


# ---------------------------------------------------------------------------
# Value  (patterns and values — used on both sides of <->)
# ---------------------------------------------------------------------------

class Value:
    """Abstract base for values/patterns."""
    __slots__ = ()


@dataclass(frozen=True)
class VUnit(Value):
    """()"""


@dataclass(frozen=True)
class VVar(Value):
    """A variable x."""
    name: str


@dataclass(frozen=True)
class VCtor(Value):
    """A nullary constructor (e.g. Z, Nil, True)."""
    name: str


@dataclass(frozen=True)
class VCted(Value):
    """A constructor applied to a value (e.g. S n, Cons (x, xs))."""
    ctor: str
    value: Value


@dataclass(frozen=True)
class VTuple(Value):
    """A tuple (v1, v2, ...)."""
    items: tuple[Value, ...]


# ---------------------------------------------------------------------------
# Expr  (right-hand side of <->; may contain let-bindings)
# ---------------------------------------------------------------------------

class Expr:
    """Abstract base for expressions (RHS of iso pairs)."""
    __slots__ = ()


@dataclass(frozen=True)
class EValue(Expr):
    """A bare value (no let-bindings)."""
    value: Value


@dataclass(frozen=True)
class ELet(Expr):
    """let p1 = omega p2 in e"""
    p1: Value       # output pattern
    iso: Iso
    p2: Value       # input value
    expr: Expr


@dataclass(frozen=True)
class ELetVal(Expr):
    """let p = v in e  (pure value binding)."""
    p: Value
    v: Value
    expr: Expr


# ---------------------------------------------------------------------------
# Iso  (isomorphism expressions)
# ---------------------------------------------------------------------------

class Iso:
    """Abstract base for isomorphisms."""
    __slots__ = ()


@dataclass(frozen=True)
class IPairs(Iso):
    """case | v1 <-> e1 | v2 <-> e2 ..."""
    pairs: tuple[tuple[Value, Expr], ...]


@dataclass(frozen=True)
class IFix(Iso):
    """fix phi. omega  (recursion)"""
    phi: str
    omega: Iso


@dataclass(frozen=True)
class ILambda(Iso):
    """fun psi -> omega  (higher-order parameter)"""
    psi: str
    omega: Iso


@dataclass(frozen=True)
class IVar(Iso):
    """An iso variable."""
    name: str


@dataclass(frozen=True)
class IApp(Iso):
    """omega1 omega2  (iso application)"""
    omega1: Iso
    omega2: Iso


@dataclass(frozen=True)
class IInvert(Iso):
    """inv omega  (inversion)"""
    omega: Iso


# ---------------------------------------------------------------------------
# Term  (top-level computations / expressions in term position)
# ---------------------------------------------------------------------------

class Term:
    """Abstract base for terms."""
    __slots__ = ()


@dataclass(frozen=True)
class TUnit(Term):
    """()"""


@dataclass(frozen=True)
class TVar(Term):
    """A variable."""
    name: str


@dataclass(frozen=True)
class TCtor(Term):
    """A nullary constructor."""
    name: str


@dataclass(frozen=True)
class TCted(Term):
    """Constructor application."""
    ctor: str
    term: Term


@dataclass(frozen=True)
class TTuple(Term):
    """Tuple."""
    items: tuple[Term, ...]


@dataclass(frozen=True)
class TApp(Term):
    """omega t  — apply an isomorphism to a term."""
    omega: Iso
    term: Term


@dataclass(frozen=True)
class TLet(Term):
    """let p = t1 in t2"""
    p: Value
    t1: Term
    t2: Term


@dataclass(frozen=True)
class TLetIso(Term):
    """iso phi = omega in t"""
    phi: str
    omega: Iso
    term: Term


# ---------------------------------------------------------------------------
# ExprIntermediate  (before expand_expr desugaring)
# ---------------------------------------------------------------------------

class ExprIntermediate:
    """Intermediate expr representation, before let-normal-form expansion."""
    __slots__ = ()


@dataclass(frozen=True)
class EIValue(ExprIntermediate):
    term: Term


@dataclass(frozen=True)
class EILet(ExprIntermediate):
    """let p1 = p2 in e"""
    p1: Value
    p2: Term
    expr: ExprIntermediate


# ---------------------------------------------------------------------------
# Type definitions and Program
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VarValue:
    """A nullary constructor variant: | Ctor"""
    name: str


@dataclass(frozen=True)
class VarIso:
    """A constructor variant with payload: | Ctor of base_type"""
    ctor: str
    arg: BaseType


Variant = VarValue | VarIso


@dataclass(frozen=True)
class Typedef:
    """type ('a, 'b) name = | ..."""
    vars: tuple[str, ...]   # type parameter names, e.g. ('a', 'b')
    type_name: str
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class Program:
    typedefs: tuple[Typedef, ...]
    term: Term


# ---------------------------------------------------------------------------
# Helper functions (ported from types.ml)
# ---------------------------------------------------------------------------

def term_of_value(v: Value) -> Term:
    """Convert a value (pattern) to a term."""
    match v:
        case VUnit():
            return TUnit()
        case VVar(name):
            return TVar(name)
        case VCtor(name):
            return TCtor(name)
        case VCted(ctor, val):
            return TCted(ctor, term_of_value(val))
        case VTuple(items):
            return TTuple(tuple(term_of_value(i) for i in items))
        case _:
            raise ValueError(f"term_of_value: unexpected {v!r}")


def value_of_expr(e: Expr) -> Value:
    """Get the final value produced by an expression (its 'result pattern')."""
    match e:
        case EValue(v):
            return v
        case ELet(expr=inner) | ELetVal(expr=inner):
            return value_of_expr(inner)
        case _:
            raise ValueError(f"value_of_expr: unexpected {e!r}")


def contains_value(what: str, v: Value) -> bool:
    """Check whether variable `what` appears in value `v`."""
    match v:
        case VUnit() | VCtor(_):
            return False
        case VVar(name):
            return name == what
        case VCted(_, val):
            return contains_value(what, val)
        case VTuple(items):
            return any(contains_value(what, i) for i in items)
        case _:
            raise ValueError(f"contains_value: unexpected {v!r}")


def contains_pairs(what: str, pairs: tuple[tuple[Value, Expr], ...]) -> bool:
    """Check whether variable `what` appears in any LHS of pairs."""
    return any(contains_value(what, v) for v, _ in pairs)


def collect_vars(v: Value) -> list[str]:
    """Collect all variable names from a value pattern (unique, sorted)."""
    def _collect(v: Value) -> list[str]:
        match v:
            case VUnit() | VCtor(_):
                return []
            case VVar(name):
                return [name]
            case VCted(_, val):
                return _collect(val)
            case VTuple(items):
                result: list[str] = []
                for i in items:
                    result.extend(_collect(i))
                return result
            case _:
                raise ValueError(f"collect_vars: unexpected {v!r}")

    seen: set[str] = set()
    result: list[str] = []
    for name in _collect(v):
        if name not in seen:
            seen.add(name)
            result.append(name)
    return sorted(result)


def nat_of_int(n: int) -> Value:
    """Build the nat encoding of a non-negative integer."""
    if n < 1:
        return VCtor("Z")
    return VCted("S", nat_of_int(n - 1))


def lambdas_of_params(params: list[str], omega: Iso) -> Iso:
    """Wrap iso in nested ILambda for each parameter name."""
    for psi in reversed(params):
        omega = ILambda(psi, omega)
    return omega


# ---------------------------------------------------------------------------
# expand / expand_expr  (let-normal form desugaring, from types.ml)
# ---------------------------------------------------------------------------

def expand(gen: Generator, t: Term) -> tuple[list[tuple[Value, Iso, Value]], Value]:
    """
    Expand a term into let-normal form.
    Returns (bindings, value) where bindings = [(output_var, iso, input_val), ...].
    Each TApp becomes a fresh variable bound via let.
    Raises ValueError for nested Let/LetIso (not supported).
    """
    match t:
        case TUnit():
            return [], VUnit()
        case TVar(name):
            return [], VVar(name)
        case TCtor(name):
            return [], VCtor(name)
        case TTuple(items):
            all_bindings: list[tuple[Value, Iso, Value]] = []
            vals: list[Value] = []
            for item in items:
                bindings, v = expand(gen, item)
                all_bindings.extend(bindings)
                vals.append(v)
            return all_bindings, VTuple(tuple(vals))
        case TCted(ctor, sub):
            bindings, v = expand(gen, sub)
            return bindings, VCted(ctor, v)
        case TApp(omega, sub):
            bindings, v = expand(gen, sub)
            fresh_name = VVar("_" + chars_of_int(gen.fresh()))
            return bindings + [(fresh_name, omega, v)], fresh_name
        case TLet():
            raise ValueError("nested let is not supported (yet)")
        case TLetIso():
            raise ValueError("nested iso binding is not supported (yet)")
        case _:
            raise ValueError(f"expand: unexpected {t!r}")


def expand_expr(gen: Generator, e: ExprIntermediate) -> Expr:
    """
    Desugar ExprIntermediate into Expr (let-normal form).
    Matches expand_expr in types.ml.
    """
    match e:
        case EIValue(t):
            bindings, v = expand(gen, t)
            result: Expr = EValue(v)
            # fold right: innermost binding first, wrapping outward
            for p1, omega, p2 in reversed(bindings):
                result = ELet(p1, omega, p2, result)
            return result
        case EILet(p1, p2, inner):
            bindings, v = expand(gen, p2)
            inner_expr = expand_expr(gen, inner)
            result_let: Expr = ELetVal(p1, v, inner_expr)
            for bp1, omega, bp2 in reversed(bindings):
                result_let = ELet(bp1, omega, bp2, result_let)
            return result_let
        case _:
            raise ValueError(f"expand_expr: unexpected {e!r}")
