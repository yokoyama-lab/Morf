"""
Morf pretty printer.

Port of the show_* functions from types.ml.
"""

from __future__ import annotations
from .ast import (
    BaseType, BtUnit, BtProduct, BtNamed, BtVar, BtCtor,
    IsoType, ItBiArrow, ItArrow, ItVar,
    Value, VUnit, VVar, VCtor, VCted, VTuple,
    Expr, EValue, ELet, ELetVal,
    Iso, IPairs, IFix, ILambda, IVar, IApp, IInvert,
    Term, TUnit, TVar, TCtor, TCted, TTuple, TApp, TLet, TLetIso,
)


# ---------------------------------------------------------------------------
# Value predicates (for smart display)
# ---------------------------------------------------------------------------

def is_int_value(v: Value) -> bool:
    """True if v is Z / S(S(...Z))."""
    match v:
        case VCted("S", inner):
            return is_int_value(inner)
        case VCtor("Z"):
            return True
        case _:
            return False


def is_list_value(v: Value) -> bool:
    """True if v is a proper Nil-terminated Cons list."""
    match v:
        case VCted("Cons", VTuple((_, tail))):
            return is_list_value(tail)
        case VCtor("Nil"):
            return True
        case _:
            return False


def is_int_term(t: Term) -> bool:
    match t:
        case TCted("S", inner):
            return is_int_term(inner)
        case TCtor("Z"):
            return True
        case _:
            return False


def is_list_term(t: Term) -> bool:
    match t:
        case TCted("Cons", TTuple((_, tail))):
            return is_list_term(tail)
        case TCtor("Nil"):
            return True
        case _:
            return False


# ---------------------------------------------------------------------------
# show_base_type
# ---------------------------------------------------------------------------

def show_base_type(bt: BaseType) -> str:
    match bt:
        case BtUnit():
            return "unit"
        case BtProduct(types):
            parts = []
            for t in types:
                if isinstance(t, BtProduct):
                    parts.append("(" + show_base_type(t) + ")")
                else:
                    parts.append(show_base_type(t))
            return " * ".join(parts)
        case BtNamed(name) | BtVar(name):
            return name
        case BtCtor(args, name):
            if not args:
                return f"unreachable (type constructor with 0 arity)"
            if len(args) == 1:
                a = args[0]
                if isinstance(a, BtProduct):
                    return "(" + show_base_type(a) + ") " + name
                return show_base_type(a) + " " + name
            # Multi-arg: (t1, t2, ...) name
            inner = ", ".join(show_base_type(a) for a in args)
            return "(" + inner + ") " + name
        case _:
            raise ValueError(f"show_base_type: unexpected {bt!r}")


# ---------------------------------------------------------------------------
# show_iso_type
# ---------------------------------------------------------------------------

def show_iso_type(it: IsoType) -> str:
    match it:
        case ItBiArrow(a, b):
            return show_base_type(a) + " <-> " + show_base_type(b)
        case ItArrow(ItVar(_) as t1, ItBiArrow() as t2):
            return show_iso_type(t1) + " -> (" + show_iso_type(t2) + ")"
        case ItArrow(ItVar(_) as t1, t2):
            return show_iso_type(t1) + " -> " + show_iso_type(t2)
        case ItArrow(t1, ItBiArrow() as t2):
            return "(" + show_iso_type(t1) + ") -> (" + show_iso_type(t2) + ")"
        case ItArrow(t1, t2):
            return "(" + show_iso_type(t1) + ") -> " + show_iso_type(t2)
        case ItVar(name):
            return name
        case _:
            raise ValueError(f"show_iso_type: unexpected {it!r}")


# ---------------------------------------------------------------------------
# show_value
# ---------------------------------------------------------------------------

def _int_of_value(v: Value) -> int:
    """Convert nat-encoded value to int."""
    acc = 0
    while True:
        match v:
            case VCted("S", inner):
                acc += 1
                v = inner
            case _:
                return acc


def _list_value_to_parts(v: Value) -> list[str]:
    """Collect list elements as strings."""
    parts: list[str] = []
    while True:
        match v:
            case VCted("Cons", VTuple((head, tail))):
                parts.append(show_value(head))
                v = tail
            case VCtor("Nil"):
                return parts
            case _:
                parts.append("; " + show_value(v))
                return parts


def show_value(v: Value) -> str:
    match v:
        case VUnit():
            return "()"
        case VCtor("Z"):
            return "0"
        case VCtor("Nil"):
            return "[]"
        case VCtor(name) | VVar(name):
            return name
        case _ if is_int_value(v):
            return str(_int_of_value(v))
        case VCted("Cons", VTuple((head, tail))):
            if is_list_value(v):
                parts = _list_value_to_parts(v)
                return "[" + "; ".join(parts) + "]"
            else:
                # Non-proper list: show with ::
                return _show_cons_value(v)
        case VTuple(items):
            return "(" + ", ".join(show_value(i) for i in items) + ")"
        case VCted(ctor, inner) if is_int_value(inner) or is_list_value(inner):
            return ctor + " " + show_value(inner)
        case VCted(ctor, VCted() as inner):
            return ctor + " (" + show_value(inner) + ")"
        case VCted(ctor, inner):
            return ctor + " " + show_value(inner)
        case _:
            raise ValueError(f"show_value: unexpected {v!r}")


def _show_cons_value(v: Value) -> str:
    """Show a non-proper cons chain with :: notation."""
    match v:
        case VCted("Cons", VTuple((head, tail))):
            return show_value(head) + " :: " + _show_cons_value(tail)
        case _:
            return show_value(v)


# ---------------------------------------------------------------------------
# show_iso (needed by show_expr)
# ---------------------------------------------------------------------------

def show_iso(omega: Iso) -> str:
    match omega:
        case IPairs(pairs):
            return _show_pairs(pairs)
        case IFix(phi, inner):
            return "fix " + phi + ". " + show_iso(inner)
        case ILambda(psi, inner):
            return "fun " + psi + " -> " + show_iso(inner)
        case IVar(name):
            return name
        case IApp(omega1, IVar(_) as omega2):
            return show_iso(omega1) + " " + show_iso(omega2)
        case IApp(omega1, omega2):
            return show_iso(omega1) + " {" + show_iso(omega2) + "}"
        case IInvert(IVar(_) as inner):
            return "inv " + show_iso(inner)
        case IInvert(inner):
            return "inv {" + show_iso(inner) + "}"
        case _:
            raise ValueError(f"show_iso: unexpected {omega!r}")


def _show_pairs(pairs: tuple[tuple[Value, Expr], ...]) -> str:
    parts = ["match with"]
    for v, e in pairs:
        parts.append("\n  | " + show_value(v) + " <-> " + show_expr(e))
    return "".join(parts)


# ---------------------------------------------------------------------------
# show_expr
# ---------------------------------------------------------------------------

def _is_complex_iso(omega: Iso) -> bool:
    return isinstance(omega, (IPairs, IFix, ILambda))


def show_expr(e: Expr) -> str:
    match e:
        case EValue(v):
            return show_value(v)
        case ELet(p1, omega, p2, inner):
            lhs = "let " + show_value(p1) + " = "
            if _is_complex_iso(omega):
                iso_str = "{" + show_iso(omega) + "} "
            else:
                iso_str = show_iso(omega) + " "
            if isinstance(p2, VCted) and (is_int_value(p2) or is_list_value(p2)):
                rhs = iso_str + show_value(p2)
            elif isinstance(p2, VCted):
                rhs = iso_str + "(" + show_value(p2) + ")"
            else:
                rhs = iso_str + show_value(p2)
            return lhs + rhs + " in\n  " + show_expr(inner)
        case ELetVal(p, v, inner):
            return "let " + show_value(p) + " = " + show_value(v) + " in\n  " + show_expr(inner)
        case _:
            raise ValueError(f"show_expr: unexpected {e!r}")


# ---------------------------------------------------------------------------
# show_term
# ---------------------------------------------------------------------------

def _int_of_term(t: Term) -> int:
    acc = 0
    while True:
        match t:
            case TCted("S", inner):
                acc += 1
                t = inner
            case _:
                return acc


def _list_term_to_parts(t: Term) -> list[str]:
    parts: list[str] = []
    while True:
        match t:
            case TCted("Cons", TTuple((head, tail))):
                parts.append(show_term(head))
                t = tail
            case TCtor("Nil"):
                return parts
            case _:
                parts.append(show_term(t))
                return parts


def show_term(t: Term) -> str:
    match t:
        case TUnit():
            return "()"
        case TCtor("Z"):
            return "0"
        case TCtor("Nil"):
            return "[]"
        case TVar(name) | TCtor(name):
            return name
        case _ if is_int_term(t):
            return str(_int_of_term(t))
        case TCted("Cons", TTuple((head, tail))):
            if is_list_term(t):
                parts = _list_term_to_parts(t)
                return "[" + "; ".join(parts) + "]"
            else:
                return _show_cons_term(t)
        case TTuple(items):
            return "(" + ", ".join(show_term(i) for i in items) + ")"
        case TCted(ctor, inner) if is_int_term(inner) or is_list_term(inner):
            return ctor + " " + show_term(inner)
        case TCted(ctor, (TApp() | TLet() | TLetIso()) as inner):
            return ctor + " (" + show_term(inner) + ")"
        case TCted(ctor, TCted() as inner):
            return ctor + " (" + show_term(inner) + ")"
        case TCted(ctor, inner):
            return ctor + " " + show_term(inner)
        case TApp(omega, inner) if _is_complex_iso(omega):
            if is_int_term(inner) or is_list_term(inner):
                return "{" + show_iso(omega) + "} " + show_term(inner)
            elif isinstance(inner, (TCted, TApp, TLet, TLetIso)):
                return "{" + show_iso(omega) + "} (" + show_term(inner) + ")"
            else:
                return "{" + show_iso(omega) + "} " + show_term(inner)
        case TApp(omega, inner):
            if is_int_term(inner) or is_list_term(inner):
                return show_iso(omega) + " " + show_term(inner)
            elif isinstance(inner, (TCted, TApp, TLet, TLetIso)):
                return show_iso(omega) + " (" + show_term(inner) + ")"
            else:
                return show_iso(omega) + " " + show_term(inner)
        case TLet(p, t1, t2):
            return "let " + show_value(p) + " = " + show_term(t1) + "\nin\n\n" + show_term(t2)
        case TLetIso(phi, omega, inner):
            return "let " + phi + " = " + show_iso(omega) + "\nin\n\n" + show_term(inner)
        case _:
            raise ValueError(f"show_term: unexpected {t!r}")


def _show_cons_term(t: Term) -> str:
    match t:
        case TCted("Cons", TTuple((head, tail))):
            return show_term(head) + " :: " + _show_cons_term(tail)
        case _:
            return show_term(t)


# ---------------------------------------------------------------------------
# show_value for lhs of match (used in error messages)
# ---------------------------------------------------------------------------

def show_pairs_lhs(v: Value, pairs: tuple[tuple[Value, Expr], ...]) -> str:
    init = "match with"
    for pv, _ in pairs:
        init += "\n  | " + show_value(pv) + " <-> ..."
    return init
