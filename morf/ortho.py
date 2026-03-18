"""
Morf orthogonality checker.

Port of ortho.ml from the OCaml implementation.
Checks that patterns on both sides of iso pairs are mutually orthogonal
(non-overlapping), ensuring reversibility.
"""

from __future__ import annotations
from .ast import (
    Value, VUnit, VVar, VCtor, VCted, VTuple,
    Expr, EValue, ELet, ELetVal,
    Generator, chars_of_int, collect_vars,
)
from .pretty import show_value


# ---------------------------------------------------------------------------
# Value-level unification (for orthogonality checking only)
# Different from eval.py's unify_value: returns None on failure (no exception).
# ---------------------------------------------------------------------------

class OrthoSubst:
    """A single substitution: replace variable `what` with value `into`."""
    def __init__(self, what: str, into: Value) -> None:
        self.what = what
        self.into = into

    def __repr__(self) -> str:
        return f"{self.what} = {show_value(self.into)}"


def _subst_value(s: OrthoSubst, v: Value) -> Value:
    """Apply substitution s to value v."""
    match v:
        case VUnit():
            return v
        case VVar(name) if name == s.what:
            return s.into
        case VVar(_):
            return v
        case VCtor(_):
            return v
        case VCted(ctor, inner):
            return VCted(ctor, _subst_value(s, inner))
        case VTuple(items):
            return VTuple(tuple(_subst_value(s, i) for i in items))
        case _:
            return v


def _subst_equations(
    s: OrthoSubst, eqs: list[tuple[Value, Value]]
) -> list[tuple[Value, Value]]:
    return [(_subst_value(s, a), _subst_value(s, b)) for a, b in eqs]


def _occurs(x: str, v: Value) -> bool:
    """Occurs check: does variable x appear in value v?"""
    match v:
        case VUnit() | VCtor(_):
            return False
        case VVar(name):
            return name == x
        case VCted(_, inner):
            return _occurs(x, inner)
        case VTuple(items):
            return any(_occurs(x, i) for i in items)
        case _:
            return False


def _unify_values(
    equations: list[tuple[Value, Value]]
) -> list[OrthoSubst] | None:
    """
    Robinson's unification for values.
    Returns list of substitutions, or None if unification fails.

    Port of ortho.ml `unify`.
    """
    if not equations:
        return []

    (a, b), rest = equations[0], equations[1:]

    if a == b:
        return _unify_values(rest)

    if isinstance(a, VVar) and not _occurs(a.name, b):
        s = OrthoSubst(a.name, b)
        result = _unify_values(_subst_equations(s, rest))
        if result is None:
            return None
        return [s] + result

    if isinstance(b, VVar) and not _occurs(b.name, a):
        s = OrthoSubst(b.name, a)
        result = _unify_values(_subst_equations(s, rest))
        if result is None:
            return None
        return [s] + result

    match (a, b):
        case (VCted(c1, v1), VCted(c2, v2)) if c1 == c2:
            return _unify_values([(v1, v2)] + rest)
        case (VTuple(items1), VTuple(items2)) if len(items1) == len(items2):
            return _unify_values(list(zip(items1, items2)) + rest)
        case _:
            return None  # unification fails


def _reduce_substs(substs: list[OrthoSubst]) -> list[OrthoSubst]:
    """
    Compose substitutions: apply each later substitution to the `into` of
    all earlier ones. Port of ortho.ml `reduce`.
    """
    result: list[OrthoSubst] = []
    for s in substs:
        reduced = OrthoSubst(
            s.what,
            _apply_substs(result, s.into) if result else s.into
        )
        result.append(reduced)
    return result


def _apply_substs(substs: list[OrthoSubst], v: Value) -> Value:
    for s in substs:
        v = _subst_value(s, v)
    return v


# ---------------------------------------------------------------------------
# is_orthogonal
# ---------------------------------------------------------------------------

def is_orthogonal(u: Value, v: Value) -> str | None:
    """
    Check if patterns u and v are orthogonal (non-overlapping).
    Returns None if orthogonal (OK), or an error message string if not.

    Port of ortho.ml `is_orthogonal`.
    """
    gen = Generator()

    def fresh_name() -> Value:
        return VVar(chars_of_int(gen.fresh()))

    def rename_vars(val: Value) -> Value:
        """Replace all variables with fresh names."""
        vars_ = collect_vars(val)
        substs = [OrthoSubst(x, fresh_name()) for x in vars_]
        result = val
        for s in substs:
            result = _subst_value(s, result)
        return result

    u_renamed = rename_vars(u)
    v_renamed = rename_vars(v)

    result = _unify_values([(u_renamed, v_renamed)])
    if result is None:
        return None  # orthogonal (unification fails = patterns can't overlap)

    # Not orthogonal: produce example
    reduced = list(reversed(_reduce_substs(result)))
    reduced.sort(key=lambda s: s.what)
    example = "; ".join(str(s) for s in reduced)
    msg = (
        f"{show_value(u_renamed)} and {show_value(v_renamed)} are not orthogonal\n"
        f"example: {example}\n"
        f"source: {show_value(u)} and {show_value(v)}"
    )
    return msg


# ---------------------------------------------------------------------------
# Linearity checking helpers
# ---------------------------------------------------------------------------

def _subst_in_expr(what: str, into: str, e: Expr) -> Expr:
    """
    Substitute variable name `what` with `into` in expr `e`.
    Port of ortho.ml `subst_in_expr`.
    """
    s = OrthoSubst(what, VVar(into))

    def go_value(v: Value) -> Value:
        return _subst_value(s, v)

    def go(e: Expr) -> Expr:
        match e:
            case EValue(v):
                return EValue(go_value(v))
            case ELet(p1, omega, p2, inner):
                from .ast import contains_value
                new_p2 = go_value(p2)
                if contains_value(what, p1):
                    return ELet(p1, omega, new_p2, inner)
                return ELet(p1, omega, new_p2, go(inner))
            case ELetVal(p, v, inner):
                from .ast import contains_value
                new_v = go_value(v)
                if contains_value(what, p):
                    return ELetVal(p, new_v, inner)
                return ELetVal(p, new_v, go(inner))
            case _:
                return e
    return go(e)


def convert_pair(v: Value, e: Expr, skip_vars: set[str] | None = None) -> tuple[Value, Expr]:
    """
    Rename all variables in (v, e) to fresh names, EXCEPT those in skip_vars.
    Used before linearity checking to avoid name clashes.

    Port of ortho.ml `convert_pair`.
    """
    if skip_vars is None:
        skip_vars = set()
    gen = Generator()

    def fresh_name() -> str:
        return "'" + chars_of_int(gen.fresh())

    # Rename vars in the LHS value
    vars_ = [x for x in collect_vars(v) if x not in skip_vars]
    renames = [(x, fresh_name()) for x in vars_]

    new_v = v
    new_e = e
    for old, new in renames:
        new_v = _subst_value(OrthoSubst(old, VVar(new)), new_v)
        new_e = _subst_in_expr(old, new, new_e)

    def process_expr(e: Expr) -> Expr:
        """Rename bound variables in let-bindings within e to fresh names."""
        match e:
            case EValue(v_out):
                # We need to rename vars in VVar that are NOT in skip_vars
                # but convert_pair original didn't seem to do this?
                # Actually, _subst_in_expr already handled the ones from LHS.
                # But what about vars that are NEITHER in LHS nor in skip_vars?
                # Linearity check will catch them as "not in context".
                return e
            case ELet(p1, omega, p2, inner):
                p1_vars = [x for x in collect_vars(p1) if x not in skip_vars]
                new_names = [(x, fresh_name()) for x in p1_vars]
                new_p1 = p1
                new_inner = inner
                for old, new in new_names:
                    new_p1 = _subst_value(OrthoSubst(old, VVar(new)), new_p1)
                    new_inner = _subst_in_expr(old, new, new_inner)
                return ELet(new_p1, omega, p2, process_expr(new_inner))
            case ELetVal(p, val, inner):
                p_vars = [x for x in collect_vars(p) if x not in skip_vars]
                new_names = [(x, fresh_name()) for x in p_vars]
                new_p = p
                new_inner = inner
                for old, new in new_names:
                    new_p = _subst_value(OrthoSubst(old, VVar(new)), new_p)
                    new_inner = _subst_in_expr(old, new, new_inner)
                return ELetVal(new_p, val, process_expr(new_inner))
            case _:
                return e

    new_e = process_expr(new_e)
    return new_v, new_e


# ---------------------------------------------------------------------------
# for_all_pairs helper
# ---------------------------------------------------------------------------

def for_all_pairs(
    f, items: list
) -> str | None:
    """
    Apply f(a, b) to all pairs a, b where a appears before b.
    Returns first error message, or None if all OK.
    """
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            err = f(a, b)
            if err is not None:
                return err
    return None
