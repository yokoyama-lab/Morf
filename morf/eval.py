"""
Morf evaluator.

Port of eval.ml from the OCaml implementation.
"""

from __future__ import annotations
from .ast import (
    Value, VUnit, VVar, VCtor, VCted, VTuple,
    Expr, EValue, ELet, ELetVal,
    Iso, IPairs, IFix, ILambda, IVar, IApp, IInvert,
    Term, TUnit, TVar, TCtor, TCted, TTuple, TApp, TLet, TLetIso,
    contains_value, contains_pairs,
    term_of_value,
)
from .pretty import show_value, show_term, show_pairs_lhs


class EvalError(Exception):
    """Runtime evaluation error."""


# ---------------------------------------------------------------------------
# matches
# ---------------------------------------------------------------------------

def matches(pattern: Value, value: Value) -> bool:
    """
    Check whether `value` matches `pattern`.
    Handles repeated variables: the first occurrence binds, subsequent
    occurrences must equal the bound value (memoized in `memo`).

    Port of eval.ml `matches`.
    """
    memo: dict[str, Value | None] = {}

    # Build the memo map from pattern variables.
    # None = seen once (unbound), Value = bound to specific value (for repeats).
    def _init(p: Value) -> None:
        match p:
            case VUnit() | VCtor(_):
                pass
            case VVar(name):
                if name in memo:
                    memo[name] = None  # mark as repeated (needs memoization)
                else:
                    memo[name] = None  # first occurrence
            case VCted(_, inner):
                _init(inner)
            case VTuple(items):
                for i in items:
                    _init(i)

    # Count occurrences
    counts: dict[str, int] = {}
    def _count(p: Value) -> None:
        match p:
            case VVar(name):
                counts[name] = counts.get(name, 0) + 1
            case VCted(_, inner):
                _count(inner)
            case VTuple(items):
                for i in items:
                    _count(i)
            case _:
                pass

    _count(pattern)
    # Initialize memo for repeated vars with a sentinel
    _UNSET = object()
    repeated_memo: dict[str, object] = {
        name: _UNSET for name, cnt in counts.items() if cnt > 1
    }

    def _matches(p: Value, v: Value) -> bool:
        match p:
            case VUnit():
                return isinstance(v, VUnit)
            case VVar(name):
                if name in repeated_memo:
                    bound = repeated_memo[name]
                    if bound is _UNSET:
                        repeated_memo[name] = v
                        return True
                    else:
                        return bound == v
                return True  # non-repeated var always matches
            case VCtor(pname):
                return isinstance(v, VCtor) and v.name == pname
            case VCted(pctor, pinner):
                return (isinstance(v, VCted) and v.ctor == pctor
                        and _matches(pinner, v.value))
            case VTuple(pitems):
                if not isinstance(v, VTuple) or len(v.items) != len(pitems):
                    return False
                return all(_matches(pi, vi) for pi, vi in zip(pitems, v.items))
            case _:
                return False

    return _matches(pattern, value)


# ---------------------------------------------------------------------------
# invert
# ---------------------------------------------------------------------------

def invert(omega: Iso) -> Iso:
    """
    Invert an isomorphism by swapping directions.

    Port of eval.ml `invert`.
    For Pairs: each (v, e) becomes (value_of_expr(e), let_chain_reversed -> v).
    For Fix/Lambda/App: recurse.
    For Invert: unwrap (double inversion cancels).
    """
    match omega:
        case IPairs(pairs):
            def invert_pair(ve: tuple[Value, Expr]) -> tuple[Value, Expr]:
                v, e = ve
                # invert_expr builds reversed let-chain
                result_v, result_e = _invert_expr(e, EValue(v))
                return result_v, result_e
            return IPairs(tuple(invert_pair(pair) for pair in pairs))
        case IFix(phi, inner):
            return IFix(phi, invert(inner))
        case ILambda(psi, inner):
            return ILambda(psi, invert(inner))
        case IVar(_):
            return omega  # variable: can't invert statically
        case IApp(omega1, omega2):
            return IApp(invert(omega1), invert(omega2))
        case IInvert(inner):
            return inner  # double inversion cancels
        case _:
            raise EvalError(f"invert: unexpected {omega!r}")


def _invert_expr(e: Expr, acc: Expr) -> tuple[Value, Expr]:
    """
    Traverse expr `e`, building the inverse let-chain in `acc`.
    Returns (final_value, inverted_expr).

    Port of the local `invert_expr` in eval.ml.
    """
    match e:
        case EValue(v):
            return v, acc
        case ELet(p1, omega, p2, inner):
            # Forward: let p1 = omega p2 in ...
            # Inverse: let p2 = inv(omega) p1 in ...  (swapped, then recurse)
            new_acc = ELet(p2, invert(omega), p1, acc)
            return _invert_expr(inner, new_acc)
        case ELetVal(p, v, inner):
            # Forward: let p = v in ...
            # Inverse: let v = p in ...
            new_acc = ELetVal(v, p, acc)
            return _invert_expr(inner, new_acc)
        case _:
            raise EvalError(f"_invert_expr: unexpected {e!r}")


# ---------------------------------------------------------------------------
# subst_* (variable substitution in Term/Iso)
# ---------------------------------------------------------------------------

def subst(from_: str, into: Term, what: Term) -> Term:
    """
    Substitute variable `from_` with term `into` in term `what`.

    Port of eval.ml `subst`.
    """
    def go(t: Term) -> Term:
        match t:
            case TVar(name) if name == from_:
                return into
            case TCted(ctor, inner):
                return TCted(ctor, go(inner))
            case TTuple(items):
                return TTuple(tuple(go(i) for i in items))
            case TApp(omega, inner):
                return TApp(omega, go(inner))
            case TLet(p, t1, t2):
                if contains_value(from_, p):
                    # from_ is bound by pattern p → don't substitute into t2
                    return TLet(p, go(t1), t2)
                return TLet(p, go(t1), go(t2))
            case TLetIso(phi, omega, inner):
                if phi == from_:
                    return t
                return TLetIso(phi, omega, go(inner))
            case _:
                return t
    return go(what)


def subst_iso(from_: str, into: Iso, what: Iso) -> Iso:
    """
    Substitute iso variable `from_` with iso `into` in iso `what`.

    Port of eval.ml `subst_iso`.
    """
    def go(omega: Iso) -> Iso:
        match omega:
            case IPairs(pairs) if not contains_pairs(from_, pairs):
                new_pairs = tuple(
                    (v, _subst_iso_in_expr(from_, into, e)) for v, e in pairs
                )
                return IPairs(new_pairs)
            case IFix(phi, inner) if phi != from_:
                return IFix(phi, go(inner))
            case ILambda(psi, inner) if psi != from_:
                return ILambda(psi, go(inner))
            case IVar(name) if name == from_:
                return into
            case IApp(omega1, omega2):
                return IApp(go(omega1), go(omega2))
            case IInvert(inner):
                return IInvert(go(inner))
            case _:
                return omega
    return go(what)


def _subst_iso_in_expr(from_: str, into: Iso, e: Expr) -> Expr:
    """Substitute iso variable in an expression."""
    match e:
        case EValue(_):
            return e
        case ELet(p1, omega, p2, inner):
            new_omega = subst_iso(from_, into, omega)
            if contains_value(from_, p1):
                return ELet(p1, new_omega, p2, inner)
            return ELet(p1, new_omega, p2, _subst_iso_in_expr(from_, into, inner))
        case ELetVal(p, v, inner):
            if contains_value(from_, p):
                return ELetVal(p, v, inner)
            return ELetVal(p, v, _subst_iso_in_expr(from_, into, inner))
        case _:
            raise EvalError(f"_subst_iso_in_expr: unexpected {e!r}")


def _subst_iso_in_term(from_: str, into: Iso, t: Term) -> Term:
    """Substitute iso variable in a term."""
    def go(t: Term) -> Term:
        match t:
            case TTuple(items):
                return TTuple(tuple(go(i) for i in items))
            case TCted(ctor, inner):
                return TCted(ctor, go(inner))
            case TApp(omega, inner):
                return TApp(subst_iso(from_, into, omega), go(inner))
            case TLet(p, t1, t2):
                if contains_value(from_, p):
                    return TLet(p, go(t1), t2)
                return TLet(p, go(t1), go(t2))
            case TLetIso(phi, omega, inner) if phi == from_:
                return TLetIso(phi, subst_iso(from_, into, omega), inner)
            case TLetIso(phi, omega, inner):
                return TLetIso(phi, subst_iso(from_, into, omega), go(inner))
            case _:
                return t
    return go(t)


# ---------------------------------------------------------------------------
# value_of_term
# ---------------------------------------------------------------------------

def value_of_term(t: Term) -> Value:
    """
    Convert a fully-evaluated term to a Value.
    Raises EvalError if the term is not fully reduced.

    Port of eval.ml `value_of_term`.
    """
    match t:
        case TUnit():
            return VUnit()
        case TVar(name):
            return VVar(name)
        case TCtor(name):
            return VCtor(name)
        case TTuple(items):
            return VTuple(tuple(value_of_term(i) for i in items))
        case TCted(ctor, inner):
            return VCted(ctor, value_of_term(inner))
        case _:
            raise EvalError(f"unreachable (unreduced term: {show_term(t)})")


# ---------------------------------------------------------------------------
# unify_value
# ---------------------------------------------------------------------------

def unify_value(u: Value, v: Value) -> list[tuple[str, Value]]:
    """
    Unify pattern `u` against value `v`, returning variable bindings.
    Raises EvalError if unification fails.

    Port of eval.ml `unify_value`.
    """
    match (u, v):
        case (VUnit(), VUnit()):
            return []
        case (VVar(name), _):
            return [(name, v)]
        case (VCtor(name1), VCtor(name2)) if name1 == name2:
            return []
        case (VCted(c1, v1), VCted(c2, v2)) if c1 == c2:
            return unify_value(v1, v2)
        case (VTuple(items1), VTuple(items2)) if len(items1) == len(items2):
            result: list[tuple[str, Value]] = []
            for a, b in zip(items1, items2):
                result.extend(unify_value(a, b))
            return result
        case _:
            raise EvalError(
                f"unable to unify {show_value(u)} and {show_value(v)}"
            )


# ---------------------------------------------------------------------------
# match_pair
# ---------------------------------------------------------------------------

def match_pair(
    pairs: tuple[tuple[Value, Expr], ...],
    v: Value
) -> tuple[Value, Expr] | None:
    """Find the first pair whose LHS pattern matches v."""
    for pat, expr in pairs:
        if matches(pat, v):
            return pat, expr
    return None


# ---------------------------------------------------------------------------
# eval / eval_iso
# ---------------------------------------------------------------------------

def eval(t: Term) -> Term:
    """
    Evaluate a term to normal form.
    Raises EvalError on runtime errors (pattern match failure, etc.).

    Port of eval.ml `eval`.
    """
    match t:
        case TUnit() | TVar(_) | TCtor(_):
            return t
        case TTuple(items):
            return TTuple(tuple(eval(i) for i in items))
        case TCted(ctor, inner):
            return TCted(ctor, eval(inner))
        case TApp(omega, sub):
            omega_ev = eval_iso(omega)
            v = value_of_term(eval(sub))
            match omega_ev:
                case IPairs(pairs):
                    # Linearity + orthogonality check (lazy import avoids circular dep)
                    from .inference import check_orth, TypeError as _IsoTypeError
                    try:
                        # We pass {} as ctx because eval currently uses substitution 
                        # instead of environments for terms.
                        check_orth(pairs, {}) 
                    except _IsoTypeError as exc:
                        raise EvalError(str(exc)) from exc
                    result = match_pair(pairs, v)
                    if result is None:
                        raise EvalError(
                            "out of domain: " + show_pairs_lhs(v, pairs)
                        )
                    pat, e = result
                    bindings = unify_value(pat, v)
                    # Apply bindings: substitute each variable in the expr term
                    t_result = _term_of_expr(e)
                    for name, val in bindings:
                        t_result = subst(name, term_of_value(val), t_result)
                    return eval(t_result)
                case _:
                    # Iso didn't reduce to Pairs; return as-is
                    return TApp(omega_ev, term_of_value(v))
        case TLet(p, t1, t2):
            v1 = value_of_term(eval(t1))
            if not matches(p, v1):
                raise EvalError(
                    f"unable to unify {show_value(p)} and {show_value(v1)}"
                )
            bindings = unify_value(p, v1)
            result = t2
            for name, val in bindings:
                result = subst(name, term_of_value(val), result)
            return eval(result)
        case TLetIso(phi, omega, inner):
            omega_ev = eval_iso(omega)
            t_result = _subst_iso_in_term(phi, omega_ev, inner)
            return eval(t_result)
        case _:
            return t


def eval_iso(omega: Iso) -> Iso:
    """
    Evaluate/reduce an isomorphism.

    Port of eval.ml `eval_iso`.
    """
    match omega:
        case IFix(phi, inner):
            # Unfold: substitute the fix itself into the body
            unfolded = subst_iso(phi, omega, inner)
            return eval_iso(unfolded)
        case IApp(omega1, omega2):
            ev1 = eval_iso(omega1)
            match ev1:
                case ILambda(psi, body):
                    applied = subst_iso(psi, omega2, body)
                    return eval_iso(applied)
                case _:
                    return omega  # can't reduce further
        case IInvert(inner):
            return eval_iso(invert(eval_iso(inner)))
        case _:
            return omega


# ---------------------------------------------------------------------------
# Helper: convert Expr back to Term for evaluation
# ---------------------------------------------------------------------------

def _term_of_expr(e: Expr) -> Term:
    """
    Convert an Expr to a Term so it can be eval'd.
    Mirrors types.ml `term_of_expr`.
    """
    match e:
        case EValue(v):
            return term_of_value(v)
        case ELet(p1, omega, p2, inner):
            return TLet(
                p1,
                TApp(omega, term_of_value(p2)),
                _term_of_expr(inner),
            )
        case ELetVal(p, v, inner):
            return TLet(p, term_of_value(v), _term_of_expr(inner))
        case _:
            raise EvalError(f"_term_of_expr: unexpected {e!r}")
