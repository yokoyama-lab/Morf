"""
Morf type inference.

Port of inference.ml from the OCaml implementation.
Implements Hindley-Milner type inference with Robinson's unification.
"""

from __future__ import annotations
import contextvars
from dataclasses import dataclass
from typing import Union

from .ast import (
    BaseType, BtUnit, BtProduct, BtNamed, BtVar, BtCtor,
    IsoType, ItBiArrow, ItArrow, ItVar,
    Value, VUnit, VVar, VCtor, VCted, VTuple,
    Expr, EValue, ELet, ELetVal,
    Iso, IPairs, IFix, ILambda, IVar, IApp, IInvert,
    Term, TUnit, TVar, TCtor, TCted, TTuple, TApp, TLet, TLetIso,
    Typedef, VarValue, VarIso, Program,
    Generator, chars_of_int, collect_vars,
    term_of_value,
)
from .ortho import is_orthogonal, convert_pair, for_all_pairs
from .pretty import show_value, show_expr, show_iso_type, show_base_type

_BOLD_PURPLE = "\x1b[1;35m"
_RESET       = "\x1b[0m"

_ISO_TYPE_SINK: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "_ISO_TYPE_SINK", default=None
)


# ---------------------------------------------------------------------------
# Internal type representation `Any`
# ---------------------------------------------------------------------------

class Any:
    """Internal type used during type inference (base types + iso types unified)."""
    __slots__ = ()


@dataclass(frozen=True)
class AUnit(Any):
    pass


@dataclass(frozen=True)
class AProduct(Any):
    items: tuple[Any, ...]


@dataclass(frozen=True)
class ANamed(Any):
    name: str


@dataclass(frozen=True)
class ABiArrow(Any):
    a: Any
    b: Any


@dataclass(frozen=True)
class AArrow(Any):
    a: Any
    b: Any


@dataclass(frozen=True)
class AVar(Any):
    id: int


@dataclass(frozen=True)
class ACtor(Any):
    args: tuple[Any, ...]
    name: str


@dataclass(frozen=True)
class AInverted(Any):
    inner: Any


# ---------------------------------------------------------------------------
# Type errors
# ---------------------------------------------------------------------------

class TypeError(Exception):
    pass


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------

@dataclass
class Subst:
    what: int   # type variable id
    into: Any


def _apply_subst(s: Subst, a: Any) -> Any:
    match a:
        case AVar(id) if id == s.what:
            return s.into
        case AProduct(items):
            return AProduct(tuple(_apply_subst(s, i) for i in items))
        case ABiArrow(x, y):
            return ABiArrow(_apply_subst(s, x), _apply_subst(s, y))
        case AArrow(x, y):
            return AArrow(_apply_subst(s, x), _apply_subst(s, y))
        case ACtor(args, name):
            return ACtor(tuple(_apply_subst(s, arg) for arg in args), name)
        case AInverted(inner):
            return AInverted(_apply_subst(s, inner))
        case _:
            return a


def _apply_substs(substs: list[Subst], a: Any) -> Any:
    for s in substs:
        a = _apply_subst(s, a)
    return a


def _subst_in_equations(s: Subst, eqs: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    return [(_apply_subst(s, a), _apply_subst(s, b)) for a, b in eqs]


def _subst_in_context(s: Subst, ctx: dict[str, Any]) -> dict[str, Any]:
    def apply_elt(elt: Any) -> Any:
        match elt:
            case ('mono', a):
                return ('mono', _apply_subst(s, a))
            case ('scheme', forall, a):
                if s.what in forall:
                    return elt  # quantified var, don't substitute
                return ('scheme', forall, _apply_subst(s, a))
            case ('functional', forall, a, n):
                if s.what in forall:
                    return elt
                return ('functional', forall, _apply_subst(s, a), n)
            case _:
                return elt
    return {k: apply_elt(v) for k, v in ctx.items()}


# ---------------------------------------------------------------------------
# Occurs check
# ---------------------------------------------------------------------------

def _occurs(x: int, a: Any) -> bool:
    match a:
        case AProduct(items):
            return any(_occurs(x, i) for i in items)
        case ACtor(args, _):
            return any(_occurs(x, arg) for arg in args)
        case ABiArrow(u, v) | AArrow(u, v):
            return _occurs(x, u) or _occurs(x, v)
        case AVar(id):
            return id == x
        case AInverted(inner):
            return _occurs(x, inner)
        case _:
            return False


# ---------------------------------------------------------------------------
# Invert iso type
# ---------------------------------------------------------------------------

def _invert_iso_type(a: Any) -> Any:
    """Invert a type: (A <-> B)^-1 = B <-> A."""
    match a:
        case ABiArrow(x, y):
            return ABiArrow(y, x)
        case AArrow(x, y):
            return AArrow(_invert_iso_type(x), _invert_iso_type(y))
        case AInverted(inner):
            return inner
        case AVar(_):
            return AInverted(a)
        case _:
            raise TypeError(f"{_show_any([], a)} is not an iso type")


def _normalize_inv(a: Any) -> Any:
    """Normalize Inverted(X) by actually inverting."""
    match a:
        case ABiArrow(_) | AArrow(_):
            return a
        case AInverted(inner):
            return _invert_iso_type(inner)
        case AVar(_):
            return a
        case _:
            raise TypeError(f"{_show_any([], a)} is not an iso type")


# ---------------------------------------------------------------------------
# Convert Any to BaseType / IsoType (for display)
# ---------------------------------------------------------------------------

def _base_of_any(a: Any) -> BaseType:
    match a:
        case AUnit():
            return BtUnit()
        case AProduct(items):
            return BtProduct(tuple(_base_of_any(i) for i in items))
        case ANamed(name):
            return BtNamed(name)
        case AVar(id):
            return BtVar("'" + chars_of_int(id))
        case ACtor(args, name):
            return BtCtor(tuple(_base_of_any(arg) for arg in args), name)
        case _:
            raise TypeError("base type is expected")


def _iso_of_any(a: Any) -> IsoType:
    match a:
        case ABiArrow(x, y):
            return ItBiArrow(_base_of_any(x), _base_of_any(y))
        case AArrow(x, y):
            return ItArrow(_iso_of_any(x), _iso_of_any(y))
        case AVar(id):
            return ItVar("'" + chars_of_int(id))
        case AInverted(AVar(id)):
            return ItVar("~'" + chars_of_int(id))
        case AInverted(inner):
            inv = _invert_iso_type(inner)
            return _iso_of_any(inv)
        case _:
            raise TypeError("iso type is expected")


# ---------------------------------------------------------------------------
# Type variable ID → display name mapping
# ---------------------------------------------------------------------------

def _tvar_map(types: list[Any]) -> dict[int, int]:
    """
    Build a renaming map: internal id → sequential display id.
    Mirrors inference.ml `tvar_map`.
    """
    mapping: dict[int, int] = {}
    counter = [0]

    def collect(a: Any) -> None:
        match a:
            case AUnit() | ANamed(_):
                pass
            case AProduct(items) | ACtor(items, _):
                for i in items:
                    collect(i)
            case ABiArrow(x, y) | AArrow(x, y):
                collect(x)
                collect(y)
            case AVar(id):
                if id not in mapping:
                    mapping[id] = counter[0]
                    counter[0] += 1
            case AInverted(inner):
                collect(inner)

    for t in types:
        collect(t)
    return mapping


def _show_any(tvar_map: dict[int, int], a: Any) -> str:
    """Display an Any type (remapped with tvar_map for cleaner output)."""
    # Apply renaming
    remapped = a
    for old_id, new_id in tvar_map.items():
        remapped = _apply_subst(Subst(old_id, AVar(new_id)), remapped)

    try:
        bt = _base_of_any(remapped)
        return show_base_type(bt)
    except TypeError:
        pass
    try:
        it = _iso_of_any(remapped)
        return show_iso_type(it)
    except TypeError:
        pass
    return "unreachable (neither base nor iso)"


# ---------------------------------------------------------------------------
# Context: maps variable names to type elements
# Elt = ('mono', Any) | ('scheme', frozenset[int], Any) | ('functional', tuple[int, ...], Any, int)
# ---------------------------------------------------------------------------

Context = dict[str, tuple]  # ('mono', Any) or ('scheme', frozenset, Any) or ('functional', forall, Any, n_items)


def _find(name: str, ctx: Context) -> tuple:
    if name not in ctx:
        raise TypeError(f"{name} was not found in current context")
    return ctx[name]


def _instantiate(gen: Generator, elt: tuple) -> Any:
    """Create a fresh instance of a type scheme."""
    match elt:
        case ('mono', a):
            return a
        case ('scheme', forall, a) | ('functional', forall, a, _):
            substs = [Subst(old_id, AVar(gen.fresh())) for old_id in forall]
            return _apply_substs(substs, a)
        case _:
            raise TypeError(f"unexpected elt: {elt!r}")


def _find_generalizable(a: Any, ctx: Context) -> list[int]:
    """Find type variables in `a` that don't appear free in `ctx`."""
    def free_in_any(a: Any) -> set[int]:
        match a:
            case AUnit() | ANamed(_):
                return set()
            case AProduct(items) | ACtor(items, _):
                result: set[int] = set()
                for i in items:
                    result |= free_in_any(i)
                return result
            case ABiArrow(x, y) | AArrow(x, y):
                return free_in_any(x) | free_in_any(y)
            case AVar(id):
                return {id}
            case AInverted(inner):
                return free_in_any(inner)
            case _:
                return set()

    def free_in_context(ctx: Context) -> set[int]:
        result: set[int] = set()
        for elt in ctx.values():
            match elt:
                case ('mono', a):
                    result |= free_in_any(a)
                case ('scheme', forall, a) | ('functional', forall, a, _):
                    result |= free_in_any(a) - set(forall)
        return result

    ctx_free = free_in_context(ctx)
    a_free = free_in_any(a)
    return list(a_free - ctx_free)


# ---------------------------------------------------------------------------
# Unification
# ---------------------------------------------------------------------------

def unify(equations: list[tuple[Any, Any]]) -> list[Subst]:
    """
    Robinson's unification for Any types.
    Raises TypeError on failure.

    Port of inference.ml `unify`.
    """
    if not equations:
        return []

    (a, b), rest = equations[0], equations[1:]

    if a == b:
        return unify(rest)

    match (a, b):
        case (AInverted(x), AInverted(y)):
            return unify([(x, y)] + rest)
        case (AInverted(i), ABiArrow(x, y)):
            return unify([(i, ABiArrow(y, x))] + rest)
        case (ABiArrow(x, y), AInverted(i)):
            return unify([(i, ABiArrow(y, x))] + rest)
        case (AInverted(i), AArrow(x, y)):
            xi = _invert_iso_type(x)
            yi = _invert_iso_type(y)
            return unify([(i, AArrow(xi, yi))] + rest)
        case (AArrow(x, y), AInverted(i)):
            xi = _invert_iso_type(x)
            yi = _invert_iso_type(y)
            return unify([(i, AArrow(xi, yi))] + rest)
        case (AVar(x), _) if not _occurs(x, b):
            s = Subst(x, b)
            result = unify(_subst_in_equations(s, rest))
            return [s] + result
        case (_, AVar(x)) if not _occurs(x, a):
            s = Subst(x, a)
            result = unify(_subst_in_equations(s, rest))
            return [s] + result
        case (AProduct(l1), AProduct(l2)) if len(l1) == len(l2):
            return unify(list(zip(l1, l2)) + rest)
        case (ABiArrow(a1, b1), ABiArrow(a2, b2)):
            return unify([(a1, a2), (b1, b2)] + rest)
        case (AArrow(a1, b1), AArrow(a2, b2)):
            return unify([(a1, a2), (b1, b2)] + rest)
        case (ACtor(l1, x1), ACtor(l2, x2)) if x1 == x2 and len(l1) == len(l2):
            return unify(list(zip(l1, l2)) + rest)
        case _:
            tmap = _tvar_map([a, b])
            raise TypeError(
                f"unable to unify {_show_any(tmap, a)} and {_show_any(tmap, b)}"
            )


# ---------------------------------------------------------------------------
# Inferred result
# ---------------------------------------------------------------------------

@dataclass
class Inferred:
    a: Any
    equations: list[tuple[Any, Any]]


def finalize(inf: Inferred) -> Any:
    """Run unification and apply substitutions to get final type."""
    substs = unify(inf.equations)
    return _apply_substs(substs, inf.a)


# ---------------------------------------------------------------------------
# Extract named types from a value pattern
# ---------------------------------------------------------------------------

def _extract_named(gen: Generator, v: Value) -> dict[str, Any]:
    """Map each variable in v to a fresh type variable."""
    match v:
        case VVar(name):
            return {name: AVar(gen.fresh())}
        case VUnit() | VCtor(_):
            return {}
        case VCted(_, inner):
            return _extract_named(gen, inner)
        case VTuple(items):
            result: dict[str, Any] = {}
            for i in items:
                result.update(_extract_named(gen, i))
            return result
        case _:
            return {}


# ---------------------------------------------------------------------------
# Linearity check
# ---------------------------------------------------------------------------

def check_pair(v: Value, e: Expr, ctx: Context) -> None:
    """
    Check that each variable introduced by the LHS pattern `v` is used
    exactly once in the RHS expression `e` (linear usage).

    Port of inference.ml `check_pair`.
    Raises TypeError on linearity violation.
    """
    v_renamed, e_renamed = convert_pair(v, e, skip_vars=set(ctx.keys()))
    msg_suffix = (
        f" in branch {show_value(v_renamed)} <->\n  {show_expr(e_renamed)}\n"
        f"source: {show_value(v)} <->\n  {show_expr(e)}"
    )

    # Track each variable: False = available, True = consumed
    consumed: dict[str, bool] = {}

    def add(name: str) -> None:
        consumed[name] = False

    def consume(name: str) -> None:
        if name not in consumed:
            if name in ctx:
                return
            raise TypeError(f"{name} is not in context{msg_suffix}")
        if consumed[name]:
            raise TypeError(f"{name} is already consumed{msg_suffix}")
        consumed[name] = True

    def check_exists_nonconsumed(name: str) -> None:
        if name not in consumed:
            if name in ctx:
                return
            raise TypeError(f"{name} is not in context{msg_suffix}")
        if consumed[name]:
            raise TypeError(f"{name} is already consumed{msg_suffix}")

    def collect_value(val: Value) -> None:
        match val:
            case VVar(name):
                add(name)
            case VCted(_, inner):
                collect_value(inner)
            case VTuple(items):
                for i in items:
                    collect_value(i)
            case _:
                pass

    def check_value(val: Value) -> None:
        match val:
            case VVar(name):
                check_exists_nonconsumed(name)
            case VCted(_, inner):
                check_value(inner)
            case VTuple(items):
                for i in items:
                    check_value(i)
            case _:
                pass

    def check_expr_proper(expr: Expr) -> None:
        match expr:
            case ELet(p1, _, p2, inner):
                for name in collect_vars(p2):
                    consume(name)
                collect_value(p1)
                check_expr_proper(inner)
            case ELetVal(p, v, inner):
                for name in collect_vars(v):
                    consume(name)
                collect_value(p)
                check_expr_proper(inner)
            case EValue(val):
                check_value(val)

    collect_value(v_renamed)
    check_expr_proper(e_renamed)


# ---------------------------------------------------------------------------
# Invert pairs (for checking the inverse direction)
# ---------------------------------------------------------------------------

def _invert_pairs(
    pairs: tuple[tuple[Value, Expr], ...]
) -> list[tuple[Value, Expr]]:
    """Port of inference.ml `invert_pairs`."""
    from .eval import invert as eval_invert
    inv_iso = eval_invert(IPairs(pairs))
    match inv_iso:
        case IPairs(inv_pairs):
            return list(inv_pairs)
        case _:
            return []


# ---------------------------------------------------------------------------
# Core type inference
# ---------------------------------------------------------------------------

def _infer_pair(
    gen: Generator, ctx: Context, v: Value, e: Expr
) -> tuple[Any, Any, list[tuple[Any, Any]]]:
    """
    Infer types for a (value, expr) pair.
    Returns (type_of_v, type_of_e, equations).

    Port of inference.ml `infer_pair`.
    """
    named = _extract_named(gen, v)
    local_ctx = {**ctx, **{k: ('mono', t) for k, t in named.items()}}
    inf_v, _ = _infer_term(term_of_value(v), gen, local_ctx)
    inf_e = _infer_expr(e, gen, local_ctx)
    equations = inf_v.equations + inf_e.equations
    return inf_v.a, inf_e.a, equations


def _generalize(
    gen: Generator,
    ctx: Context,
    p: Value,
    a: Any,
    equations: list[tuple[Any, Any]],
    disabled: bool = False,
) -> tuple[Context, list[tuple[Any, Any]]]:
    """
    Port of inference.ml `generalize`.
    Runs unification, generalizes free variables, extends context with
    pattern variables bound to their inferred types.
    """
    substs = unify(equations)
    u = _apply_substs(substs, a)

    # Check it's a base type
    _base_of_any(u)  # raises TypeError if not

    ctx2 = _apply_substs_to_context(substs, ctx)
    forall = _find_generalizable(u, ctx2)

    named = _extract_named(gen, p)
    ctx3 = {**ctx2, **{k: ('mono', t) for k, t in named.items()}}
    inf_p, _ = _infer_term(term_of_value(p), gen, ctx3)
    es = [(inf_p.a, u)] + inf_p.equations
    substs2 = unify(es)

    new_entries: dict[str, tuple] = {}
    for name, t in named.items():
        t2 = _apply_substs(substs2, t)
        if disabled:
            new_entries[name] = ('mono', t2)
        else:
            new_entries[name] = ('scheme', tuple(forall), t2)

    return ({**ctx2, **new_entries}, equations + es)


def _apply_substs_to_context(substs: list[Subst], ctx: Context) -> Context:
    result = ctx
    for s in substs:
        result = _subst_in_context(s, result)
    return result


def _generalize_iso(
    gen: Generator,
    ctx: Context,
    phi: str,
    a: Any,
    equations: list[tuple[Any, Any]],
) -> tuple[Context, str]:
    """
    Port of inference.ml `generalize_iso`.
    Returns (new_ctx, display_string).
    """
    substs = unify(equations)
    u = _apply_substs(substs, a)

    # Check it's an iso type
    _iso_of_any(u)  # raises if not

    u_show = _normalize_inv(u)
    tmap = _tvar_map([u_show])
    display = _show_any(tmap, u_show)

    name_display = phi[:9] + "..." if len(phi) > 12 else f"{phi:<12}"
    line = f"| {name_display} : {display}"
    print(_BOLD_PURPLE + line + _RESET)
    sink = _ISO_TYPE_SINK.get()
    if sink is not None:
        sink.append(line)

    ctx2 = _apply_substs_to_context(substs, ctx)
    forall = _find_generalizable(u, ctx2)

    # Check if it returns a tuple for functional embedding
    is_functional = False
    n_items = 0
    match u:
        case ABiArrow(_, AProduct(items)) if len(items) > 1:
            is_functional = True
            n_items = len(items)
        case _:
            pass

    if is_functional:
        generalized = ('functional', tuple(forall), u, n_items)
    else:
        generalized = ('scheme', tuple(forall), u)

    return ({**ctx2, phi: generalized}, line)


def _infer_term(t: Term, gen: Generator, ctx: Context) -> tuple[Inferred, Term]:
    """
    Port of inference.ml `infer_term`.
    Returns (Inferred, TransformedTerm).
    """
    match t:
        case TUnit():
            return Inferred(AUnit(), []), TUnit()
        case TVar(name):
            elt = _find(name, ctx)
            return Inferred(_instantiate(gen, elt), []), TVar(name)
        case TCtor(name):
            elt = _find(name, ctx)
            return Inferred(_instantiate(gen, elt), []), TCtor(name)
        case TTuple(items):
            infs_and_terms = [_infer_term(i, gen, ctx) for i in items]
            infs = [it[0] for it in infs_and_terms]
            terms = [it[1] for it in infs_and_terms]
            a = AProduct(tuple(inf.a for inf in infs))
            eqs = [eq for inf in infs for eq in inf.equations]
            return Inferred(a, eqs), TTuple(tuple(terms))
        case TApp(omega, sub):
            # Check if this is a functional isomorphism name
            match omega:
                case IVar(name) if name in ctx and ctx[name][0] == 'functional':
                    # Desugar only if NOT in a context that expects a tuple.
                    # But how do we know the context here? 
                    # For now, let's keep it but maybe nat.morf can use a marker.
                    # No, the user wants it to be automatic.
                    
                    # Actually, if the user explicitly wrote:
                    #   let (m, n) = add (2, 3)
                    # This is a TLet where t1 is TApp.
                    # We can handle this in TLet case by passing a flag.
                    pass 

            inf_omega = _infer_iso(omega, gen, ctx)
            inf_sub, sub_t = _infer_term(sub, gen, ctx)
            fresh = AVar(gen.fresh())
            eqs = (inf_omega.equations + inf_sub.equations
                   + [(inf_omega.a, ABiArrow(inf_sub.a, fresh))])
            
            # If result is a tuple, and we are NOT being matched by a tuple pattern, 
            # maybe we should desugar here? 
            # This is hard because we don't know the parent.
            
            return Inferred(fresh, eqs), TApp(omega, sub_t)

        case TCted(ctor, sub):
            elt = _find(ctor, ctx)
            a_ctor = _instantiate(gen, elt)
            inf_sub, sub_t = _infer_term(sub, gen, ctx)
            fresh = AVar(gen.fresh())
            eqs = inf_sub.equations + [(a_ctor, ABiArrow(inf_sub.a, fresh))]
            return Inferred(fresh, eqs), TCted(ctor, sub_t)

        case TLet(p, t1, t2):
            # If t1 is a TApp of a functional iso, and p is NOT a VTuple,
            # then we desugar t1.
            
            def should_desugar(rhs: Term, pat: Value) -> bool:
                if not isinstance(rhs, TApp): return False
                match rhs.omega:
                    case IVar(name) if name in ctx and ctx[name][0] == 'functional':
                        return not isinstance(pat, VTuple)
                return False

            if should_desugar(t1, p):
                # Desugar t1: let p = (let (r, _) = f u in r) in t2
                # This is a bit complex to do here. 
                # Let's just manually implement the functional application.
                rhs_app = cast(TApp, t1)
                name = cast(IVar, rhs_app.omega).name
                _, _, _, n_items = ctx[name]
                
                inf_omega = _infer_iso(rhs_app.omega, gen, ctx)
                inf_sub, sub_t = _infer_term(rhs_app.sub, gen, ctx)
                
                fresh_res_name = f"_{name}_res"
                p_items = [VVar(fresh_res_name)] + [VVar("_") for _ in range(n_items - 1)]
                
                fresh_app_a = AVar(gen.fresh())
                eqs_app = (inf_omega.equations + inf_sub.equations
                           + [(inf_omega.a, ABiArrow(inf_sub.a, fresh_app_a))])
                
                # let (r, _) = f u in let p = r in t2
                inner_let = TLet(p, TVar(fresh_res_name), t2)
                ctx2, es = _generalize(gen, ctx, VTuple(tuple(p_items)), fresh_app_a, eqs_app)
                inf2, t2_t = _infer_term(inner_let, gen, ctx2)
                
                return (Inferred(inf2.a, eqs_app + es + inf2.equations),
                        TLet(VTuple(tuple(p_items)), TApp(rhs_app.omega, sub_t), t2_t))

            inf1, t1_t = _infer_term(t1, gen, ctx)
            ctx2, es = _generalize(gen, ctx, p, inf1.a, inf1.equations)
            inf2, t2_t = _infer_term(t2, gen, ctx2)
            return Inferred(inf2.a, inf1.equations + es + inf2.equations), TLet(p, t1_t, t2_t)
        case TLetIso(phi, omega, t_body):
            inf_omega = _infer_iso(omega, gen, ctx)
            ctx2, _ = _generalize_iso(gen, ctx, phi, inf_omega.a, inf_omega.equations)
            inf_t, t_t = _infer_term(t_body, gen, ctx2)
            return Inferred(inf_t.a, inf_omega.equations + inf_t.equations), TLetIso(phi, omega, t_t)
        case _:
            raise TypeError(f"_infer_term: unexpected {t!r}")


def _infer_expr(e: Expr, gen: Generator, ctx: Context) -> Inferred:
    """
    Port of inference.ml `infer_expr`.
    """
    match e:
        case EValue(v):
            inf, _ = _infer_term(term_of_value(v), gen, ctx)
            return inf
        case ELet(p1, omega, p2, inner):
            t1 = TApp(omega, term_of_value(p2))
            inf1, _ = _infer_term(t1, gen, ctx)
            ctx2, es = _generalize(gen, ctx, p1, inf1.a, inf1.equations, disabled=True)
            inf2 = _infer_expr(inner, gen, ctx2)
            return Inferred(inf2.a, inf1.equations + es + inf2.equations)
        case ELetVal(p, v, inner):
            inf1, _ = _infer_term(term_of_value(v), gen, ctx)
            ctx2, es = _generalize(gen, ctx, p, inf1.a, inf1.equations, disabled=True)
            inf2 = _infer_expr(inner, gen, ctx2)
            return Inferred(inf2.a, inf1.equations + es + inf2.equations)
        case _:
            raise TypeError(f"_infer_expr: unexpected {e!r}")


def check_orth(pairs: tuple[tuple[Value, Expr], ...], ctx: Context) -> None:
    """
    Check linearity and orthogonality of a pair list.
    Public API: can be called from eval.py or tests directly.
    """
    # Linearity check
    for v, e in pairs:
        check_pair(v, e, ctx)
    # Orthogonality of LHS
    lhs = [v for v, _ in pairs]
    err = for_all_pairs(is_orthogonal, lhs)
    if err:
        raise TypeError(err)
    # Orthogonality of RHS
    from .ast import value_of_expr
    rhs = [value_of_expr(e) for _, e in pairs]
    err = for_all_pairs(is_orthogonal, rhs)
    if err:
        raise TypeError(err)


def _infer_iso(omega: Iso, gen: Generator, ctx: Context) -> Inferred:
    """
    Port of inference.ml `infer_iso`.
    """
    match omega:
        case IPairs(pairs):
            # Check orthogonality both forward and inverse
            check_orth(pairs, ctx)
            inv_pairs = _invert_pairs(pairs)
            check_orth(tuple(inv_pairs), ctx)

            # Infer type for each pair
            pair_types = [_infer_pair(gen, ctx, v, e) for v, e in pairs]
            types_v = [av for av, _, _ in pair_types]
            types_e = [ae for _, ae, _ in pair_types]
            all_eqs = [eq for _, _, eqs in pair_types for eq in eqs]

            if not types_v:
                raise TypeError("empty case expression")

            a = types_v[0]
            b = types_e[0]
            # All LHS types must be equal, all RHS types must be equal
            eq_v = [(types_v[i], types_v[i + 1]) for i in range(len(types_v) - 1)]
            eq_e = [(types_e[i], types_e[i + 1]) for i in range(len(types_e) - 1)]
            return Inferred(ABiArrow(a, b), eq_v + eq_e + all_eqs)

        case IFix(phi, inner):
            fresh = AVar(gen.fresh())
            ctx2 = {**ctx, phi: ('mono', fresh)}
            inf = _infer_iso(inner, gen, ctx2)
            return Inferred(inf.a, [(fresh, inf.a)] + inf.equations)

        case ILambda(psi, inner):
            fresh = AVar(gen.fresh())
            ctx2 = {**ctx, psi: ('mono', fresh)}
            inf = _infer_iso(inner, gen, ctx2)
            return Inferred(AArrow(fresh, inf.a), inf.equations)

        case IVar(name):
            elt = _find(name, ctx)
            return Inferred(_instantiate(gen, elt), [])

        case IApp(omega1, omega2):
            inf1 = _infer_iso(omega1, gen, ctx)
            inf2 = _infer_iso(omega2, gen, ctx)
            fresh = AVar(gen.fresh())
            eqs = inf1.equations + inf2.equations + [(inf1.a, AArrow(inf2.a, fresh))]
            return Inferred(fresh, eqs)

        case IInvert(inner):
            inf = _infer_iso(inner, gen, ctx)
            fresh = AVar(gen.fresh())
            return Inferred(fresh, [(fresh, AInverted(inf.a))] + inf.equations)

        case _:
            raise TypeError(f"_infer_iso: unexpected {omega!r}")


# ---------------------------------------------------------------------------
# Build context from type definitions
# ---------------------------------------------------------------------------

def _arity_map(typedefs: tuple[Typedef, ...]) -> dict[str, int]:
    return {td.type_name: len(td.vars) for td in typedefs}


def _any_of_base(
    bt: BaseType,
    var_map: dict[str, int],
    arity_map: dict[str, int],
) -> Any:
    """Convert a BaseType to Any, checking arity."""
    match bt:
        case BtUnit():
            return AUnit()
        case BtProduct(types):
            return AProduct(tuple(_any_of_base(t, var_map, arity_map) for t in types))
        case BtNamed(name):
            arity = arity_map.get(name)
            if arity is None:
                raise TypeError(f"{name} was not found in current context")
            if arity != 0:
                raise TypeError(
                    f"{name} expects arity of {arity} but provided with 0"
                )
            return ANamed(name)
        case BtVar(name):
            if name not in var_map:
                raise TypeError(f"type variable {name} not in scope")
            return AVar(var_map[name])
        case BtCtor(args, name):
            arity = arity_map.get(name)
            if arity is None:
                raise TypeError(f"{name} was not found in current context")
            if arity != len(args):
                raise TypeError(
                    f"{name} expects arity of {arity} but provided with {len(args)}"
                )
            return ACtor(
                tuple(_any_of_base(a, var_map, arity_map) for a in args),
                name,
            )
        case _:
            raise TypeError(f"_any_of_base: unexpected {bt!r}")


def build_ctx(gen: Generator, typedefs: tuple[Typedef, ...]) -> Context:
    """
    Build the initial type context from type definitions.
    Includes built-in types: nat, bool, 'a list.

    Port of inference.ml `build_ctx`.
    """
    # Built-in types
    builtins = (
        Typedef((), "nat", (VarValue("Z"), VarIso("S", BtNamed("nat")))),
        Typedef((), "bool", (VarValue("False"), VarValue("True"))),
        Typedef(("'a",), "list", (VarValue("Nil"), VarIso("Cons", BtProduct((BtVar("'a"), BtCtor((BtVar("'a"),), "list")))))),
    )
    
    # Only include built-ins not redefined in typedefs
    defined_names = {td.type_name for td in typedefs}
    all_typedefs = tuple(b for b in builtins if b.type_name not in defined_names) + typedefs
    
    arity_map = _arity_map(all_typedefs)
    seen: set[str] = set()
    ctx: Context = {}

    for td in all_typedefs:
        if td.type_name in seen:
            raise TypeError(f"{td.type_name} is defined more than once")
        seen.add(td.type_name)

        # Map type parameter names to fresh ids
        var_map: dict[str, int] = {}
        for var_name in td.vars:
            var_map[var_name] = gen.fresh()

        forall = list(var_map.values())

        # The type itself
        if forall:
            inner: Any = ACtor(tuple(AVar(id) for id in forall), td.type_name)
        else:
            inner = ANamed(td.type_name)

        # Each constructor
        for variant in td.variants:
            match variant:
                case VarValue(name):
                    elt = ('scheme', tuple(forall), inner)
                    ctx[name] = elt
                case VarIso(ctor, arg):
                    a = _any_of_base(arg, var_map, arity_map)
                    iso_type = ABiArrow(a, inner)
                    elt = ('scheme', tuple(forall), iso_type)
                    ctx[ctor] = elt

    return ctx


# ---------------------------------------------------------------------------
# Top-level type inference for a program
# ---------------------------------------------------------------------------

def infer_program(program: Program) -> tuple[Any, list[str], Program]:
    """
    Type-check a full program.
    Returns (result_type, list_of_printed_iso_types, transformed_program).
    Raises TypeError on type errors.
    """
    gen = Generator()
    ctx = build_ctx(gen, program.typedefs)
    lines: list[str] = []
    token = _ISO_TYPE_SINK.set(lines)
    try:
        inf, transformed_term = _infer_term(program.term, gen, ctx)
        final_type = finalize(inf)
        transformed_program = Program(program.typedefs, transformed_term)
    finally:
        _ISO_TYPE_SINK.reset(token)
    return final_type, lines, transformed_program
