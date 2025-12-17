

from __future__ import annotations

from typing import List

from egglog import *
from egglog.bindings import EggSmolError

from src.slide.log import DEBUG


class RelExpr(Expr):
    def __init__(self, rel: StringLike) -> None: ...

    @classmethod
    def var(cls, name: StringLike) -> RelExpr: ...

    def __or__(self, other: RelExpr) -> RelExpr: ...

    def __and__(self, other: RelExpr) -> RelExpr: ...

    def __floordiv__(self, other: RelExpr) -> RelExpr: ...

    def sequence(self, other: RelExpr) -> RelExpr: ...


M = RelExpr('M')
R = RelExpr('R')
W = RelExpr('W')
RCsc = RelExpr('RCsc')
AQ = RelExpr('AQ')
RL = RelExpr('RL')
AMO = RelExpr('AMO')
AQRL = RelExpr('AQRL')
X = RelExpr('X')
rmw = RelExpr('rmw')
data = RelExpr('data')
ctrl = RelExpr('ctrl')
addr = RelExpr('addr')
fence = RelExpr('fence')
rfi = RelExpr('rfi')
rfe = RelExpr('rfe')
rf = RelExpr('rf')
co = RelExpr('co')
coi = RelExpr('coi')
coe = RelExpr('coe')
fr = RelExpr('fr')
rsw = RelExpr('rsw')
po = RelExpr('po')
loc = RelExpr('loc')
po_loc = RelExpr('po-loc')
empty = RelExpr('empty')

RELATIONS = {
    'M': M,
    'R': R,
    'W': W,
    'RCsc': RCsc,
    'AQ': AQ,
    'RL': RL,
    'AQRL': AQRL,
    'AMO': AMO,
    'X': X,
    'rmw': rmw,
    'data': data,
    'ctrl': ctrl,
    'addr': addr,
    'fence': fence,
    'rfi': rfi,
    'rfe': rfe,
    'rf': rf,
    'co': co,
    'coi': coi,
    'coe': coe,
    'rsw': rsw,
    'po': po,
    'loc': loc,
    'po-loc':po_loc,
    'empty': empty
}
UNARY_RELATIONS = [W, R, M, AQ, RL, AMO, X, RCsc]
BINARY_RELATIONS = [rmw, data, addr, ctrl, fence, rfi, rfe, rf, co, coi, coe, rsw, po, loc, po_loc]
ALL_RELATIONS = list(RELATIONS.values())
_egraph = EGraph()


@_egraph.register
def _rel_expr_rule(a: RelExpr, b: RelExpr, c: RelExpr, d: RelExpr):
    """
    Simplification Rules for RelExpr.

    Warning: do not use associative law: (AB)C = A(BC),
    because it is redundant and makes egraph saturation non-terminate.
    # yield rewrite((a & b) & c).to(a & (b & c))
    # yield rewrite((a | b) | c).to(a | (b | c))
    """

    # commutative law: AB = BA
    yield rewrite(a | b).to(b | a)
    yield rewrite(a & b).to(b & a)

    # distribution law: (AC)(BC) = (AB)C
    yield rewrite((a | c) & (b | c)).to((a & b) | c)
    yield rewrite((a & c) | (b & c)).to((a | b) & c)

    # associative law: A(BC) = (AB)C
    # yield rewrite(a | (b | c)).to(a | b | c)
    # yield rewrite((a | b) | c).to(a | (b | c))
    # yield rewrite(a & (b & c)).to(a & b & c)
    # yield rewrite((a & b) & c).to(a & (b & c))

    # empty law: A&O = O; A|O = A
    yield rewrite(a & empty).to(empty)
    yield rewrite(a | empty).to(a)

    # inclusion
    for x in UNARY_RELATIONS:
        yield rewrite(M & x).to(x)
        yield rewrite(M | x).to(M)
        yield rewrite(x // M).to(empty)

    for x in UNARY_RELATIONS:
        yield rewrite(x.sequence(x)).to(x)
    for x in [rfi, rfe]:
        yield rewrite(x | rf).to(rf)
        yield rewrite(x & rf).to(x)
        yield rewrite(x // rf).to(empty)
    for x in [coi, coe]:
        yield rewrite(x | co).to(co)
        yield rewrite(x & co).to(co)
        yield rewrite(x // co).to(empty)

    for x in ALL_RELATIONS:
        yield rewrite(x // x).to(empty)
        yield rewrite(x | x).to(x)
        yield rewrite(x & x).to(x)

    # mutually exclusive
    for x in UNARY_RELATIONS:
        for y in BINARY_RELATIONS:
            yield rewrite(x & y).to(empty)
            yield rewrite(x // y).to(x)
            yield rewrite(y // x).to(y)
    yield rewrite(W & R).to(empty)
    for x in [data, ctrl, addr, rmw]:
        yield rewrite(W.sequence(x)).to(empty)
    for x in [data, ctrl, coi, coe, co, rmw]:
        yield rewrite(x.sequence(R)).to(empty)

    # unary-binary concat: this is a grammar constraint
    for x in UNARY_RELATIONS:
        for y in UNARY_RELATIONS:
            yield rewrite(x.sequence(y)).to(x & y)

    # unary-binary and or: this is a grammar constraint
    for x in UNARY_RELATIONS:
        for y in BINARY_RELATIONS:
            yield rewrite(x | y).to(empty)
            yield rewrite(x & y).to(empty)

    # M
    for x in BINARY_RELATIONS:
        yield rewrite(M.sequence(x)).to(x)
        yield rewrite(x.sequence(M)).to(M.sequence(x))

    for x in BINARY_RELATIONS:
        for y in BINARY_RELATIONS:
            yield rewrite(x.sequence(y)).to(empty)

    for x in [rmw, data, addr, ctrl, fence, rsw]:
        yield rewrite(po & x).to(x)
        yield rewrite(po | x).to(po)
        yield rewrite(x // po).to(empty)


def check_eq(e1: RelExpr, e2: RelExpr, display: bool = False) -> bool:
    """
    Check if two expressions are semantically equivalent.
    """
    _egraph.push()
    expr1 = _egraph.let("expr1", e1)
    expr2 = _egraph.let("expr2", e2)
    _egraph.saturate()

    if display:
        _egraph.display()

    is_eq = True

    try:
        _egraph.check(eq(expr1).to(expr2))
    except EggSmolError:
        is_eq = False
    finally:
        _egraph.pop()
        return is_eq


def cost(a: RelExpr) -> int:
    return str(a).count('RelExpr')


def can_be_simplified(a: RelExpr, limit: int = 100) -> bool:
    _egraph.push()
    b = _egraph.simplify(a, limit)
    DEBUG(f'{str(a)} is simplified to {str(b)}')
    _egraph.pop()
    return cost(a) > cost(b)


def simplify(a: RelExpr, limit: int = 100) -> RelExpr:
    return _egraph.simplify(a, limit)


def extract_list(exprs: List[RelExpr]) -> List[RelExpr]:
    """
    Extract unique expressions in the simplest form.
    :param exprs: a list of expressions
    :return: a list of unique expressions

    Example:
    extract_list([M|R, R|M]) = [M]
    """
    _egraph.push()
    for i, expr in enumerate(exprs):
        _egraph.let(f"expr{i}", expr)
    _egraph.saturate()
    unique_expr = []
    visited = []
    for e in [_egraph.extract(expr) for expr in exprs]:
        e_str = str(e)
        if e_str not in visited:
            unique_expr.append(e)
            visited.append(e_str)
    _egraph.pop()
    return unique_expr


def _extract_list_index_once(exprs: List[RelExpr]) -> List[int]:
    _egraph.push()
    _egraph.let("empty", empty)
    for i, expr in enumerate(exprs):
        _egraph.let(f"expr{i}", expr)
    _egraph.saturate(max=5)
    unique_expr = []
    visited = [str(empty)]
    simplified_exprs = []

    for i, expr in enumerate(exprs):
        e = _egraph.extract(expr)
        simplified_exprs.append(e)
    # simplified_exprs = [_egraph.extract(expr) for expr in exprs]
    # _egraph.display()
    for i, e in enumerate(simplified_exprs):
        e_str = str(e)
        if e_str not in visited:
            unique_expr.append(i)
            visited.append(e_str)
    _egraph.pop()
    return unique_expr


def _extract_list_index_divide(exprs: List[RelExpr], indices: List, size_limit) -> List[int]:
    assert indices

    if len(indices) <= size_limit:
        return [indices[i] for i in _extract_list_index_once([exprs[i] for i in indices])]

    return extract_list_index(exprs, indices[0:size_limit]) \
        + extract_list_index(exprs, indices[size_limit:])


def extract_list_index(exprs: List[RelExpr], indices: List = None) -> List[int]:
    """
    Extract unique expressions and return their indices.
    :param exprs: a list of expressions
    :param indices: expression indices
    :return: a list of indices

    Example:
    extract_list([M|R, R|M]) = [0]
    """

    if indices is None:
        indices = range(len(exprs))

    size_limit = 200
    list_length = len(indices)

    while list_length > size_limit:
        indices = _extract_list_index_divide(exprs, indices, size_limit)
        new_list_length = len(indices)
        if new_list_length < list_length:
            list_length = new_list_length
        else:
            size_limit *= 2

    return _extract_list_index_divide(exprs, indices, size_limit)
