

from __future__ import annotations

from egglog import *


class SymExpr(Expr):
    def __init__(self, rel: StringLike) -> None: ...

    @classmethod
    def var(cls, name: StringLike) -> SymExpr: ...

    def __or__(self, other: SymExpr) -> SymExpr: ...

    def __and__(self, other: SymExpr) -> SymExpr: ...

    def __floordiv__(self, other: SymExpr) -> SymExpr: ...

    def sequence(self, other: SymExpr) -> SymExpr: ...


empty = SymExpr('empty')


class SymExprEGraph:
    def __init__(self, exprs: list[SymExpr]):
        _egraph = EGraph()
        self.exprs = exprs

        @_egraph.register
        def _rel_expr_rule(a: SymExpr, b: SymExpr):
            """
            Simplification Rules for SymExpr.
            """

            # commutative law: AB = BA
            yield rewrite(a | b).to(b | a)
            yield rewrite(a & b).to(b & a)

            # M1;M2 => M1&M2
            yield rewrite(SymExpr('t2').sequence(SymExpr('t2'))).to(empty)

        for i, expr in enumerate(exprs):
            _egraph.let(f"expr{i}", expr)

        self.egraph = _egraph
        self.unique_exprs = []

    def saturate(self, limit: int = 5):
        self.egraph.saturate(max=limit)
        visited = []
        simplified = [self.egraph.extract(expr) for expr in self.exprs]

        for e in simplified:
            e_str = str(e)
            if e_str not in visited:
                self.unique_exprs.append(str(e))
                visited.append(e_str)

    def is_unique(self, x: SymExpr) -> bool:
        return str(x) in self.unique_exprs
