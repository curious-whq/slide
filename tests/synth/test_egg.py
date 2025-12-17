

"""Test for egglog"""

from __future__ import annotations

from egglog import *


class TestEgg:
    """
    For egglog, see https://egglog-python.readthedocs.io/latest/index.html.
    For examples, see https://egglog-python.readthedocs.io/latest/auto_examples/.
    """

    def test_schedule(self):
        left = relation("left", i64)
        right = relation("right", i64)

        x, y = vars_("x y", i64)
        step_left = ruleset(
            rule(
                left(x),
                right(x),
            ).then(left(x + 1)),
        )
        step_right = ruleset(
            rule(
                left(x),
                right(y),
                eq(x).to(y + 1),
            ).then(right(x)),
        )

        egraph = EGraph()
        egraph.register(left(i64(0)), right(i64(0)))
        egraph.run((step_right.saturate() + step_left.saturate()) * 10)
        egraph.check(left(i64(10)), right(i64(9)))
        egraph.check_fail(left(i64(11)), right(i64(10)))

    def test_eq(self):
        class Num(Expr):
            def __init__(self, value: i64Like) -> None: ...

            @classmethod
            def var(cls, name: StringLike) -> Num: ...

            def __add__(self, other: Num) -> Num: ...

            def __mul__(self, other: Num) -> Num: ...

        expr1 = Num(2) * (Num.var("x") + Num(3))
        expr2 = Num(6) + Num(2) * Num.var("x")

        a, b, c = vars_("a b c", Num)
        i, j = vars_("i j", i64)

        check(
            # Check that these expressions are equal
            eq(expr1).to(expr2),
            # After running these rules, up to ten times
            ruleset(
                rewrite(a + b).to(b + a),
                rewrite(a * (b + c)).to((a * b) + (a * c)),
                rewrite(Num(i) + Num(j)).to(Num(i + j)),
                rewrite(Num(i) * Num(j)).to(Num(i * j)),
            )
            * 10,
            # On these two initial expressions
            expr1,
            expr2,
        )

    def test_eq2(self):
        class Num(Expr):
            def __init__(self, value: i64Like) -> None: ...

            @classmethod
            def var(cls, name: StringLike) -> Num:  ...

            def __add__(self, other: Num) -> Num: ...

            def __mul__(self, other: Num) -> Num: ...

        egraph = EGraph()
        egraph.let("expr1", Num(2) * (Num.var("x") + Num(3)))
        egraph.let("expr2", Num(6) + Num(2) * Num.var("x"))

        @egraph.register
        def _num_rule(a: Num, b: Num, c: Num, i: i64, j: i64):
            yield rewrite(a + b).to(b + a)
            yield rewrite(a * (b + c)).to((a * b) + (a * c))
            yield rewrite(Num(i) + Num(j)).to(Num(i + j))
            yield rewrite(Num(i) * Num(j)).to(Num(i * j))

        egraph.saturate()
        # egraph.display()

    def test_fib(self):
        @function
        def fib(x: i64Like) -> i64: ...

        f0, f1, x = vars_("f0 f1 x", i64)
        check(
            eq(fib(i64(7))).to(i64(21)),
            ruleset(
                rule(
                    eq(f0).to(fib(x)),
                    eq(f1).to(fib(x + 1)),
                ).then(set_(fib(x + 2)).to(f0 + f1)),
            )
            * 7,
            set_(fib(0)).to(i64(1)),
            set_(fib(1)).to(i64(1)),
        )

    def test_set(self):
        egraph = EGraph()

        class Set(Expr):
            def __init__(self) -> None: ...

            @classmethod
            def var(cls, name: StringLike) -> Set:  ...

            def __and__(self, other: Set) -> Set: ...

            def __or__(self, other: Set) -> Set: ...

            def __rtruediv__(self, other: Set) -> Set: ...

        A, B = Set.var('A'), Set.var('B')

        @egraph.register
        def _set_rule(a: Set, b: Set):
            yield rewrite(a & a).to(a)
            yield rewrite(a | a).to(a)
            yield rewrite(a & b).to(b & a)
            yield rewrite(a | b).to(b | a)

        expr1 = egraph.let("expr1", B | A)
        expr2 = egraph.let("expr2", (A & A) | B)

        egraph.saturate()
        # 'B | A' is equal to '(A & A) | B'
        egraph.check(eq(expr1).to(expr2))
        # egraph.display()

    def test_a22(self):
        egraph = EGraph()

        class Num(Expr):
            def __init__(self, value: i64) -> None: ...

            @classmethod
            def var(cls, name: StringLike) -> Num:  ...

            def __mul__(self, other: Num) -> Num: ...

            def __truediv__(self, other: Num) -> Num: ...

            def __lshift__(self, other: Num) -> Num: ...

        a = Num.var('a')

        @egraph.register
        def _set_rule(a: Num, b: i64):
            yield rewrite(a * Num(b) / Num(b)).to(a)
            yield rewrite(a * Num(i64(2))).to(a << Num(i64(1)))

        expr1 = egraph.let("expr1", a * Num(i64(2)) / Num(i64(2)) * Num(i64(2)))
        expr2 = egraph.let("expr2", a << Num(i64(1)))
        egraph.saturate()
        egraph.check(eq(expr1).to(expr2))
        # egraph.display()

    def test_boolean(self):
        T = Bool(True)
        F = Bool(False)
        check(eq(T & T).to(T))
        check(eq(T & F).to(F))
        check(eq(T | F).to(T))
        check(ne(T | F).to(F))

        check(eq(i64(1).bool_lt(2)).to(T))
        check(eq(i64(2).bool_lt(1)).to(F))
        check(eq(i64(1).bool_lt(1)).to(F))

        check(eq(i64(1).bool_le(2)).to(T))
        check(eq(i64(2).bool_le(1)).to(F))
        check(eq(i64(1).bool_le(1)).to(T))

        R = relation("R", i64)

        @function
        def f(i: i64Like) -> Bool: ...

        i = var("i", i64)
        check(
            eq(f(0)).to(T),
            ruleset(rule(R(i)).then(set_(f(i)).to(T))) * 3,
            R(i64(0)),
        )
