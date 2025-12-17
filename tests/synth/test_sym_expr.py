

from src.slide.synth.sym_expr import *


class TestSymExpr:

    def test_commutative_law_or(self):
        A, B = SymExpr('a'), SymExpr('b')
        a_or_b, b_or_a = A | B, B | A
        sf = SymExprEGraph([a_or_b, b_or_a])
        sf.saturate()
        assert sf.is_unique(a_or_b) ^ sf.is_unique(b_or_a)

    def test_commutative_law_and(self):
        A, B = SymExpr('a'), SymExpr('b')
        a_and_b, b_and_a = A & B, B & A
        sf = SymExprEGraph([a_and_b, b_and_a])
        sf.saturate()
        assert sf.is_unique(a_and_b) ^ sf.is_unique(b_and_a)