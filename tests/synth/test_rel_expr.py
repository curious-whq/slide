

import sys
sys.path.append('../src')
from src.slide.synth.rel_expr import *

class TestRelExpr:

    def test_commutative_law(self):
        assert check_eq(R | W, W | R)
        assert check_eq(R & W, W & R)
        assert check_eq(R & (data | addr), (data | addr) & R)
        assert check_eq(W & (data | addr), (addr | data) & W)

    def test_empty_law(self):
        assert check_eq(M & empty, empty)
        assert check_eq(R | empty, R)
        assert check_eq(empty | empty, empty)
        assert check_eq(empty & empty, empty)
        assert check_eq((data | ctrl | addr) & empty, empty)

    def test_distribution_law(self):
        assert check_eq((M & R) | (W & R), (M | W) & R)
        assert check_eq((data & po) | (loc & po), (data | loc) & po)
        assert check_eq((data & loc) | po, (data | po) & (loc | po))

    def test_inclusion(self):
        assert not check_eq(M & R, M)
        assert check_eq((R & M) | W, (M & W) | R)
        assert check_eq(R | M, M)
        assert check_eq(R.sequence(M), R)
        assert check_eq(M.sequence(W), W)
        assert check_eq(M.sequence(W), W.sequence(W))
        assert check_eq(W.sequence(M), M.sequence(W))

    def test_exclusive(self):
        assert check_eq(R & W, empty)
        assert check_eq(R & ctrl, empty)
        assert check_eq(M & data, empty)

    def test_simplification(self):
        # slide.log.LOG_LEVEL = 'DEBUG'
        assert can_be_simplified(R | M)
        assert not can_be_simplified(data | ctrl)
        assert can_be_simplified(data.sequence(ctrl))  # grammar constraint

    def test_cost(self):
        assert cost(R | M) == 2
        assert cost(R | M & R) == 3
        assert cost(R.sequence(M) | M) == 3
        assert cost((data & po) | (loc & po)) == 4

    def test_simplify(self):
        assert cost(simplify(R | M)) == 1

    def test_extract_list(self):
        assert len(extract_list([R | M, M])) == 1
        assert len(extract_list_index([R | M, M])) == 1
        assert len(extract_list([M | R, M, R | M, data | ctrl, ctrl | data] * 100)) == 2
        assert extract_list_index([M | R, M, R | M, data | ctrl, ctrl | data] * 100) == [0, 3]
        assert extract_list_index([M | R, M, R | M, data | ctrl, ctrl | data] * 200) == [0, 3]

    def test_to_be_deleted(self):
        assert check_eq(rfi | rf, rf)
        assert check_eq(rfi // rf, empty)
        assert check_eq(M.sequence(rmw), rmw)

        assert check_eq(W.sequence(addr), empty)
        assert check_eq(W.sequence(ctrl), empty)
        assert check_eq(W.sequence(data), empty)

        assert check_eq(M.sequence(addr), addr)
        assert check_eq(rsw // rsw, empty)

    def test_M(self):
        assert check_eq(M | M, M)
        assert check_eq(M & M, M)
        assert check_eq(M | rmw, empty)
        assert len(extract_list_index([M | data, M | rmw], [0, 1])) == 0
