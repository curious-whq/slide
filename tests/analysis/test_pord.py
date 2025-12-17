
"""Test for Partial Order"""
import copy
import itertools


def total_order(n_writes, n_others):
    exes = list(itertools.permutations(range(n_writes + n_others)))
    return len(exes)


def partial_order(n_writes, n_others):
    write_seqs = list(itertools.permutations([f'w{i}' for i in range(n_writes)]))
    others = [f'o{i}' for i in range(n_others)]
    other_seqs = [[[] for i in range(n_writes + 1)]]

    for o in others:
        other_seqs_new = []
        for ps in other_seqs:
            for pos in range(n_writes + 1):
                ps_new = copy.deepcopy(ps)
                ps_new[pos].append(o)
                other_seqs_new.append(ps_new)
        other_seqs = other_seqs_new[:]

    def merge(write_seq, other_seq):
        exe = []
        for i in range(n_writes):
            exe.extend(other_seq[i])
            exe.append(write_seq[i])
        exe.extend(other_seq[-1])
        return exe

    return [merge(w, o) for w in write_seqs for o in other_seqs]


NW = 5
NO = 5


class TestPord:
    def test_partial_order(self):
        exes = partial_order(NW, NO)
        print(len(exes))

    def test_total_order(self):
        print(total_order(NW, NO))
