
import itertools
import os.path
from typing import List

from src.slide import config
from src.slide.synth.gnode import GNode, Grammar
from src.slide.synth.rel_expr import extract_list_index
from src.slide.synth.sym_expr import SymExprEGraph
from src.slide.utils import file_util, time_util

# flags
flag_filter_by_commutative_law_equivalence = False


class Spec:
    def __init__(self):
        pass

    def validate(self, program):
        return False


def bottom_up_enum_search(g: Grammar = Grammar(), spec: Spec = Spec(), max_num_of_patches: int = 1000):
    """
    The implementation of bottom up search conforms to the Algorithm 1 in the paper
    "Program Synthesis with Best-First Bottom-Up Search" (JAIR 2023)
    """
    output_file = os.path.join(config.OUTPUT_DIR, 'bottom_up_enum_search.log')
    file_util.clear_file(output_file)

    p_list = []
    size = 1
    eq_size_limit = 6
    while True:  # or: while not timeout
        new_progs = generate_new_programs(g, p_list, size, max_num_of_patches - len(p_list))
        if len(new_progs) == 0:
            size += 1
            continue

        p_list.extend(new_progs)
        print(f"size limit: {size}. #new program: {len(new_progs)} #program in total: {len(p_list)}")

        # Equivalence Checking
        if size <= eq_size_limit:
            # when size > eq_size_limit, eq check is too slow
            rel_expr_list = [node.to_rel_expr() for node in p_list]
            time_util.update_start_time()
            indices = extract_list_index(rel_expr_list)
            p_list = [p_list[index] for index in indices]
            print(f"#unique programs: {len(p_list)}, time_cost: {time_util.cal_time_cost()}\n")

        file_util.write_list_to_file(output_file, p_list, False)

        if len(p_list) > max_num_of_patches:
            break
        size += 1

    return p_list


def generate_new_programs(g: Grammar, p_list: List[GNode], size: int, max_num_of_patches: int):
    progs = []

    for rule in g.rules:
        size_rule = rule.get_size()
        if size_rule > size:
            continue
        non_terminals = rule.get_non_terminal_leaf_nodes()
        size_nt = len(non_terminals)
        if size_nt:
            # create a map for size to p in p_list
            print('p_list')
            for p in p_list:
                print(p)
            p_sizes = list(set([p.get_size() for p in p_list]))
            p_sizes.sort()
            size_p_map = {s: [] for s in p_sizes}
            for p in p_list:
                size_p_map[p.get_size()].append(p)

            # calculate the size limit of p for any non-terminal
            size_rest = size - size_rule + size_nt
            size_limit = size_rest - (size_nt - 1) * p_sizes[0]
            if size_limit < p_sizes[0]:
                return progs

            def find_sum_combs(n, target_sum, numbers):
                nums = list(filter(lambda x: x <= target_sum, numbers))
                combinations = list(itertools.product(nums, repeat=n))
                return [c for c in combinations if sum(c) == target_sum]

            combs = list(set(find_sum_combs(size_nt, size_rest, p_sizes)))

            # Check Equivalence by SymExpr
            comb_sym_map = {
                comb: rule.to_sym_expr([f't{n}' for n in comb]) for comb in combs
            }
            sym_exprs = list(comb_sym_map.values())
            egraph = SymExprEGraph(sym_exprs)
            egraph.saturate()
            combs = [comb for comb in combs if egraph.is_unique(comb_sym_map[comb])]

            def find_combs_lists(lists, index=0, current=None):
                if current is None:
                    current = []
                if index == len(lists):
                    yield current
                else:
                    for item in lists[index]:
                        yield from find_combs_lists(lists, index + 1, current + [item])

            for comb in combs:
                gen = find_combs_lists(list(map(lambda s: size_p_map[s], comb)))
                try:
                    while len(progs) <= max_num_of_patches:
                        targets = next(gen)
                        prog = expand(rule, non_terminals, targets)
                        progs.append(prog)
                except StopIteration:
                    pass
        else:
            # rules without any non-terminals
            if size_rule == size:
                progs.append(rule)  # need deepcopy?

    return progs


def program_generator(g: Grammar, p_list: List[GNode], size: int):
    for rule in g.rules:
        size_rule = rule.get_size()
        if size_rule > size:
            continue
        non_terminals = rule.get_non_terminal_leaf_nodes()
        size_nt = len(non_terminals)
        if size_nt:
            # create a map for size to p in p_list
            p_sizes = list(set([p.get_size() for p in p_list]))
            p_sizes.sort()
            size_p_map = {s: [] for s in p_sizes}
            for p in p_list:
                size_p_map[p.get_size()].append(p)

            # calculate the size limit of p for any non-terminal
            size_rest = size - size_rule + size_nt
            size_limit = size_rest - (size_nt - 1) * p_sizes[0]
            if size_limit < p_sizes[0]:
                return

            def find_sum_combs(n, target_sum, numbers):
                nums = list(filter(lambda x: x <= target_sum, numbers))
                combinations = list(itertools.product(nums, repeat=n))
                return [c for c in combinations if sum(c) == target_sum]

            combs = list(set(find_sum_combs(size_nt, size_rest, p_sizes)))

            # Check Equivalence by SymExpr
            comb_sym_map = {
                comb: rule.to_sym_expr([f't{n}' for n in comb]) for comb in combs
            }
            sym_exprs = list(comb_sym_map.values())
            egraph = SymExprEGraph(sym_exprs)
            egraph.saturate()
            combs = [comb for comb in combs if egraph.is_unique(comb_sym_map[comb])]

            def find_combs_lists(lists, index=0, current=None):
                if current is None:
                    current = []
                if index == len(lists):
                    yield current
                else:
                    for item in lists[index]:
                        yield from find_combs_lists(lists, index + 1, current + [item])

            for comb in combs:
                gen = find_combs_lists(list(map(lambda s: size_p_map[s], comb)))
                try:
                    while True:
                        targets = next(gen)
                        prog = expand(rule, non_terminals, targets)
                        yield prog
                except StopIteration:
                    pass
        else:
            # rules without any non-terminals
            if size_rule == size:
                yield rule  # need deepcopy?


def expand(rule: GNode, non_terminals: List[GNode], targets: List[GNode]) -> GNode:
    # copy and replace
    new_node = GNode()
    new_node.copy_tree(rule)  # copy root tree
    indices = [n.get_index_in_root_tree() for n in non_terminals]
    nodes = [new_node.get_node_by_index(i) for i in indices]
    for i, node in enumerate(nodes):
        assert node.parent, f"[Error] cur node: {node} has no parent"
        node.parent.replace_child(node, targets[i])
        # CAUTION: every time the gnode tree is changed, the gnode traversed_nodes must be reset.
        node.parent.traversed_nodes = None
    new_node.traversed_nodes = None
    return new_node
