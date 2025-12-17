

"""
implement top-down enumerative search based synthesis
"""
import os.path
from collections import deque
from typing import List

import networkx as nx
import toolz as tz

from src.slide import config
from src.slide.analysis import Event, GlobalRelationAnalyzer
from src.slide.synth.gnode import GNode
from src.slide.utils import file_util

print_flag = False
# this is a previous version (2024-1-29), now commented. The new version if implemented in synth.py (for bottom-up
# search)
# TODO: turn into the Grammer of bottom-up search
class Grammar:
    def __init__(self):
        """
        Grammar initialization based on the defined grammar in tools/antlr4/cat_ppo/PPO.g4.
        """
        # non-terminals
        self.v = ['expr']
        # terminals
        self.sigma = {
            'MEM_TYPE': ['M', 'RCsc', 'AQ', 'RL', 'R', 'W', 'AMO', 'X'],
            'RELATION': ['rmw', 'addr', 'data', 'ctrl', 'fence', 'po', 'po-loc', 'rf', 'rfi', 'rsw']
        }
        # relations
        self.relations = {
            'expr': [
                ['RELATION'],
                ['(', 'expr', ')'],
                ['expr', '\\', 'expr'],
                ['expr', '|', 'expr'],
                ['expr', ';', 'expr'],
                ['[', 'MEM_TYPE', ']']
            ],
            'MEM_TYPE': [[memtype] for memtype in self.sigma['MEM_TYPE']],
            'RELATION': [[relation] for relation in self.sigma['RELATION']]}

        # start symbol: is a non-terminal from v
        self.s = GNode('expr')


class Spec:
    def __init__(self):
        pass

    def validate(self, program):
        return False


# @profile
def top_down_enum_search(g: Grammar = Grammar(), spec: Spec = Spec(), max_num_of_patches: int = 1000):
    """
    The func implementation is based on the algorithm (see 4.1.1 Top-down Tree search) of the classic book for
    synthesis: <Program synthesis> Sumit Gulwani, Oleksandr Polozov, Rishabh Singh
    """
    output_file = os.path.join(config.OUTPUT_DIR, 'top_down_enum_search.log')
    file_util.clear_file(output_file)
    p_list: List[GNode] = [g.s]
    # pv_set: Set[GNode] = {g.s} # the original algorithm used the pv_set, but seems useless here
    syntax_valid_patch_list = []
    tried_p_list = {g.s}
    syntax_valid_patch_cnt = 0
    repeat_cnt = 0
    all_patch_cnt = 1  # count the initial [g.s] in p_list
    while len(p_list):
        p: GNode = p_list.pop(0)

        # we use a different pipeline, i.e., we run validation outside the synthesis function.
        # if spec.validate(p):
        #     return p

        non_terminals = get_all_non_terminals(p)  # get all non-terminals (to be replaced)
        for non_terminal in non_terminals:  # traverse
            replacements = non_terminal.get_replacements()  # get replacements
            for replacement in replacements:
                if str(replacement) in tried_p_list:  # filter duplicate
                    repeat_cnt += 1
                    continue
                if replacement.get_number_of_non_terminal_leafs() == 0:  # save syntax-valid patch
                    file_util.write_str_to_file(output_file, str(replacement) + '\n')
                    syntax_valid_patch_cnt += 1
                    syntax_valid_patch_list.append(replacement)
                # insert replacement into p_list according to its number_of_non_terminal_leafs.
                if len(p_list) == 0:
                    p_list.append(replacement)
                else:
                    for i, node in enumerate(p_list):
                        has_inserted = False
                        if replacement.get_number_of_non_terminal_leafs() < node.get_number_of_non_terminal_leafs():
                            p_list.insert(i, replacement)
                            has_inserted = True
                            break
                    if not has_inserted:
                        p_list.append(replacement)

                # pv_set.add(replacement)
                tried_p_list.add(str(replacement))
                all_patch_cnt += 1

        if len(syntax_valid_patch_list) > max_num_of_patches:
            break
    # print('tried_p_list:', len(tried_p_list), all_patch_cnt, syntax_valid_patch_cnt, repeat_cnt)
    # for i, node in enumerate(tried_p_list):
    #     print(i, str(node))
    return syntax_valid_patch_list


# @profile
def cal_enum_search_space_with_depth_limit(g: Grammar = Grammar(), spec: Spec = Spec(), max_depth: int = 7):
    # output_file = os.path.join(config.OUTPUT_DIR, 'top_down_enum_search_with_depth_litmit.log')
    # file_util.clear_file(output_file)
    p_list = deque([g.s])
    tried_p_list = {str(g.s): [g.s]}
    all_patch_cnt = 1  # count the initial [g.s] in p_list
    while len(p_list):
        p: GNode = p_list.pop()
        non_terminals = get_all_non_terminals(p)  # get all non-terminals (to be replaced)
        for non_terminal in non_terminals:  # traverse
            replacements = non_terminal.get_replacements()  # get replacements
            for replacement in replacements:
                # continue if exceeds the max_depth
                if replacement.get_depth() > max_depth:
                    # file_util.write_str_to_file(output_file, f'max_depth\n')
                    continue
                # filter node that has the same tree (rather than has the same __str__ output), this is to caculate
                # the search space of the synthesis approach
                replacement_str = str(replacement)
                if replacement_str in tried_p_list.keys():
                    hit = False
                    for node in tried_p_list[replacement_str]:
                        if node.is_equal(replacement):
                            hit = True
                            break
                    if hit:
                        # file_util.write_str_to_file(output_file, f'continue\n')
                        continue
                    else:
                        tried_p_list[replacement_str].append(replacement)
                else:
                    tried_p_list[replacement_str] = [replacement]

                p_list.append(replacement)
                all_patch_cnt += 1
                # file_util.write_str_to_file(output_file, f'{all_patch_cnt}\n')
    print('\n', all_patch_cnt)
    return tried_p_list


def get_all_non_terminals(p: GNode) -> List[GNode]:
    leaf_nodes = p.get_leaf_nodes()
    if leaf_nodes:
        return [leaf_node for leaf_node in leaf_nodes if not leaf_node.is_terminal()]
    else:  # p has no leafs
        if not p.is_terminal():
            return [p]
    return []



if __name__ == "__main__":
    g = Grammar()
    spec = Spec()
    top_down_enum_search(g, spec, 10000000)
