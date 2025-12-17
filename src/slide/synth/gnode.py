from enum import Enum
from typing import List, Self

from src.slide.synth.rel_expr import RelExpr, RELATIONS, simplify
from src.slide.synth.sym_expr import SymExpr





MEM_TYPES = ['M', 'RCsc', 'AQ', 'RL', 'R', 'W', 'AMO', 'X', 'AQRL', 'XSc', 'XLr']
RELATION = ['rmw', 'addr', 'data', 'ctrl', 'fence', 'po', 'loc', 'rf', 'rfi', 'rsw', 'co', 'coi', 'coe', 'fr', 'po-loc',
            'fence_rw_rw','fence_rw_w','fence_rw_r', 'fence_w_rw','fence_w_r','fence_w_w','fence_r_rw','fence_r_r','fence_r_w','fence_tso',]


class Grammar:
    def __init__(self):
        """
        Grammar initialization based on the defined grammar in tools/antlr4/cat_ppo/PPO.g4.
        """
        # non-terminals
        self.v = ['expr', 'RELATION']
        # terminals
        self.sigma = {
            'RELATION': #['M', 'R', 'W', 'addr', 'data', 'ctrl']

            ['M', 'RCsc', 'AQ', 'RL', 'R', 'W', 'AMO', 'X','AQRL', 'XSc', 'XLr', 'rmw', 'addr', 'data', 'ctrl', 'fence', 'po',
                'loc', 'rf', 'rfi', 'rsw','po-loc', 'fence_rw_rw','fence_rw_w','fence_rw_r', 'fence_w_rw','fence_w_r','fence_w_w','fence_r_rw','fence_r_r','fence_r_w','fence_tso']
        }
        # rules
        self.rules = [
                         GNode('expr', None, [GNode('expr'), GNode(';'), GNode('expr')]),
                         GNode('expr', None, [GNode('expr'), GNode('|'), GNode('expr')]),
                         GNode('expr', None, [GNode('expr'), GNode('&'), GNode('expr')]),
                         GNode('expr', None, [GNode('expr'), GNode('\\'), GNode('expr')])
                     ] + \
                     [
                         GNode('expr', None, [GNode(relation)]) for relation in self.sigma['RELATION']
                     ]

        # start symbol: is a non-terminal from v
        self.s = GNode('expr')


class ExprType(Enum):
    Mem_type = 1
    Relation = 2
    Unknown = 3  # for gnode with non-terminals
    Conflict = 4  # type conflicts


def is_mem_type(type):
    return type in MEM_TYPES


def is_relation_type(type):
    return type in RELATION


def get_cur_expr_type(type):
    if is_mem_type(type):
        return ExprType.Mem_type
    elif is_relation_type(type):
        return ExprType.Relation
    else:
        assert False


class GNode:
    def __init__(self, type: str = None, parent: Self = None, children: List[Self] = None):
        """
        :param type:
        :param parent:
        :param children: [CAUTION] we cannot set the default value of children as [] in the above
        parameter declaration part, as this would lead to all children list of different nodes at
        the same level have the same address (i.e., the [] above) (I use id() to figure out this
        difficult bug in my prior version). The correct initilization is to assign [] to
        self.children below. 2023-12-23 22:07:37
        """
        self.type = type
        self.parent = parent
        self.children = []
        if children is not None:
            self.add_children_from_list(children)
        self.root = None

        # for transformation from ppo rule to python function
        self.events = [None, None]

        # create this member to avoid repeatedly call number_of_non_terminal_leafs()
        self.number_of_non_terminal_leafs = None

        # for depth limit
        self.depth = None

        # to save time (avoid repeated calculation)
        self.traversed_nodes = []
        # print(f'traversed_nodes: {self.traversed_nodes}')
        # exit()

        # for type (conflict) check
        self.expr_type = None

        # for egg check
        self.rel_expr = None
        self.rel_expr_simp = None

    def get_expr_type(self):
        if self.expr_type is not None:
            return self.expr_type
        self.expr_type = self.cal_expr_type()
        return self.expr_type

    def cal_expr_type(self):
        assert self.get_number_of_non_terminal_leafs() == 0
        if self.is_terminal():
            return get_cur_expr_type(type)
        else:  # expr or RELATION
            if self.type == 'RELATION':
                assert len(self.children) == 1
                return get_cur_expr_type(self.children[0].type)
            else:  # expr
                if len(self.children) == 1:
                    # print('children',self.children)
                    # print('children.type',self.children[0].type)
                    assert self.children[0].type == 'RELATION'
                    return self.children[0].get_expr_type()  # recursive
                else:  # has operator
                    op = self.children[1].type
                    assert op in ['\\', '&', '|', ';']
                    assert len(self.children) == 3
                    left_type = self.children[0].get_expr_type()
                    right_type = self.children[2].get_expr_type()
                    if left_type == ExprType.Conflict or right_type == ExprType.Conflict:  # early check
                        return ExprType.Conflict
                    match op:
                        case '\\' | '&':  # relation \ relation  relation & relation
                            assert left_type == right_type and left_type == ExprType.Relation
                            return left_type
                        case '|':  # relation | relation    mem | mem
                            assert left_type == right_type
                            return left_type
                        case ';':  # relation ; mem   mem ; relation
                            assert left_type != right_type
                            return ExprType.Relation

    def reset(self):
        self.number_of_non_terminal_leafs = None

    def set_left_event(self, event_name: str, cnt: int = -1):
        if self.events[0] is not None:
            return cnt  # avoid repeatedly assignment (due to \ or | operator)
        self.events[0] = event_name
        if cnt > 0:
            return cnt + 1

    def set_right_event(self, event_name: str, cnt=-1):
        if self.events[1] is not None:
            return cnt  # avoid repeatedly assignment (due to \ or | operator)
        self.events[1] = event_name
        if cnt > 0:
            return cnt + 1

    def get_right_event(self):
        return self.events[1]

    def get_left_event(self):
        return self.events[0]

    def is_terminal(self):
        return self.type not in Grammar().v

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def get_root(self):
        """
        Get root node.
        """
        if self.root:
            return self.root
        if self.parent is None:
            return self

        parent = self.parent
        while parent.parent is not None:
            parent = parent.parent
        return parent
    

    def add_children(self, *children: List[Self]):
        """
        Add children in the format of elements, e.g., child_1, child_2, child_3
        """
        for child in children:
            self.children.append(child)
            # set parent for each child
            child.set_parent(self)

    def add_children_from_list(self, children: List[Self]):
        """
        Add children in the format of list, e.g., [child_1, child_2, child_3]
        """
        for child in children:
            self.children.append(child)
            child.set_parent(self)

    def replace_child(self, cur_child: Self, new_child: Self):
        index = self.children.index(cur_child)
        self.children[index] = new_child
        new_child.set_parent(self)

    def set_parent(self, parent: Self):
        self.parent = parent

    def traverse_cur_node(self) -> List[Self]:
        """
        depth-first search (pre-order)
        """
        # print('traverse_cur_node',self,type(self),self.traversed_nodes)
        if self.traversed_nodes!= None:
            if len(self.traversed_nodes) != 0:
                return self.traversed_nodes
        self.traversed_nodes = [self]  # add itself
        # print('children:',self.children)
        for child in self.children:
            self.traversed_nodes.extend(child.traverse_cur_node())
        return self.traversed_nodes

    def traverse_root_tree(self):
        all_nodes = self.get_root().traverse_cur_node()
        return all_nodes

    def get_size(self):
        # print('get_size:',self)
        return len(self.traverse_cur_node())

    def get_leaf_nodes(self):
        traversed_nodes = self.traverse_cur_node()
        leaf_nodes = [traversed_node for traversed_node in traversed_nodes if
                      traversed_node.is_leaf()]
        return leaf_nodes

    def get_non_terminal_leaf_nodes(self):
        return [node for node in self.get_leaf_nodes() if not node.is_terminal()]

    def get_depth(self):
        if self.depth is not None:
            return self.depth
        if len(self.children) == 0:
            self.depth = 1
            return self.depth
        self.depth = max([child.get_depth() for child in self.children]) + 1
        return self.depth

    # @profile
    def get_replacements(self) -> List[Self]:
        """
        only for non-terminals
        """
        assert not self.is_terminal()
        replacements = []
        grammar = Grammar()

        assert self.type in grammar.rules.keys()
        replace_list = grammar.rules[self.type]
        for elements in replace_list:
            """
            1. deep copy the whole tree (get a new tree now)
            2. find the node and set its children.
            """
            # PRUNE: avoid meaningless replacement, e.g., (po) -> ((po))
            if self.parent:
                leaf_nodes = self.parent.get_leaf_nodes()
                if "".join(elements) == '(expr)' or "".join(elements) == '[rel]':
                    if leaf_nodes[0].type in ['[', '('] and leaf_nodes[-1].type in [']', ')']:
                        continue

            # start replacement process
            root = self.get_root()
            root_deepcopy = GNode()
            root_deepcopy.copy_tree(root)  # copy root tree

            cur_index = self.get_index_in_root_tree()  # locate current node
            node = root_deepcopy.get_node_by_index(cur_index)  # get current node in copy root
            children = [GNode(element) for element in elements]
            node.add_children_from_list(children)  # replace: add elements to node children

            # append the replaced full tree
            replacements.append(root_deepcopy)
        return replacements

    # @profile
    def copy_tree(self, ori_tree: Self):
        """
        copy the original tree
        members:
        self.type = type
        self.parent = parent
        self.children = [] #
        self.root = None
        self.events = [None, None]  # not included in copy_tree()
        self.number_of_non_terminal_leafs = None # set as None
        """
        # copy
        self.copy_members(ori_tree)
        for child in ori_tree.children:
            new_child = GNode()
            new_child.copy_tree(child)
            self.add_children(new_child)

    def copy_members(self, ori_tree: Self):
        self.type = ori_tree.type
        self.parent = ori_tree.parent
        self.root = ori_tree.root

    def get_number_of_terminal_leafs(self):
        return len(self.get_leaf_nodes()) - len(self.get_non_terminal_leaf_nodes())

    def get_number_of_non_terminal_leafs(self):
        if self.number_of_non_terminal_leafs is not None:  # must use "number_of_non_terminal_leafs is not None" rather than number_of_non_terminal_leafs. as when number_of_non_terminal_leafs is 0, the expr "number_of_non_terminal_leafs" will be False.
            return self.number_of_non_terminal_leafs
        self.number_of_non_terminal_leafs = len(self.get_non_terminal_leaf_nodes())
        return self.number_of_non_terminal_leafs

    def get_index_in_root_tree(self):
        all_nodes = self.traverse_root_tree()
        return all_nodes.index(self)

    def get_node_by_index(self, index: int):
        all_nodes = self.traverse_root_tree()
        return all_nodes[index]

    def __str__(self):
        # return "".join([node.type for node in self.get_leaf_nodes()])
        str = self.get_str()
        if str[0] == '(' and str[-1] == ')':
            return str[1:-1]
        return str

    def get_str(self):
        if len(self.children) == 0:
            return self.type
        else:
            if self.type == "RELATION":
                return self.children[0].type
            else:  # expr
                if len(self.children) == 1:
                    return self.children[0].get_str()
                else:
                    assert len(self.children) == 3
                    left = self.children[0]
                    op = self.children[1]
                    right = self.children[2]
                    return f"({left.get_str()}{op.type}{right.get_str()})"

    def __repr__(self):
        # return f"{str(self)} events: {self.events}"
        # return f"{str(self)}"
        return "".join([node.type for node in self.get_leaf_nodes()])

    def is_equal(self, node: Self):
        if self.type != node.type:
            return False
        if len(self.children) != len(node.children):
            return False
        else:
            for i, child in enumerate(self.children):
                if not child.is_equal(node.children[i]):
                    return False
        return True

    def to_sym_expr(self, leaves: List[str]) -> SymExpr:
        """
        transfer to sym_expr
        """
        assert len(self.children) == 3
        assert len(leaves) == 2
        left, right = SymExpr(leaves[0]), SymExpr(leaves[1])
        op = self.children[1]
        match op.type:
            case '|':
                return left | right
            case '&':
                return left & right
            case '\\':
                return left // right
            case ';':
                return left.sequence(right)
            case _:
                assert False

    def to_rel_expr(self) -> RelExpr:
        """
        transfer to rel_expr
        """
        if self.rel_expr is not None:
            return self.rel_expr
        self.rel_expr = self._get_rel_expr()
        return self.rel_expr

    def to_rel_expr_simp(self) -> RelExpr:
        if self.rel_expr_simp is not None:
            return self.rel_expr_simp
        self.rel_expr_simp = simplify(self.to_rel_expr())
        return self.rel_expr_simp

    def _get_rel_expr(self):
        if len(self.children) == 0:
            return get_rel_expr_for_type(self.type)
        else:
            if self.type == "RELATION":
                return get_rel_expr_for_type(self.children[0].type)
            else:  # expr
                # print('_get_rel_expr',self.children)
                if len(self.children) == 1:
                    return self.children[0]._get_rel_expr()
                else:
                    assert len(self.children) == 3
                    left = self.children[0]
                    op = self.children[1]
                    right = self.children[2]
                    match op.type:
                        case '|':
                            return left._get_rel_expr() | right._get_rel_expr()
                        case '&':
                            return left._get_rel_expr() & right._get_rel_expr()
                        case '\\':
                            return left._get_rel_expr() // right._get_rel_expr()
                        case ';':
                            return left._get_rel_expr().sequence(right._get_rel_expr())
                        case _:
                            # print(op.type)
                            assert False


def get_rel_expr_for_type(type):
    # print('get_rel_expr_for_type',type)
    return RELATIONS[type]
