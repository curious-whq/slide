


"""
transform the synthesized ppo (GNode instance) into a python function
"""
from typing import List

from src.slide.synth.gnode import GNode

MEM_TYPES = ['M', 'RCsc', 'AQ', 'RL', 'R', 'W', 'AMO', 'X','AQRL', 'XSc', 'XLr']
RELATIONS = ['rmw', 'addr', 'data', 'ctrl', 'fence', 'po', 'loc', 'rf', 'rfi', 'rsw', 'co', 'coi', 'coe', 'fr', 'fri', 'fre', 'po-loc',
                 'fence_rw_rw','fence_rw_w','fence_rw_r','fence_r_rw','fence_r_r',
             'fence_r_w','fence_w_rw','fence_w_w','fence_w_r','fence_tso','fence'
             ]

def transform(root: GNode,ppo_index=0):
    """
    Transform gnode tree (i.e., root) into a python function.
    1) assign event names to nodes;
    2) get python function string by tree traversing.
    """
    event_names = assign_event_names(root)
    python_func_body = get_python_func(root)

    python_func_name = 'ppo_candidate_func'+str(ppo_index)
    

    # identify event parameters of the function
    leaf_nodes = [node for node in root.get_leaf_nodes() if node.type in MEM_TYPES + RELATIONS]
    event_name_para_1 = leaf_nodes[0].get_left_event()
    event_name_para_2 = leaf_nodes[-1].get_right_event()

    if len(event_names) == 1:  # e.g., [M]
        assert event_names[0] == 'e1'
        python_func_string = f"""
def {python_func_name}(ra, e1: Event, e2: Event) -> bool:
    return {python_func_body}
"""
    elif len(event_names) == 2:
        python_func_string = f"""
def {python_func_name}(ra, {event_names[0]}: Event, {event_names[-1]}: Event) -> bool:
    return {python_func_body}
"""
    #CHANGE:all(not)=any(not)
    else:  # when has more than two events, need to add any([xxx])
        quantifier='any'
        if 'not' in python_func_body:
            quantifier='all'
        event_string = ''
        for event_name in event_names:
            if event_name not in [event_name_para_1, event_name_para_2]:
                event_string += f' for {event_name} in ra.execution '
        python_func_string = f"""
def {python_func_name}(ra, {event_name_para_1}: Event, {event_name_para_2}) -> bool:
    return {quantifier}([{python_func_body} {event_string}])
# """

    python_func_string=python_func_string.replace('po-loc','po_loc').replace('-1','__1')  
    return python_func_string


def assign_event_names(root: GNode):
    """
    Assign event names to relation nodes and mem_typs nodes.
    """
    leaf_nodes = root.get_leaf_nodes()
    cnt = 1
    # # filter out [ ] ( )
    # leaf_nodes = [leaf_node for leaf_node in leaf_nodes if leaf_node.type not in ['[', ']', '(',
    #                                                                               ')']]
    for i, leaf_node in enumerate(leaf_nodes):
        if leaf_node.type in MEM_TYPES or leaf_node.type in RELATIONS:  # one event
            # for the first node
            if i == 0:
                leaf_node.set_left_event(f'e{cnt}')  # assign a new event, e.g., e1
                if leaf_node.type in MEM_TYPES:
                    cnt = leaf_node.set_right_event(f'e{cnt}', cnt)  # assign the same event,
                    # e.g., e1
                else:
                    cnt = leaf_node.set_right_event(f'e{cnt + 1}', cnt + 1)  # assign a new event,
                    # e,g., e2
            else:
                """
                for the target nodes except for the first node, the general pipeline:
                1) locate the operator node, the left operand node, and the right_operand_node (
                that contains the target node)
                3) assign the right event of the left operand node as the left event of the 
                target node.
                """
                operator_node, left_operand_node, right_operand_node = locate_operator_node(
                    leaf_node)

                right_event = get_right_event(left_operand_node)
                # for operator node \ and |, left_operand_node event names are consistent with
                # the right_operand_node event names.
                if operator_node.type in ['\\', '|']:
                    left_event = get_left_event(left_operand_node)  # further get left event
                    leaf_node.set_left_event(left_event)

                    right_most_node = get_rightmost_node(right_operand_node)  # find the rightmost
                    # node
                    right_most_node.set_right_event(right_event)
                # otherwise, assign the right event of the left operand node as the left event of
                # the target node.
                else:
                    leaf_node.set_left_event(right_event)

                # set right event of the leaf_node
                if leaf_node.type in MEM_TYPES:
                    leaf_node.set_right_event(leaf_node.get_left_event())
                else:
                    cnt = leaf_node.set_right_event(f'e{cnt}', cnt)
    # get and return the used event names
    event_names = []
    for i in range(1, cnt):
        event_names.append(f"e{i}")
    return event_names


def locate_operator_node(leaf_node: GNode) -> List[GNode]:
    """
    Identify the tree where the leaf_node is in the right operand, and then
    1) locate operator
    2) locate left operand node
    return: operator, left_operand, right_operand
    """
    cur_node = leaf_node
    parent = cur_node.parent
    while parent:
        if len(parent.children) == 3 and parent.children[-1] == cur_node:
            return [parent.children[1], parent.children[0], parent.children[2]]
        cur_node = parent
        parent = parent.parent
    assert False, f'error in loop for leaf node: {leaf_node}'


def get_rightmost_node(node: GNode):
    """
    Get the rightmost mem_type or relation node of the given node
    """
    for node in reversed(node.get_leaf_nodes()):
        if node.type in MEM_TYPES or node.type in RELATIONS:
            return node


def get_right_event(operand_node: GNode):
    """
    Get the right event name of the operand node
    """
    for node in reversed(operand_node.get_leaf_nodes()):
        if node.type in MEM_TYPES or node.type in RELATIONS:
            return node.get_right_event()


def get_left_event(operand_node: GNode):
    """
    Get the left event name of the operand node
    """
    for node in operand_node.get_leaf_nodes():
        if node.type in MEM_TYPES or node.type in RELATIONS:
            return node.get_left_event()


def get_python_func(root: GNode):
    """
    Get python function string by tree traversing.
    """
    # the root is supposed to be expr type
    # print('root txt',root.type,root)
    if root.type == 'expr':
        children = root.children
        op_child = find_operator(root)  # start from the operator node
        if op_child:
            assert len(children) == 3 and children.index(op_child) == 1
            if op_child.type == ';':  # ; -> and
                return f"{get_python_func(children[0])} and {get_python_func(children[2])}"
            elif op_child.type == '|':  # | -> or
                return f"({get_python_func(children[0])} or {get_python_func(children[2])})"
            else:  # \ -> not()
                return f"{get_python_func(children[0])} and not ({get_python_func(children[2])})"
        else:  # if there is no operator node, then the root could be mem_type node (e.g., [M]),
            # relation node (e.g., rsw), or other node (e.g., (expr) )
            return get_python_func(children[0])
    elif root.type == 'RELATION':
        children = root.children
        if str(root) in MEM_TYPES:
            return f"ra.{str(children[0])}({children[0].get_left_event()})"
        elif str(root) in RELATIONS:
            return f"ra.{str(children[0])}({children[0].get_left_event()},{children[0].get_right_event()})"
    else:
        raise Exception(f'Unexpected GNode to be handled:{root}')


def find_operator(root: GNode):
    """
    Find operator node.
    """
    for child in root.children:
        if child.type in [';', '\\', '|']:
            return child
    return None
