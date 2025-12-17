import copy
from copy import deepcopy
from enum import Enum
from os import WCONTINUED
from typing import List, Tuple

# from networkx.algorithms.walks import number_of_walks

from src.slide.analysis import Event
from src.slide.synth.memory_relation import *



class PPOFlag(Enum):
    Strengthen = 1
    Relaxed = 2

class PPOValidFlag(Enum):
    Valid = 1
    Invalid = 2

class PPOInitFlag(Enum):
    Init = 1 # must be Valid
    Added = 2 # added
    Verified = 3 # pass synth validate
    Removed = 4 # removed

class PPOOpType(Enum):
    StrengthenToInvalid = 1
    RelaxedToInvalid = 2
    ValidToInvalid = 3
    InvalidToValid = 4
    Added = 5
    Pass = 6

class SinglePPO:

    def __init__(self, ppo: List[MemoryRelation], ppo_flag = PPOFlag.Strengthen):
        self.ppo = ppo
        # self.justify_flag = self.justify()
        self.start = None
        self.end = None
        if len(ppo) > 0:
            self.start = ppo[0]
            self.end = ppo[-1]

        self.size = len(ppo)
        self.flag = ppo_flag


    def justify(self): # for special case eg:AMO
        justify_ppo = []
        index = 0
        while(True):
            if index == len(self.ppo):
                break
            # process AMO
            now_relation = self.ppo[index]
            if type(now_relation) == AMO:
                if index + 2 >= len(self.ppo):
                    return False
                other_amo = self.ppo[index + 2]
                if type(other_amo) != AMO:
                    return False # show deprecated
                justify_ppo.append(now_relation)
                index += 2
            else:
                justify_ppo.append(now_relation)

            # last, the index+=1
            index += 1
        self.ppo = justify_ppo
        self.start = self.ppo[0]
        self.end = self.ppo[-1]
        self.size = len(justify_ppo)
        return True


    def rev_flag(self):
        if self.flag == PPOFlag.Strengthen:
            self.flag = PPOFlag.Relaxed
        else:
            self.flag =PPOFlag.Strengthen

    def __repr__(self):
        # cat like
        return ";".join([relation.get_ppo_name() for relation in self.ppo])

    def get_relation_str(self):
        return ';'.join([str(relation) for relation in self.ppo])

    def eq_by_relation_str(self, other):
        if other is None or not isinstance(other, SinglePPO):
            return False
        return self.get_relation_str() == other.get_relation_str()

    def __eq__(self, other):
        if other is None or not isinstance(other, SinglePPO):
            return False
        return str(other) == str(self) # The expression of ppo is in strings, so compare its strings to be equal


    def without_start_and_end_str(self):
        # for MixPPO get str
        ppo_str = self.ppo[1:-1]

        return ';'.join([ppo_item.get_ppo_name() for ppo_item in ppo_str])

    def __hash__(self):
        return hash(str(self))

    def is_contain(self, other):
        # Return True to indicate that self contains other
        contain_unary_dict = {
                R:[Lr,AMO],
                W:[Sc,AMO],
            }
        if self == other:
            return True
        # print(self,other)
        if not (type(self.start) == type(other.start) or (type(self.start) in contain_unary_dict and type(other.start) in contain_unary_dict[type(self.start)])) :
            return False
        if not (type(self.end) == type(other.end) or (type(self.end) in contain_unary_dict and type(other.end) in contain_unary_dict[type(self.end)])) :
            return False

        if type(self.start) in [AMO,Lr,Sc]:
            if self.start.annotation == other.start.annotation:
                pass
            elif self.start.annotation > other.start.annotation:
                return False
            elif not (self.start.annotation > other.start.annotation or self.start.annotation < other.start.annotation):
                return False

        if type(self.end) in [AMO,Lr,Sc]:
            if self.end.annotation == other.end.annotation:
                pass
            elif self.end.annotation > other.end.annotation:
                return False
            elif not (self.end.annotation > other.end.annotation or self.end.annotation < other.end.annotation):
                return False

        #For control dependencies (ctrl), special handling is required,
        # because all instructions following a ctrl instruction
        # have a control dependency on the instructions preceding the ctrl.
        ctrlindexs = []
        # If B must be satisfied if A is satisfied, then B contains A
        queue = []
        queue.append((1, 1, [(0, 0)]))  #
        while len(queue) > 0:
            index, other_index, pair_list = queue.pop(0)
            if index == self.size and other_index == other.size:  # because the end don't need think
                return True
            if index >= self.size or other_index >= other.size:
                continue
            binary_relation, binary_other_relation = self.ppo[index], other.ppo[other_index]
            if binary_relation >= binary_other_relation or index in ctrlindexs:  # eg:po>po-loc
                queue.append((index, other_index + 2, pair_list))

                unary_relation, unary_other_relation = self.ppo[index + 1], other.ppo[other_index + 1]
                # print('u',unary_other_relation,unary_relation)
                if type(unary_relation) == type(unary_other_relation) or (type(unary_relation) in contain_unary_dict and type(unary_other_relation) in contain_unary_dict[type(unary_relation)]) :
                    # print('u',unary_other_relation,unary_relation)
                    if type(unary_relation) in [AMO,Lr,Sc] :
                        if unary_relation.annotation == unary_other_relation.annotation:
                            pass
                        elif unary_relation.annotation > unary_other_relation.annotation:
                            continue
                        elif not( unary_relation.annotation > unary_other_relation.annotation or unary_relation.annotation < unary_other_relation.annotation):
                            continue
                    queue.append((index + 2, other_index + 2, pair_list + [(index + 1, other_index + 1)]))
            if type(binary_relation) in [CtrlS, CtrlD]:
                ctrlindexs.append(index)
        return False

    def check_ppo(self):
        for i, relation in enumerate(self.ppo):
            pre = i - 1 if i != 0 else len(self.ppo) - 1
            suc = i + 1 if i != len(self.ppo) - 1 else 0
            if suc - pre != 2 :
                continue
            assert self.ppo[suc].relation_flag != relation.relation_flag, 'Two neighboring relationships cannot be the same'

            if relation.relation_flag:
                if not relation.check(self.ppo[pre], self.ppo[suc]):
                    return False
        return True


    def get_no_Lr_and_Sc_ppo(self):
        no_Lr_and_Sc_ppo = copy.deepcopy(self.ppo)
        for i in range(len(self.ppo)):
            if isinstance(no_Lr_and_Sc_ppo[i], Sc):
                no_Lr_and_Sc_ppo[i] = W()
            if isinstance(no_Lr_and_Sc_ppo[i], Rmw):
                no_Lr_and_Sc_ppo[i] = PoLoc()
            if isinstance(no_Lr_and_Sc_ppo[i], Lr):
                no_Lr_and_Sc_ppo[i] = R()
        return SinglePPO(no_Lr_and_Sc_ppo, self.flag)
    
    def get_counter_ppo_list(self):
        # return List[(SinglePPO counter_ppo,SinglePPO litmus_ppo)],counter_ppo is want to distinguish, litmus_ppo to create new litmus test
        # if have po/ctrl/po-loc -> return []
        relation_list = self.ppo
        counter_relation_list = []
        litmus_ppo_relation_list = []
        flag =False
        trans_flag = False
        for i in range(len(relation_list)):
            if type(relation_list[i]) in [Po, PoLoc, CtrlD, CtrlS]:
                trans_flag = True
            if type(relation_list[i]) in [Rfi]:
                if trans_flag and type(relation_list[i-1]) == W :
                    return []
                pre_relation_flag = False
                if i-2 > 0 and type(relation_list[i-2]) in [PoLoc,Coi,Fri]:# transitivity
                    pre_relation_flag = True
                suc_relation_flag = False
                if i+2 < len(relation_list) and type(relation_list[i+2]) in[PoLoc,Coi,Fri]:# transitivity

                    suc_relation_flag = True
                if pre_relation_flag or suc_relation_flag:
                    continue
                flag = True
                counter_relation_list.append(PoLoc())
                litmus_ppo_relation_list.append(PoLoc())
                if type(relation_list[i] in [Rfi, Coi]):
                    litmus_ppo_relation_list.append(W())
                else:
                    litmus_ppo_relation_list.append(R())
                litmus_ppo_relation_list.append(PoLoc())
            else:
                counter_relation_list.append(copy.deepcopy(self.ppo[i]))
                litmus_ppo_relation_list.append(copy.deepcopy(self.ppo[i]))
        if not flag :
            return []
        return [(SinglePPO(counter_relation_list, self.flag),SinglePPO(litmus_ppo_relation_list, self.flag))]




class MixPPO:
    def __init__(self, strengthen_ppo: SinglePPO, relaxed_ppo_list: List[SinglePPO], init_flag = PPOInitFlag.Added):
        self.strengthen_ppo = strengthen_ppo
        self.relaxed_ppo_list = relaxed_ppo_list # can be []
        self.start =self.strengthen_ppo.start
        self.end =self.strengthen_ppo.end
        self.check()
        self.init_flag = init_flag
        pass

    def check(self):
        assert self.strengthen_ppo is not None
        assert self.relaxed_ppo_list is not None
        assert self.strengthen_ppo.flag == PPOFlag.Strengthen
        contain_unary_dict = {
                R:[Lr,AMO],
                W:[Sc,AMO]
            }
        for relaxed_ppo in self.relaxed_ppo_list:
            assert relaxed_ppo.flag == PPOFlag.Relaxed
            print(relaxed_ppo)
            print(self.strengthen_ppo)
            assert type(relaxed_ppo.start) == type(self.start) or type(relaxed_ppo.start) in contain_unary_dict.get(type(self.start),[])
            assert type(relaxed_ppo.end) == type(self.end) or type(relaxed_ppo.end) in contain_unary_dict.get(type(self.end),[])
            if type(relaxed_ppo.start) in [AMO, Lr, Sc] and type(self.start) in [AMO, Lr, Sc]:
                assert relaxed_ppo.start.annotation > self.start.annotation or relaxed_ppo.start.annotation == self.start.annotation
            if type(relaxed_ppo.end) in [AMO, Lr, Sc] and type(self.end) in [AMO, Lr, Sc]:
                assert relaxed_ppo.end.annotation > self.end.annotation or relaxed_ppo.end.annotation == self.end.annotation

    def __repr__(self):
        start = self.start
        end = self.end
        outer_list = []
        ppo_str = f"{start.get_ppo_name()};{self.strengthen_ppo.without_start_and_end_str()}"
        for relaxed_ppo in self.relaxed_ppo_list:
            if relaxed_ppo.start != self.start or relaxed_ppo.end != self.end:
                outer_list.append(relaxed_ppo)
                continue
            ppo_str += rf"\({relaxed_ppo.without_start_and_end_str()})"
        ppo_str += f";{end.get_ppo_name()}"
        if outer_list!=[]:
            ppo_str = f'({ppo_str})'
            for relaxed_ppo in outer_list:
                ppo_str += rf'\({str(relaxed_ppo)})'
        return ppo_str

    def get_gnode_form(self):
        cat_form = str(self)
        gnode_form = cat_form.replace('[','')
        gnode_form = gnode_form.replace(']','')
        gnode_form = gnode_form.replace('fencerel(Fence.rw.rw)','fence_rw_rw')
        gnode_form = gnode_form.replace('fencerel(Fence.rw.r)','fence_rw_r')
        gnode_form = gnode_form.replace('fencerel(Fence.rw.w)','fence_rw_w')
        gnode_form = gnode_form.replace('fencerel(Fence.w.rw)','fence_w_rw')
        gnode_form = gnode_form.replace('fencerel(Fence.w.r)','fence_w_r')
        gnode_form = gnode_form.replace('fencerel(Fence.w.w)','fence_w_w')
        gnode_form = gnode_form.replace('fencerel(Fence.r.rw)','fence_r_rw')
        gnode_form = gnode_form.replace('fencerel(Fence.r.r)','fence_r_r')
        gnode_form = gnode_form.replace('fencerel(Fence.r.w)','fence_r_w')
        gnode_form = gnode_form.replace('fencerel(Fence.tso)','fence_tso')
        return gnode_form


def get_ppo_contain_list(ppo: SinglePPO, depth)->List[SinglePPO]:
    ppo_list = []
    ppo_queue = [(ppo,0)]
    ppo_dict = {(ppo,0):1}
    # mutate relation get list
    unary_relation_list = [R(),W(),AMO(),Lr(),Sc(),AMO(Annotation.AQ),AMO(Annotation.RL),Sc(Annotation.RL),Sc(Annotation.RL),Lr(Annotation.AQ),Lr(Annotation.RL),
                           AMO(Annotation.AQRL), Lr(Annotation.AQRL), Sc(Annotation.AQRL)
                           ]
    while True:
        if len(ppo_queue) == 0:
            break
        ppo, position = ppo_queue.pop(0)
        # print(ppo, position)
        if  ppo.size > depth:
            continue
        if position == ppo.size:
            ppo_list.append(ppo)
            continue
        relation_list = ppo.ppo
        mutate_relation_list = relation_list[position].get_contain_relation()
        mutate_relation_list.append(copy.deepcopy(relation_list[position]))

        for mutate_relation in mutate_relation_list:
            copy_list = deepcopy(relation_list)
            copy_list[position] = mutate_relation
            if mutate_relation.relation_flag:
                if position + 1 >= ppo.size or position - 1 < 0:
                    continue
                pre_relation = copy_list[position - 1]
                suc_relation = copy_list[position + 1]
                if not mutate_relation.check(pre_relation, suc_relation):
                    continue
                # add relation
                if len(copy_list) + 2 <= depth:
                    if type(mutate_relation) == PoLoc:
                        for relation in unary_relation_list:
                            po_loc_list = deepcopy(copy_list)
                            po_loc_list.insert(position, deepcopy(relation))
                            po_loc_list.insert(position, PoLoc())
                            add_ppo = (SinglePPO(po_loc_list), position+1)
                            if add_ppo not in ppo_dict:
                                ppo_dict[add_ppo] = 1
                                ppo_queue.append(add_ppo)

                        if type(suc_relation) == R:
                            rsw_list = deepcopy(copy_list)
                            rsw_list.insert(position, R())
                            rsw_list.insert(position, Rsw())
                            add_ppo = (SinglePPO(rsw_list), position + 1)
                            if add_ppo not in ppo_dict:
                                ppo_dict[add_ppo] = 1
                                ppo_queue.append(add_ppo)

                    if type(mutate_relation) == Po:
                        for relation in unary_relation_list:
                            po_list = deepcopy(copy_list)
                            po_list.insert(position, deepcopy(relation))
                            po_list.insert(position, Po())
                            add_ppo = (SinglePPO(po_list), position + 1)
                            if add_ppo not in ppo_dict:
                                ppo_dict[add_ppo] = 1
                                ppo_queue.append(add_ppo)

                        for relation in unary_relation_list:
                            po_loc_list = deepcopy(copy_list)
                            po_loc_list.insert(position, deepcopy(relation))
                            po_loc_list.insert(position, PoLoc())
                            add_ppo = (SinglePPO(po_loc_list), position + 1)
                            if add_ppo not in ppo_dict:
                                ppo_dict[add_ppo] = 1
                                ppo_queue.append(add_ppo)

                        if type(suc_relation) == R:
                            rsw_list = deepcopy(copy_list)
                            rsw_list.insert(position, R())
                            rsw_list.insert(position, Rsw())
                            add_ppo = (SinglePPO(rsw_list), position + 1)
                            if add_ppo not in ppo_dict:
                                ppo_dict[add_ppo] = 1
                                ppo_queue.append(add_ppo)

                    if type(mutate_relation) == Rsw:
                        rsw_list = deepcopy(copy_list)
                        rsw_list.insert(position, R())
                        rsw_list.insert(position, Rsw())
                        add_ppo = (SinglePPO(rsw_list), position + 1)
                        if add_ppo not in ppo_dict:
                            ppo_dict[add_ppo] = 1
                            ppo_queue.append(add_ppo)

            add_ppo = (SinglePPO(copy_list), position + 1)
            if add_ppo not in ppo_dict:
                ppo_dict[add_ppo] = 1
                ppo_queue.append(add_ppo)



    return ppo_list

def get_ppo_item_by_str(ppo_string, flag = PPOFlag.Strengthen):
    ppo_string = ppo_string.replace('[', '').replace(']', '')
    list = ppo_string.split(';')
    ppo_list = []
    str_to_relation_map = {
        'M': M(),
        'W': W(),
        'R': R(),
        'AMO': AMO(),
        'AQ': AMO(Annotation.AQ),
        'RL': AMO(Annotation.RL),
        'AQRL': AMO(Annotation.AQRL),
        'X': Sc(),
        'XLr':Lr(),
        'XSc':Sc(),
        'Sc': Sc(),
        'Lr': Lr(),
        'po': Po(),
        'po-loc': PoLoc(),
        'fence': FenceD(),
        'fencerel(Fence.w.r)': FenceD(FenceAnnotation.W_R),
        'fencerel(Fence.w.rw)': FenceD(FenceAnnotation.W_RW),
        'fencerel(Fence.w.w)': FenceD(FenceAnnotation.W_W),
        'fencerel(Fence.rw.r)': FenceD(FenceAnnotation.RW_R),
        'fencerel(Fence.rw.w)': FenceD(FenceAnnotation.RW_W),
        'fencerel(Fence.rw.rw)': FenceD(FenceAnnotation.RW_RW),
        'fencerel(Fence.r.r)': FenceD(FenceAnnotation.R_R),
        'fencerel(Fence.r.w)': FenceD(FenceAnnotation.R_W),
        'fencerel(Fence.r.rw)': FenceD(FenceAnnotation.R_RW),
        'fencerel(Fence.tso)': FenceD(FenceAnnotation.TSO),
        'fence_w_r': FenceD(FenceAnnotation.W_R),
        'fence_w_rw': FenceD(FenceAnnotation.W_RW),
        'fence_w_w': FenceD(FenceAnnotation.W_W),
        'fence_rw_r': FenceD(FenceAnnotation.RW_R),
        'fence_rw_w': FenceD(FenceAnnotation.RW_W),
        'fence_rw_rw': FenceD(FenceAnnotation.RW_RW),
        'fence_r_r': FenceD(FenceAnnotation.R_R),
        'fence_r_w': FenceD(FenceAnnotation.R_W),
        'fence_r_rw': FenceD(FenceAnnotation.R_RW),
        'fence_tso': FenceD(FenceAnnotation.TSO),
        'fri': Fri(),
        'fre': Fre(),
        'fr': Fri(),
        'coe': Coe(),
        'coi': Coi(),
        'co': Coi(),
        'rfi': Rfi(),
        'rf': Rfi(),
        'rfe': Rfe(),
        'addr': AddrD(),
        'ctrl': CtrlD(),
        'data': DataD(),
        'rsw': Rsw(),
        'rmw': Rmw()
    }
    for i,item in enumerate(list):
        ppo_list.append(copy.deepcopy(str_to_relation_map[item]))

    return SinglePPO(ppo_list, flag)



if __name__ == '__main__':
    # test extend
    # list = get_ppo_contain_list(SinglePPO([R(),Rsw(),R()]), 7)
    # with open('ppo_extend.txt','w') as f:
    #     list = sorted(list, key=lambda item: (item.size,str(item)))
    #     for item in list:
    #         f.write(str(item))
    #         f.write('\n')

    # test contain
    ppo_1 = SinglePPO([R(),Po(),AMO(Annotation.AQ)])
    ppo_2 = SinglePPO([R(),Po(),AMO(Annotation.RL)])
    print(ppo_1.is_contain(ppo_2))
    print(ppo_2.is_contain(ppo_1))