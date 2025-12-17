import copy
import sys
from typing import List
from enum import Enum

from src.slide.prog import LoadInst, StoreInst, AmoInst, IType, MemoryAccessInst, FenceInst, FenceTsoInst, MoFlag


class MemoryRelation:
    '''
    relation_flag:Marks whether the current relationship is unary(False) or binary(True)
    '''

    def __init__(self, name, relation_flag):
        self.name = name
        self.relation_flag = relation_flag
        self.ppo_name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def check(self, pre, suc):  # check pre and suc relation,only for binary relation
        return True

    def address_expression(self, pre, suc):  # add address expression,only for binary relation
        return None

    def relation_expression(self, pre, suc):  # translate into diyone7 str
        return None

    def is_create_thread(self):  # create new thread
        return False

    def __eq__(self, other):
        return str(self) == str(other)

    def get_ppo_name(self):
        if self.relation_flag:
            return self.ppo_name
        else:
            return f'[{self.ppo_name}]'

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        # check diy7 generate litmus
        return False

    def get_contain_relation(self):
        return []


class Annotation(Enum):
    P = '',
    AQ = '.aq',
    RL = '.rl',
    AQRL = '.aq.rl'

    def __repr__(self):
        return f'{self.value}'


    def __lt__(self, other):
        order = {
            Annotation.P: 0,
            Annotation.AQ: 1,
            Annotation.RL: 1,
            Annotation.AQRL: 2
        }
        return order[self] < order[other]

    def __gt__(self, other):
        order = {
            Annotation.P: 0,
            Annotation.AQ: 1,
            Annotation.RL: 1,
            Annotation.AQRL: 2
        }
        return order[self] > order[other]


def getSuc(relation: MemoryRelation, pre: MemoryRelation, suc: MemoryRelation):  # get suc string
    pre_list = [Po, PoLoc, FenceD, FenceS, Rsw]
    suc_list = [Po, PoLoc, FenceD, FenceS, CtrlD, CtrlS, AddrD, AddrS, DataS, DataD, Rsw]
    pre_annotation_str, suc_annotation_str = '', ''
    pre_str, suc_str = '', ''
    annotation_type_array = [AMO, Sc, Lr]
    # print(relation, pre, suc)
    if type(pre) in annotation_type_array or type(suc) in annotation_type_array:
        pre_annotation_str, suc_annotation_str = 'P', 'P'
        if type(pre) in annotation_type_array:
            if pre.annotation == Annotation.AQRL or pre.annotation == Annotation.RL:
                pre_annotation_str = 'Rl'
            if type(pre) == AMO:
                pre_str = relation_str_map[W]  # W
            else:
                pre_str = relation_str_map[type(pre)]
        else:
            pre_str = relation_str_map[type(pre)]
        if type(suc) in annotation_type_array:
            if suc.annotation == Annotation.AQRL or suc.annotation == Annotation.AQ:
                suc_annotation_str = 'Aq'
            if type(suc) == AMO:
                suc_str = relation_str_map[R]  # R
            else:
                suc_str = relation_str_map[type(suc)]
        else:
            suc_str = relation_str_map[type(suc)]
    else:
        pre_str, suc_str = relation_str_map[type(pre)], relation_str_map[type(suc)]
    if pre_annotation_str == 'P' and suc_annotation_str == 'P':
        pre_annotation_str, suc_annotation_str = '', ''

    if type(relation) not in suc_list:  # remove pre_str
        suc_str = ''
    if type(relation) not in pre_list:
        pre_str = ''
    return pre_str + suc_str + pre_annotation_str + suc_annotation_str


def type_check(obj: MemoryRelation, types: List):  # type(obj) in types
    if type(obj) in types:
        return True
    return False


class M(MemoryRelation):  # only use it in final fuse
    def __init__(self):
        super().__init__('M', False)

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if isinstance(pre, LoadInst) or isinstance(pre, StoreInst):
            return True
        if isinstance(pre, AmoInst):
            return True


class W(MemoryRelation):
    def __init__(self):
        super().__init__('W', False)

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if isinstance(pre, StoreInst):
            return True
        return False


class R(MemoryRelation):
    def __init__(self):
        super().__init__('R', False)

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if isinstance(pre, LoadInst):
            return True
        return False


# FIX ME : AMO inst has 2 variable
class AMO(MemoryRelation):
    def __init__(self, annotation: Annotation = Annotation.P):
        super().__init__('AMO', False)
        self.annotation = annotation
        if annotation == Annotation.P:
            self.ppo_name = 'AMO'
        elif annotation == Annotation.AQ:
            self.ppo_name = 'AQ'
        elif annotation == Annotation.RL:
            self.ppo_name = 'RL'
        else:
            self.ppo_name = 'AQRL'

    def set_annotation(self, annotation: Annotation):  # [AQ,RL,AQRL,]
        self.annotation = annotation

    def __str__(self):
        return self.name + repr(self.annotation)

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if isinstance(pre, AmoInst):
            if pre.type == IType.Amo:
                flag = pre.flag
                if flag == MoFlag.Acquire and self.annotation == Annotation.AQ:
                    return True
                elif flag == MoFlag.Strong and self.annotation == Annotation.AQRL:
                    return True
                elif flag == MoFlag.Release and self.annotation == Annotation.RL:
                    return True
                elif flag == MoFlag.Relax and self.annotation == Annotation.P:
                    return True
        return False


class Sc(MemoryRelation):
    def __init__(self, annotation: Annotation = Annotation.P):
        super().__init__('XSc', False)
        self.annotation = annotation
        self.inst_name = f'sc.w'
        if annotation == Annotation.P:
            self.ppo_name = 'XSc'
        elif annotation == Annotation.AQ:
            self.ppo_name = 'AQ'
            self.inst_name += '.aq'
        elif annotation == Annotation.RL:
            self.ppo_name = 'RL'
            self.inst_name += '.rl'
        else:
            self.ppo_name = 'AQRL'
            self.inst_name += '.aq.rl'

    def set_annotation(self, annotation: Annotation):
        self.annotation = annotation

    def __str__(self):
        return self.name + repr(self.annotation)

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if isinstance(pre, AmoInst):
            if pre.type == IType.Sc:
                return True
            return False


class Lr(MemoryRelation):
    def __init__(self, annotation: Annotation = Annotation.P):
        super().__init__('XLr', False)
        self.annotation = annotation
        self.inst_name = f'lr.w'
        if annotation == Annotation.P:
            self.ppo_name = 'XLr'
        elif annotation == Annotation.AQ:
            self.ppo_name = 'AQ'
            self.inst_name += '.aq'
        elif annotation == Annotation.RL:
            self.ppo_name = 'RL'
            self.inst_name += '.rl'
        else:
            self.ppo_name = 'AQRL'
            self.inst_name += '.aq.rl'
        # Fix : change to more width

    def set_annotation(self, annotation: Annotation):
        self.annotation = annotation

    def __str__(self):
        return self.name + repr(self.annotation)

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if isinstance(pre, AmoInst):
            if pre.type == IType.Lr:
                return True
        return False


class Rmw(MemoryRelation):
    def __init__(self, annotation: Annotation = Annotation.P):
        super().__init__('rmw', True)

    def check(self, pre, suc):
        if type(pre) == Lr and type(suc) == Sc:
            # TODO :process aq.rl
            if pre.annotation == Annotation.P and suc.annotation == Annotation.P:
                return True
        if type(pre) == AmoInst and type(suc) == AmoInst:
            if pre.annotation == suc.annotation:
                return True
        return False

    def address_expression(self, pre, suc):
        return pre == suc

    def relation_expression(self, pre, suc):
        if type(pre) == Lr and type(suc) == Sc:
            return f'{relation_str_map[PoLoc]}{getSuc(PoLoc(), pre, suc)}'
        return f'{relation_str_map[type(self)]}'

    def __gt__(self, other: MemoryRelation):
        return False

    def __ge__(self, other: MemoryRelation):
        if type(other) in [Rmw, PoLoc, Rfi]:
            return True
        return False

    def __lt__(self, other: MemoryRelation):
        if type(other) in [Po]:
            return True
        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if suc is None:
            return False
        # if midden is not None:
        #     return False
        if not isinstance(suc, AmoInst) or not isinstance(pre, AmoInst):
            return False
        if suc.type == IType.Sc and pre.type == IType.Lr:
            if str(pre.rs1) == str(suc.rs1):
                return True
            if (str(pre.rs1), str(suc.rs1)) in equal_regs_list:
                return True

        return False


class Po(MemoryRelation):
    def __init__(self):
        super().__init__('po', True)

    def address_expression(self, pre, suc):
        return pre != suc

    def relation_expression(self, pre, suc):
        return f'{relation_str_map[type(self)]}{getSuc(self, pre, suc)}'

    def __gt__(self, other: MemoryRelation):
        if type(other) in [PoLoc, FenceD, FenceS, CtrlD, CtrlS, AddrD, AddrS, DataD, Coi, Fri, Rfi, Rsw]:
            return True
        return False

    def __ge__(self, other: MemoryRelation):
        if type(other) in [Po, PoLoc, FenceD, FenceS, CtrlD, CtrlS, AddrD, AddrS, DataD, Coi, Fri, Rfi, Rsw, Rmw]:
            return True
        return False

    def __lt__(self, other: MemoryRelation):
        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if suc is None:
            return False
        # if midden is not None:
        #     return False
        if not isinstance(suc, AmoInst) and not isinstance(suc, MemoryAccessInst):
            return False
        if not isinstance(pre, AmoInst) and not isinstance(pre, MemoryAccessInst):
            return False
        if str(pre.rs1) != str(suc.rs1) and (str(pre.rs1), str(suc.rs1)) not in equal_regs_list:
            return True
        return False

    def get_contain_relation(self):
        return copy.deepcopy(
            [PoLoc(), FenceD(), FenceS(), CtrlD(), CtrlS(), AddrD(), AddrS(), DataD(), Coi(), Fri(), Rfi(), Rsw()])


class PoLoc(MemoryRelation):
    def __init__(self):
        super().__init__('po-loc', True)

    def address_expression(self, pre, suc):
        return pre == suc

    def relation_expression(self, pre, suc):
        return f'{relation_str_map[type(self)]}{getSuc(self, pre, suc)}'

    def __gt__(self, other: MemoryRelation):
        # if type(other) in [FenceS, CtrlS, AddrS, DataS, Coi, Fri, Rfi, Rsw]:
        # if type(other) in [Coi, Fri, Rfi, Rsw]: #think these is equal
        # return True
        return False

    def __ge__(self, other: MemoryRelation):
        if type(other) in [PoLoc, Coi, Fri, Rfi, Rsw, Rmw]: #think these is equal
            return True
        return False

    def __lt__(self, other: MemoryRelation):
        if type(other) in [Po]:  # to Fix
            return True
        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if suc is None:
            return False
        # if midden is not None:
        #     return False
        if not isinstance(suc, AmoInst) and not isinstance(suc, MemoryAccessInst):
            return False
        if not isinstance(pre, AmoInst) and not isinstance(pre, MemoryAccessInst):
            return False
        if str(pre.rs1) == str(suc.rs1):
            return True
        # print(equal_regs_list)
        if (str(pre.rs1), str(suc.rs1)) in equal_regs_list:
            return True
        return False


class FenceAnnotation(Enum):
    TSO = 0,
    R_R = 1,
    R_W = 2,
    R_RW = 3,
    W_R = 4,
    W_W = 5,
    W_RW = 6,
    RW_R = 7,
    RW_W = 8,
    RW_RW = 9,


fence_annotation_map = {
    FenceAnnotation.TSO: 'tso',
    FenceAnnotation.R_R: 'r.r',
    FenceAnnotation.R_W: 'r.w',
    FenceAnnotation.R_RW: 'r.rw',
    FenceAnnotation.W_R: 'w.r',
    FenceAnnotation.W_W: 'w.w',
    FenceAnnotation.W_RW: 'w.rw',
    FenceAnnotation.RW_R: 'rw.r',
    FenceAnnotation.RW_W: 'rw.w',
    FenceAnnotation.RW_RW: 'rw.rw',
}


class FenceS(MemoryRelation):
    def __init__(self, annotation=FenceAnnotation.RW_RW):
        super().__init__('fenceS', True)
        self.annotation = annotation
        self.ppo_name = f'fencerel(Fence.{fence_annotation_map[annotation]})'

    def address_expression(self, pre, suc):
        return pre == suc

    def relation_expression(self, pre, suc):
        return f'{relation_str_map[type(self)]}.{fence_annotation_map[self.annotation]}s{getSuc(self, pre, suc)}'

    # to Fix
    def __gt__(self, other: MemoryRelation):
        return False

    def __ge__(self, other: MemoryRelation):
        if type(other) in [FenceS]:
            return True
        return False

    def __lt__(self, other: MemoryRelation):
        if type(other) in [Po, PoLoc]:
            return True
        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if suc is None:
            return False
        if midden is None:
            return False
        if not isinstance(suc, AmoInst) and not isinstance(suc, MemoryAccessInst):
            return False
        if not isinstance(pre, AmoInst) and not isinstance(pre, MemoryAccessInst):
            return False
        if str(pre.rs1) != str(suc.rs1) and (str(pre.rs1), str(suc.rs1)) not in equal_regs_list:
            return False
        # if len(midden) == 0:
        #     return False
        for midden_inst in midden:
            # TODO: add Fence.i
            if isinstance(midden_inst, FenceInst):
                fence_pre, fence_suc = midden_inst.pre, midden_inst.suc
                if fence_annotation_map[self.annotation] == f'{fence_pre}.{fence_suc}':
                    return True
            if isinstance(midden_inst, FenceTsoInst):
                if self.annotation == FenceAnnotation.TSO:
                    return True
        return False


class FenceD(MemoryRelation):
    def __init__(self, annotation=FenceAnnotation.RW_RW):
        super().__init__('fenceD', True)
        self.annotation = annotation
        self.ppo_name = f'fencerel(Fence.{fence_annotation_map[annotation]})'

    def address_expression(self, pre, suc):
        return pre != suc

    def relation_expression(self, pre, suc):
        return f'{relation_str_map[type(self)]}.{fence_annotation_map[self.annotation]}d{getSuc(self, pre, suc)}'

    def __gt__(self, other: MemoryRelation):
        return False

    def __ge__(self, other: MemoryRelation):
        return False

    def __lt__(self, other: MemoryRelation):
        if type(other) in [Po]:
            return True
        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if suc is None:
            return False
        if midden is None:
            return False
        if not isinstance(suc, AmoInst) and not isinstance(suc, MemoryAccessInst):
            return False
        if not isinstance(pre, AmoInst) and not isinstance(pre, MemoryAccessInst):
            return False
        if str(pre.rs1) == str(suc.rs1):
            return False
        if (str(pre.rs1),str(suc.rs1)) in equal_regs_list:
            return False
        # if len(midden) == 0:
        #     return False
        for midden_inst in midden:
            # TODO: add Fence.i
            if isinstance(midden_inst, FenceInst):
                fence_pre, fence_suc = midden_inst.pre, midden_inst.suc
                if fence_annotation_map[self.annotation] == f'{fence_pre}.{fence_suc}':
                    return True
            if isinstance(midden_inst, FenceTsoInst):
                if self.annotation == FenceAnnotation.TSO:
                    return True
        return False


class Fri(MemoryRelation):
    def __init__(self):
        super().__init__('fri', True)
        self.ppo_name = 'fri'

    def check(self, pre, suc):
        return type_check(pre, [R, Lr]) and type_check(suc, [W, Sc])

    def address_expression(self, pre, suc):
        return pre == suc

    def relation_expression(self, pre, suc):
        return f'{relation_str_map[type(self)]}{getSuc(self, pre, suc)}'

    def __gt__(self, other: MemoryRelation):
        return False

    def __ge__(self, other: MemoryRelation):
        if type(other) in [Fri]:
            return True
        return False

    def __lt__(self, other: MemoryRelation):
        # if type(other) in [Po, PoLoc, DataS, AddrS, CtrlS]:
        # return True
        if type(other) in [Po, DataS, AddrS, CtrlS]:
            return True

        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if suc is None:
            return False
        # if midden is not None:
        #     return False
        if not (isinstance(suc, AmoInst) and suc.type == IType.Sc) and not isinstance(suc, StoreInst):
            return False
        if not (isinstance(pre, AmoInst) and pre.type == IType.Lr) and not isinstance(pre, LoadInst):
            return False
        if str(pre.rs1) == str(suc.rs1):
            return True

        if (str(pre.rs1), str(suc.rs1)) in equal_regs_list:
            return True

        return False


class Fre(MemoryRelation):
    def __init__(self):
        super().__init__('fre', True)
        self.ppo_name = 'fre'

    def check(self, pre, suc):
        return type_check(pre, [R, Lr]) and type_check(suc, [W, Sc])

    def address_expression(self, pre, suc):
        return pre == suc

    def is_create_thread(self):
        return True

    def relation_expression(self, pre, suc):
        return f'{relation_str_map[type(self)]}{getSuc(self, pre, suc)}'

    def __gt__(self, other: MemoryRelation):
        return False

    def __ge__(self, other: MemoryRelation):
        return False

    def __lt__(self, other: MemoryRelation):
        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None, same_address=False):
        if suc is None:
            return False
        # if midden is not None:
        #     return False
        if not (isinstance(suc, AmoInst) and suc.type == IType.Sc) and not isinstance(suc, StoreInst):
            return False
        if not (isinstance(pre, AmoInst) and pre.type == IType.Lr) and not isinstance(pre, LoadInst):
            return False
        if same_address:
            return True
        return False


class Coi(MemoryRelation):
    def __init__(self):
        super().__init__('coi', True)
        self.ppo_name = 'coi'

    def check(self, pre, suc):
        return type_check(pre, [W, AMO, Sc]) and type_check(suc, [W, Sc])

    def address_expression(self, pre, suc):
        return pre == suc

    def relation_expression(self, pre, suc):
        return f'{relation_str_map[type(self)]}{getSuc(self, pre, suc)}'

    def __gt__(self, other: MemoryRelation):
        return False

    def __ge__(self, other: MemoryRelation):
        if type(other) in [Coi]:
            return True
        return False

    def __lt__(self, other: MemoryRelation):
        # if type(other) in [Po, PoLoc, DataS, AddrS, CtrlS]:
        #     return True
        if type(other) in [Po, DataS, AddrS, CtrlS]:
            return True
        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if suc is None:
            return False
        # if midden is not None:
        #     return False
        if not (isinstance(suc, AmoInst) and suc.type == IType.Sc) and not isinstance(suc, StoreInst):
            return False
        if not (isinstance(pre, AmoInst) and (pre.type == IType.Sc or pre.type == IType.Amo)) and not isinstance(pre,
                                                                                                                 StoreInst):
            return False
        if str(pre.rs1) == str(suc.rs1):
            return True
        if (str(pre.rs1), str(suc.rs1)) in equal_regs_list:
            return True
        return False


class Coe(MemoryRelation):
    def __init__(self):
        super().__init__('coe', True)
        self.ppo_name = 'coe'

    def check(self, pre, suc):
        return type_check(pre, [W, AMO, Sc]) and type_check(suc, [W, Sc])

    def address_expression(self, pre, suc):
        return pre == suc

    def is_create_thread(self):
        return True

    def relation_expression(self, pre, suc):
        return f'{relation_str_map[type(self)]}{getSuc(self, pre, suc)}'

    def __gt__(self, other: MemoryRelation):
        return False

    def __ge__(self, other: MemoryRelation):
        return False

    def __lt__(self, other: MemoryRelation):
        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None, same_address=False):
        if suc is None:
            return False
        # if midden is not None:
        #     return False
        if not (isinstance(suc, AmoInst) and suc.type == IType.Sc) and not isinstance(suc, StoreInst):
            return False
        if not (isinstance(pre, AmoInst) and (pre.type == IType.Sc or pre.type == IType.Amo)) and not isinstance(pre,
                                                                                                                 StoreInst):
            return False
        if same_address:
            return True
        print('te')
        return False


class Rfi(MemoryRelation):
    def __init__(self):
        super().__init__('rfi', True)

    def check(self, pre, suc):
        return type_check(pre, [W, AMO, Sc]) and type_check(suc, [R, AMO, Lr])

    def address_expression(self, pre, suc):
        return pre == suc

    def relation_expression(self, pre, suc):
        return f'{relation_str_map[type(self)]}{getSuc(self, pre, suc)}'

    def __gt__(self, other: MemoryRelation):
        # if type(other) in [PoLoc]: #to Fix
        # return True
        return False

    def __ge__(self, other: MemoryRelation):
        if type(other) in [Rfi]:
            return True
        return False

    def __lt__(self, other: MemoryRelation):
        # if type(other) in [Po, PoLoc, DataS, AddrS, CtrlS]:
        #     return True
        if type(other) in [Po, DataS, AddrS, CtrlS]:
            return True
        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if suc is None:
            return False
        # if midden is not None:
        #     return False
        if not ((isinstance(suc, AmoInst) and (suc.type == IType.Lr or suc.type == IType.Amo)) or isinstance(suc, LoadInst)):
            return False
        if not ((isinstance(pre, AmoInst) and (pre.type == IType.Sc or pre.type == IType.Amo)) or isinstance(pre, StoreInst)):
            return False
        
        if str(pre.rs1) == str(suc.rs1):
            return True
        if (str(pre.rs1), str(suc.rs1)) in equal_regs_list:
            return True
        return False


class Rfe(MemoryRelation):
    def __init__(self):
        super().__init__('rfe', True)

    def check(self, pre, suc):
        return type_check(pre, [W, AMO, Sc]) and type_check(suc, [R, AMO, Lr])

    def address_expression(self, pre, suc):
        return pre == suc

    def is_create_thread(self):
        return True

    def relation_expression(self, pre, suc):
        return f'{relation_str_map[type(self)]}{getSuc(self, pre, suc)}'

    def __gt__(self, other: MemoryRelation):
        return False

    def __ge__(self, other: MemoryRelation):
        return False

    def __lt__(self, other: MemoryRelation):
        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None, same_address=False):
        if suc is None:
            return False
        # if midden is not None:
        #     return False
        if not (isinstance(suc, AmoInst) and (suc.type == IType.Lr or suc.type == IType.Amo)) and not isinstance(suc,
                                                                                                                 LoadInst):
            return False
        if not (isinstance(pre, AmoInst) and (pre.type == IType.Sc or pre.type == IType.Amo)) and not isinstance(pre,
                                                                                                                 StoreInst):
            return False
        if same_address:
            return True
        return False


# FIX: about dependency, only use midden to check
class AddrD(MemoryRelation):
    def __init__(self):
        super().__init__('addrD', True)
        self.ppo_name = 'addr'

    def check(self, pre, suc):
        return type_check(pre, [R, Lr])

    def relation_expression(self, pre, suc):
        return f'{relation_str_map[type(self)]}{getSuc(self, pre, suc)}'

    def address_expression(self, pre, suc):
        return pre != suc

    def __gt__(self, other: MemoryRelation):
        return False

    def __ge__(self, other: MemoryRelation):
        if type(other) in [AddrD]:
            return True
        return False

    def __lt__(self, other: MemoryRelation):
        if type(other) in [Po]:
            return True
        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if suc is None:
            return False
        if midden is None:
            return False
        if not (isinstance(pre, AmoInst) and pre.type == IType.Lr) and not isinstance(pre, LoadInst):
            return False
        for inst in midden:
            if inst.name == 'add':
                if str(pre.rs1) != str(inst.rs1) and (str(pre.rs1), str(inst.rs1)) not in equal_regs_list:
                    return True
        return False


class AddrS(MemoryRelation):
    def __init__(self):
        super().__init__('addrS', True)
        self.ppo_name = 'addr'

    def check(self, pre, suc):
        return type_check(pre, [R, Lr])

    def relation_expression(self, pre, suc):
        return f'{relation_str_map[type(self)]}{getSuc(self, pre, suc)}'

    def address_expression(self, pre, suc):
        return pre == suc

    def __gt__(self, other: MemoryRelation):
        return False

    def __ge__(self, other: MemoryRelation):
        if type(other) in [AddrS]:
            return True
        return False

    def __lt__(self, other: MemoryRelation):
        if type(other) in [Po, PoLoc]:
            return True
        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if suc is None:
            return False
        if midden is None:
            return False
        if not (isinstance(pre, AmoInst) and pre.type == IType.Lr) and not isinstance(pre, LoadInst):
            return False
        for inst in midden:
            if inst.name == 'add':
                if str(pre.rs1) == str(inst.rs1):
                    return True
                if (str(pre.rs1), str(suc.rs1)) in equal_regs_list:
                    return True
        return False


class CtrlD(MemoryRelation):
    def __init__(self):
        super().__init__('ctrlD', True)
        self.ppo_name = 'ctrl'

    def check(self, pre, suc):
        return type_check(pre, [R, Lr])

    def relation_expression(self, pre, suc):
        return f'{relation_str_map[type(self)]}{getSuc(self, pre, suc)}'

    def address_expression(self, pre, suc):
        return pre != suc

    def __gt__(self, other: MemoryRelation):
        return False

    def __ge__(self, other: MemoryRelation):
        if type(other) in [CtrlD]:
            return True
        return False

    def __lt__(self, other: MemoryRelation):
        if type(other) in [Po]:
            return True
        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if suc is None:
            return False
        if midden is None:
            return False
        if not (isinstance(pre, AmoInst) and pre.type == IType.Lr) and not isinstance(pre, LoadInst):
            return False
        if str(pre.rs1) == str(suc.rs1):
            return False
        if (str(pre.rs1), str(suc.rs1)) in equal_regs_list:
            return False
        for inst in midden:
            if inst.name == 'bne':
                return True
        return False


class CtrlS(MemoryRelation):
    def __init__(self):
        super().__init__('ctrlS', True)
        self.ppo_name = 'ctrl'

    def check(self, pre, suc):
        return type_check(pre, [R, Lr])

    def relation_expression(self, pre, suc):
        return f'{relation_str_map[type(self)]}{getSuc(self, pre, suc)}'

    def address_expression(self, pre, suc):
        return pre == suc

    def __gt__(self, other: MemoryRelation):
        return False

    def __ge__(self, other: MemoryRelation):
        if type(other) in [CtrlS]:
            return True
        return False

    def __lt__(self, other: MemoryRelation):
        if type(other) in [Po, PoLoc]:
            return True
        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if suc is None:
            return False
        if midden is None:
            return False
        if not (isinstance(pre, AmoInst) and pre.type == IType.Lr) and not isinstance(pre, LoadInst):
            return False
        if str(pre.rs1) != str(suc.rs1) and (str(pre.rs1), str(suc.rs1)) not in equal_regs_list:
            return False

        for inst in midden:
            if inst.name == 'bne':
                return True


        return False


class DataD(MemoryRelation):
    def __init__(self):
        super().__init__('dataD', True)
        self.ppo_name = 'data'

    def check(self, pre, suc):
        return type_check(pre, [R, Lr]) and type_check(suc, [W, Sc])

    def relation_expression(self, pre, suc):
        return f'{relation_str_map[type(self)]}{getSuc(self, pre, suc)}'

    def address_expression(self, pre, suc):
        return pre != suc

    def __gt__(self, other: MemoryRelation):
        return False

    def __ge__(self, other: MemoryRelation):
        if type(other) in [DataD]:
            return True
        return False

    def __lt__(self, other: MemoryRelation):
        if type(other) in [Po]:
            return True
        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if suc is None:
            return False
        if midden is None:
            return False
        if not (isinstance(pre, AmoInst) and pre.type == IType.Lr) and not isinstance(pre, LoadInst):
            return False
        if not isinstance(suc, StoreInst) and not (isinstance(suc, StoreInst) and suc.type == IType.Sc):
            return False
        if str(pre.rs1) == str(suc.rs1):
            return False
        if (str(pre.rs1), str(suc.rs1)) in equal_regs_list:
            return False
        for inst in midden:
            if inst.name == 'ori':
                if str(inst.rd) == str(inst.rs1):
                    return True
        return False


class DataS(MemoryRelation):
    def __init__(self):
        super().__init__('dataS', True)
        self.ppo_name = 'data'

    def check(self, pre, suc):
        return type_check(pre, [R, Lr]) and type_check(suc, [W, Sc])

    def relation_expression(self, pre, suc):
        return f'{relation_str_map[type(self)]}{getSuc(self, pre, suc)}'

    def address_expression(self, pre, suc):
        return pre == suc

    def __gt__(self, other: MemoryRelation):
        return False

    def __ge__(self, other: MemoryRelation):
        if type(other) in [CtrlS]:
            return True
        return False

    def __lt__(self, other: MemoryRelation):
        if type(other) in [Po, PoLoc]:
            return True
        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if suc is None:
            return False
        if midden is None:
            return False
        if not (isinstance(pre, AmoInst) and pre.type == IType.Lr) and not isinstance(pre, LoadInst):
            return False
        if not isinstance(suc, StoreInst) and not (isinstance(suc, StoreInst) and suc.type == IType.Sc):
            return False
        if str(pre.rs1) != str(suc.rs1) and (str(pre.rs1), str(suc.rs1)) not in equal_regs_list:
            return False

        for inst in midden:
            if inst.name == 'ori':
                if str(inst.rd) == str(inst.rs1):
                    return True
        return False


class Rsw(MemoryRelation):
    def __init__(self):
        super().__init__('rsw', True)

    def check(self, pre, suc):
        return type_check(pre, [R]) and type_check(suc, [R])

    def relation_expression(self, pre, suc):
        return f'{relation_str_map[type(self)]}{getSuc(self, pre, suc)}'

    def address_expression(self, pre, suc):
        return pre == suc

    def __gt__(self, other: MemoryRelation):
        return False

    def __ge__(self, other: MemoryRelation):
        if type(other) in [Rsw]:
            return True
        return False

    def __lt__(self, other: MemoryRelation):
        if type(other) in [Po, PoLoc]:
            return True
        return False

    def static_check(self, pre, suc=None, midden=None, equal_regs_list=None):
        if suc is None:
            return False
        # if midden is not None:
        #     return False
        if not (isinstance(suc, AmoInst) and (suc.type == IType.Lr or suc.type == IType.Amo)) and not isinstance(suc,
                                                                                                                 LoadInst):
            return False
        if not (isinstance(pre, AmoInst) and pre.type == IType.Lr) and not isinstance(pre, LoadInst):
            return False
        if str(pre.rs1) == str(suc.rs1):
            return True
        if (str(pre.rs1), str(suc.rs1)) in equal_regs_list:
            return True
        return False


unary_pre_relation_map = {  # out thread
    W: [Coe, Fre],
    R: [Rfe],
    AMO: [Rfe],
    Sc: [Coe, Fre],
    Lr: [Rfe],

}

unary_suc_relation_map = {  # out thread
    W: [Coe, Rfe],
    R: [Fre],
    AMO: [Coe, Rfe],
    Sc: [Coe, Rfe],
    Lr: [Fre]
}

# jump to other thread
jump_relation_list = [Coe, Rfe, Fre]

# ppo can add relation map, only for init
ppo_can_add_relation_map = {
    W: [Coi, Rfi, FenceD, FenceS, Po, PoLoc, Coe, Rfe],
    Sc: [Coi, Rfi, FenceD, FenceS, Po, PoLoc, Coe, Rfe],
    AMO: [Coi, Rfi, FenceD, FenceS, Po, PoLoc, Coe, Rfe],
    R: [Fri, FenceD, FenceS, Po, PoLoc, AddrS, AddrD, DataD, DataS, CtrlD, CtrlS, Fre],
    Lr: [Fri, FenceD, FenceS, Po, PoLoc, AddrS, AddrD, DataD, DataS, CtrlD, CtrlS, Rmw, Fre],
    Fri: [W, Sc],
    Fre: [W, Sc],
    Coi: [W, Sc],
    Coe: [W, Sc],
    Rfi: [R, Lr, AMO],
    Rfe: [R, Lr, AMO],
    FenceD: [W, R, Sc, Lr, AMO],
    FenceS: [W, R, Sc, Lr, AMO],
    Po: [W, R, Sc, Lr, AMO],
    PoLoc: [W, R, Sc, Lr, AMO],
    AddrD: [W, R, Sc, Lr, AMO],
    AddrS: [W, R, Sc, Lr, AMO],
    DataD: [W, R, Sc, Lr, AMO],
    DataS: [W, R, Sc, Lr, AMO],
    CtrlD: [W, R, Sc, Lr, AMO],
    CtrlS: [W, R, Sc, Lr, AMO],
    Rmw: [Sc],
}
# ppo add relation map
ppo_add_relation_map = {
    W: [Coi, Rfi, FenceD, FenceS, Po, PoLoc],
    Sc: [Coi, Rfi, FenceD, FenceS, Po, PoLoc],
    AMO: [Coi, Rfi, FenceD, FenceS, Po, PoLoc],
    R: [Fri, FenceD, FenceS, Po, PoLoc, AddrS, AddrD, DataD, DataS, CtrlD, CtrlS],
    Lr: [Fri, FenceD, FenceS, Po, PoLoc, AddrS, AddrD, DataD, DataS, CtrlD, CtrlS, Rmw],
    Fri: [W],
    Fre: [W],
    Coi: [W],
    Coe: [W],
    Rfi: [R],
    Rfe: [R],
    FenceD: [W, R],
    FenceS: [W, R],
    Po: [W, R],
    PoLoc: [W, R],
    AddrD: [W, R],
    AddrS: [W, R],
    DataD: [W, R],
    DataS: [W, R],
    CtrlD: [W, R],
    CtrlS: [W, R],
    Rmw: [Sc],
}

other_add_relation_map = {
    W: [FenceD, FenceS],
    R: [FenceD, FenceS],
    Fri: [W],
    Fre: [W],
    Coi: [W],
    Coe: [W],
    Rfi: [R],
    Rfe: [R],
    FenceD: [W, R],
    FenceS: [W, R],
    Po: [W, R],
    PoLoc: [W, R],
    AddrD: [W, R],
    AddrS: [W, R],
    DataD: [W, R],
    DataS: [W, R],
    CtrlD: [W, R],
    CtrlS: [W, R]
}

relation_str_map = {
    Fri: 'Fri',
    Fre: 'Fre',
    Coi: 'Coi',
    Coe: 'Coe',
    Rfi: 'Rfi',
    Rfe: 'Rfe',
    FenceD: 'Fence',
    FenceS: 'Fence',
    Po: 'Pod',
    PoLoc: 'Pos',
    W: 'W',
    R: 'R',
    AddrD: 'DpAddrd',
    AddrS: 'DpAddrs',
    DataD: 'DpDatad',
    DataS: 'DpDatas',
    CtrlD: 'DpCtrld',
    CtrlS: 'DpCtrls',
    Sc: 'W',
    Lr: 'R',
    Rsw: 'Pos',
    Rmw: 'Rmw',
}


def fuse_w_and_r_to_m(relation_a, relation_b):
    # print(relation_a, relation_b)
    if len(relation_b) != 1 and len(relation_a) != 1:
        return False, None
    relation_a = relation_a[0]
    relation_b = relation_b[0]
    if type(relation_a) == W and type(relation_b) == R:
        return True, M()
    elif type(relation_a) == R and type(relation_b) == W:
        return True, M()
    else:
        return False, None


can_fuse_relation = [
    fuse_w_and_r_to_m,
]