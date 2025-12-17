from typing import List

import networkx as nx
from egglog.bindings import Relation

from src.slide.analysis import Event, GlobalRelationAnalyzer
import toolz as tz

from src.slide.prog import IType, MoFlag
from src.slide.synth import transform
from src.slide.synth.diy7_generator import Cycle
from src.slide.synth.memory_relation import *
from src.slide.synth.ppo_def import SinglePPO, get_ppo_item_by_str
from src.slide.utils.ppo.ppo_parser import parse_to_gnode_tree


def event_to_rel_relation(relation: str):

    def get_fence_rw_rw_relation():
        return FenceD(annotation=FenceAnnotation.RW_RW)

    def get_fence_rw_r_relation():
        return FenceD(annotation=FenceAnnotation.RW_R)

    def get_fence_rw_w_relation():
        return FenceD(annotation=FenceAnnotation.RW_W)

    def get_fence_w_rw_relation():
        return FenceD(annotation=FenceAnnotation.W_RW)

    def get_fence_w_r_relation():
        return FenceD(annotation=FenceAnnotation.W_R)

    def get_fence_w_w_relation():
        return FenceD(annotation=FenceAnnotation.W_W)

    def get_fence_r_rw_relation():
        return FenceD(annotation=FenceAnnotation.R_RW)

    def get_fence_r_r_relation():
        return FenceD(annotation=FenceAnnotation.R_R)

    def get_fence_r_w_relation():
        return FenceD(annotation=FenceAnnotation.R_W)

    def get_fence_tso_relation():
        return FenceD(annotation=FenceAnnotation.TSO)

    relation_dict = {
        'rmw' : Rmw,
        'fence': FenceD,
        'rfi': Rfi,
        'rfe': Rfe,
        'rsw': Rsw,
        'coi': Coi,
        'coe': Coe,
        'fri': Fri,
        'fre': Fre,
        'po_loc': PoLoc,
        'addr' : AddrD,
        'data' : DataD,
        'ctrl' : CtrlD,
        'po' : Po,
        'fence_rw_rw': get_fence_rw_rw_relation,
        'fence_rw_w': get_fence_rw_w_relation,
        'fence_rw_r': get_fence_rw_r_relation,
        'fence_w_rw': get_fence_w_rw_relation,
        'fence_w_r': get_fence_w_r_relation,
        'fence_w_w': get_fence_w_w_relation,
        'fence_r_rw': get_fence_r_rw_relation,
        'fence_r_r': get_fence_r_r_relation,
        'fence_r_w': get_fence_r_w_relation,
        'fence_tso': get_fence_tso_relation,
    }
    return relation_dict[relation]()

def event_to_mem_relation(event: Event):
    inst_type = event.inst.type
    if inst_type == IType.Load:
        return R()
    elif inst_type == IType.Store:
        return W()
    else:
        flag_dict={
            MoFlag.Acquire: Annotation.AQ,
            MoFlag.Release: Annotation.RL,
            MoFlag.Strong: Annotation.AQRL,
            MoFlag.Relax: Annotation.P,
            }
        annotation_type = flag_dict[event.inst.inst.flag]

        if inst_type == IType.Amo:
            return AMO(annotation_type)
        elif inst_type == IType.Sc:
            return Sc(annotation_type)
        elif inst_type == IType.Lr:
            return Lr(annotation_type)

    assert False, f'this {event} not match mem relation'



def path_to_ppo(path, start_relation = None, end_relation = None) -> SinglePPO:
    relation_list = []
    if start_relation is not None:
        relation_list.append(event_to_rel_relation(start_relation))

    for i,(ei, ej, relation) in enumerate(path):
        mem_relation = event_to_mem_relation(ei)
        rel_relation = event_to_rel_relation(relation)
        relation_list.append(mem_relation)
        relation_list.append(rel_relation)

        if i == len(path) - 1:
            mem_relation = event_to_mem_relation(ej)
            relation_list.append(mem_relation)

    if end_relation is not None:
        relation_list.append(event_to_rel_relation(end_relation))

    return SinglePPO(relation_list)

print_flag = True
def get_violate_func_list(ra, func_list): #func_list = List[(func_name,func)]

    violate_func_list = []
    #1. Find all PPOs contained in the exe.
    rel_to_funcname_dict = {}
    for func_name, _, func_string, _ in func_list:
        exec(func_string,globals())
        func = globals()[func_name]
        func_rels = ra.find_all_by_func(func)
        for rel in func_rels:
            rel_to_funcname_dict.setdefault((ra.execution.index(rel[0]), ra.execution.index(rel[1])), []).append(
                func_name)

    for key in rel_to_funcname_dict:
        print(key, rel_to_funcname_dict[key])

    # 2. Extract the cycles formed by the EXE files.
    rels_list = ['co', 'rfe', 'fr', 'ppo']
    all_rels = list(tz.concat([ra.find_all(r) for r in rels_list]))
    all_rels_idx = list(set([(ra.execution.index(r[0]), ra.execution.index(r[1])) for r in all_rels]))
    print('all_rels_idx', all_rels_idx)

    if len(all_rels_idx) == 0:
        return []
    graph = nx.DiGraph()
    graph.add_edges_from(all_rels_idx)
    cycles = list(nx.simple_cycles(graph))
    print(len(cycles))

    # 3. Traverse the cycles. If the rel corresponding to the ppo is in the cycle, add it to violate_func_list.
    for cycle in cycles:
        print(cycle)
        for i in range(len(cycle)):
            j = i+1 if i+1 < len(cycle) else 0
            print(cycle[i],cycle[j])
            if (cycle[i],cycle[j]) in all_rels_idx and (cycle[i],cycle[j]) in rel_to_funcname_dict:
                print(f'add {rel_to_funcname_dict[(cycle[i],cycle[j])]} to violate list')
                violate_func_list.append(rel_to_funcname_dict[(cycle[i],cycle[j])])

    return violate_func_list


class Constraint:

    def get_all_paths_from_e1_to_e2(self, e1, e2):
        middle_events = [e for e in self.exe if (self.ra.po(e1, e) and self.ra.po(e, e2)) or
                         (self.ra.AMO(e) and ((e.inst == e1.inst and self.ra.W(e) != self.ra.W(e1))
                                              or (e.inst == e2.inst and self.ra.W(e) != self.ra.W(e2))))]  # fix: about amo
        middle_events = sorted(middle_events, key=lambda e: e.inst.pc)
        events = [e1] + middle_events + [e2]
        # amo event must make pair
        amo_dict = {}
        if print_flag:
            print('events sequence',)
            for e in events:
                print(f'{e}, is write event:{self.ra.W(e)}')

        for e in events:
            if self.ra.AMO(e):
                if str(e) in amo_dict:
                    amo_dict[str(e)] = 1
                else:
                    amo_dict[str(e)] = 0
        for key in amo_dict:
            if amo_dict[key] == 0:
                return []

        order = {}
        for i, ei in enumerate(events):
            order[f'{str(ei)} {self.ra.W(ei)}'] = i

        mg = nx.MultiGraph()

        for i, ei in enumerate(events):
            for j, ej in enumerate(events[i + 1:]):
                relations = self.get_relations_between_two_events(ei, ej)

                # process amo
                if self.ra.AMO(ej):
                    if self.ra.W(ej) and ei.inst != ej.inst:
                        continue
                if self.ra.AMO(ei):
                    if self.ra.R(ei) and ei.inst != ej.inst:
                        continue

                if print_flag:
                    ei_w_type = 'W' if self.ra.W(ei)else 'R'
                    ej_w_type = 'W' if self.ra.W(ej)else 'R'
                    print(f'ei,ej,relations, {ei}({ei_w_type}), {ej}({ej_w_type}), {relations}')

                for relation in relations:
                    mg.add_edge(self.exe.index(ei), self.exe.index(ej), relation)

        paths = list(nx.all_simple_edge_paths(mg, self.exe.index(e1), self.exe.index(e2)))
        paths = [[(self.exe[ei], self.exe[ej], relation) for ei, ej, relation in path] for path in paths]
        result = []
        for path in paths:
            amo_dict = {}
            flag = True
            for ei, ej, relation in path:
                if order[f'{str(ei)} {self.ra.W(ei)}'] > order[f'{str(ei)} {self.ra.W(ei)}']:
                    flag = False
                    # For an amo instruction, both events are required to be in the path


            # check amo is pair
            for i,(ei, ej, relation) in enumerate(path):
                if self.ra.AMO(ei):
                    if str(ei) in amo_dict:
                        amo_dict[str(ei)] = 1
                    else:
                        amo_dict[str(ei)] = 0
                if i == len(path)-1:
                    if self.ra.AMO(ej):
                        if str(ej) in amo_dict:
                            amo_dict[str(ej)] = 1
                        else:
                            amo_dict[str(ej)] = 0

            for key in amo_dict:
                if amo_dict[key] == 0:
                    flag = False
                    break

            if print_flag:
                print(f'this path is {flag}')
                for item in path:
                    print(item)
            if flag:
                result.append(path)

        return result

    def get_path_from_e1_to_e2_by_ppo(self, e1, e2, ppo):
        if print_flag:
            print('enter get_path_from_e1_to_e2_by_ppo')
        if e1.pid != e2.pid:
            return None
        paths = self.get_all_paths_from_e1_to_e2(e1, e2)
        if print_flag:
            print('get_all_paths_from_e1_to_e2',paths)
        for path in paths:
            if not path:
                continue
            need_check_ppo = path_to_ppo(path)
            if print_flag:
                print('need_check_ppo',need_check_ppo,'target_ppo',ppo)
            if need_check_ppo == ppo:
                return path
        if print_flag:
            print('exit get_path_from_e1_to_e2_by_ppo')
        return None # Note It is not in po order


    def __init__(self, e1: Event, e2: Event, ra: GlobalRelationAnalyzer, exe: List[Event]):
        self.e1, self.e2, self.ra, self.exe = e1, e2, ra, exe

        self.mem_type_dict = {  # 'M': ra.M,
            'R': ra.R,
            'W': ra.W,
            'AMO': ra.AMO,
            'X': ra.X,
            'XSc': ra.XSc,
            'XLr': ra.XLr,
            'RL': ra.RL,
            'AQ': ra.AQ,
            'AQRL': ra.AQRL
        }
        self.relation_dict = {'rmw': ra.rmw,
                              'addr': ra.addr,
                              'data': ra.data,
                              'ctrl': ra.ctrl,
                              'fence': ra.fence,
                              'po': ra.po,
                              'po_loc': ra.po_loc,
                              'rfi': ra.rfi,
                              'rfe': ra.rfe,
                              'rsw': ra.rsw,
                              'coi': ra.coi,
                              'coe': ra.coe,
                              'fri': ra.fri,
                              'fre': ra.fre,
                              'fence_rw_rw': ra.fence_rw_rw,
                              'fence_rw_r': ra.fence_rw_r,
                              'fence_rw_w': ra.fence_rw_w,
                              'fence_r_rw': ra.fence_r_rw,
                              'fence_r_r': ra.fence_r_r,
                              'fence_r_w': ra.fence_r_w,
                              'fence_w_rw': ra.fence_w_rw,
                              'fence_w_r': ra.fence_w_r,
                              'fence_w_w': ra.fence_w_w,
                              'fence_tso': ra.fence_tso,
                              }

        self.relations = []

        if print_flag:
            print('--------------create constraint-------------------------')
            print('use constraint extract ppo from e1,e2',self.e1, self.e2)

        # get all paths (a list of edges) from e1 to e2
        self.paths = self.get_all_paths_from_e1_to_e2(self.e1, self.e2)
        if print_flag:
            print('now check the paths', self.e1, self.e2)
            for path in self.paths:
                print(path)

        # get PPO candidates
        self.candidate_ppos = []
        self.diy_cycles = []
        for path in self.paths:
            pass_lr_sc_flag = True
            # 1) get ppo string
            ppo = ''
            exe_string = ''
            if print_flag:
                print('from this path get ppo')
            for index,(ei, ej, relation) in enumerate(path):
                exe_string += f'P{ei.pid}: <0x{"{:0>2X}".format(ei.inst.pc)}> '
                if print_flag:
                    print(ei, ej, self.ra.W(ei), self.ra.W(ej))
                mem_types_i = self.get_mem_types(ei)
                # specify a memtype
                assert len(mem_types_i) > 0, "len(mem_types_i) must > 0"
                mem_type_i = mem_types_i[-1] # get strict relation
                ppo += f'{mem_type_i};{relation};'
                # process lr and sc, only rmw relation can pass
            #     if self.ra.X(ei) and index==0 and ei.inst.inst.type == IType.Sc:
            #         pass_lr_sc_flag = False
            #         break
            #     # print('ei type',ei.inst.inst, type(ei.inst.inst))
            #     # if isinstance(ei.inst.inst, AmoInst):
            #     #     if ei.inst.inst.type == IType.Lr:
            #     #         # Lr can appear on its own
            #     #         if not self.ra.rmw(ei,ej) or relation != 'rmw':
            #     #             pass_lr_sc_flag = False
            #     #             break
                if self.ra.X(ej) and ej.inst.inst.type == IType.Sc:
                    if self.ra.X(ei) and self.ra.rmw(ei,ej):
                        if not relation!= 'rmw':
                            pass_lr_sc_flag = False
                            break
                    if relation not in ['coi', 'po', 'po_loc', 'fri']:
                        pass_lr_sc_flag = False
                        break
            #
            if not pass_lr_sc_flag:
                continue
            exe_string += f'P{path[-1][1].pid}: <0x{"{:0>2X}".format(path[-1][1].inst.pc)}>'
            mem_types_j = self.get_mem_types(path[-1][1])
            assert len(mem_types_j) > 0, "len(mem_types_j) must > 0"
            mem_type_j = mem_types_j[-1]
            ppo += mem_type_j
            ppo = ppo.replace('po_loc','po-loc')

            if print_flag:
                print('candidate_add_ppo', ppo)

            # 2) for each ppo, get it's corresponding cycle for acyclic('co', 'rfe', 'fr', 'ppo')
            all_rels = list(tz.concat([ra.find_all(r) for r in ['coi','coe', 'rfe', 'fri','fre', 'ppo']]))
            # all_rels.append((e1, e2))  # add e1, e2 that has ppo now.

            if print_flag:
                print('all_rels')
                for rel in all_rels:
                    ei, ej = rel
                    print('rel', ei, ej, self.ra.W(ei), ra.W(ej), self.get_relations_between_two_events(ei, ej))


            # 2.1) For this new ppo, we need to add all the event pairs that have their relationships.
            ppo_index = -1
            # ppo_to_func(ppo,ppo_index)
            root = parse_to_gnode_tree(ppo)
            # update the string ppo to the SinglePPO
            ppo = get_ppo_item_by_str(ppo)
            python_func_string = transform.transform(root, ppo_index=ppo_index)
            print('python_func_string', python_func_string)
            exec(python_func_string, globals())
            ppo_func = globals()[f'ppo_candidate_func{ppo_index}'.replace('-1','__1')]
            add_rels = ra.find_all_by_func(ppo_func)

            if print_flag:
                print('add_rels')
                for rel in add_rels:
                    ei, ej = rel
                    print('add rel', ei, ej, self.ra.W(ei), ra.W(ej), self.get_relations_between_two_events(ei, ej))

            add_rels_final = []
            for rel in add_rels:
                ei, ej = rel
                ppo_path = self.get_path_from_e1_to_e2_by_ppo(ei, ej, ppo)
                if ppo_path is not None:
                    add_rels_final.append(rel)
            
            add_rels_ids = [(self.exe.index(ei), self.exe.index(ej)) for ei, ej in add_rels_final ]

            all_rels.extend(add_rels_final)



            if print_flag:
                print('add_rels_final')
                for rel in add_rels_final:
                    ei, ej = rel
                    print('add rel final', ei, ej, self.ra.W(ei), ra.W(ej), self.get_relations_between_two_events(ei, ej))


            if print_flag:
                print('check all rels')
                for rel in all_rels:
                    ei, ej = rel
                    print('check rel', ei, ej, self.ra.W(ei), ra.W(ej),
                          self.get_relations_between_two_events(ei, ej))
            graph = nx.DiGraph()
            graph.add_edges_from(all_rels)
            cycles = list(nx.simple_cycles(graph))

            if len(cycles) == 0:
                continue

            

            # 2.3) insert event which in ppo
            new_cycles = []
            for cycle in cycles:
                new_cycle = []
                for i, ei in enumerate(cycle):
                    new_cycle.append(ei)
                    j = i+1
                    if j >= len(cycle):
                        j =0
                    ej = cycle[j]
                    ei_id, ej_id = self.exe.index(ei), self.exe.index(ej)

                    if (ei_id, ej_id) in add_rels_ids:
                        print('need insert')
                        insert_path = self.get_path_from_e1_to_e2_by_ppo(ei, ej, ppo)
                        print('insert_path', insert_path)
                        insert_event_list = []
                        for insert_ei, insert_ej, relation in insert_path:
                            if self.exe.index(insert_ei) == self.exe.index(ei):
                                continue
                            insert_event_list.append(insert_ei)
                        new_cycle.extend(insert_event_list)
                new_cycles.append(new_cycle)
            cycles = new_cycles

            # 2.4) check amo is pair

            cycle_list = []
            for cycle in cycles:
                if len(cycle)==2:  # remove only e1,e2 cycle
                    continue
                amo_map={}
                for item in cycle:
                    if ra.AMO(item):
                        if item.inst in amo_map:
                            amo_map[item.inst]=True
                        else:
                            amo_map[item.inst]=False
                print(amo_map)
                for key,value in amo_map.items():
                    if not value:
                        continue
                # check e1,e2 in cycle
                if e1 in cycle and e2 in cycle:
                    cycle_list.append(cycle)

            if print_flag:
                print('filter_cycle', cycle_list)

            if len(cycle_list) == 0:
                continue

            # 2.3) get first cycle
            cycle = cycle_list[0]

            if print_flag:
                print('ppo',ppo)
                print('e1',e1)
                print('e2',e2)
                print('cycle str')
                for item in cycle:
                    print(item)

            # 3) get diy cycle

            diy_cycle = self.get_diy_cycle(cycle, e1, e2, path, ra, mutate=False)

            if diy_cycle is None:
                continue


            # ppo = ppo.replace('po_loc','po-loc')#change
            # justify ppo (for amo)
            ppo.justify()
            self.candidate_ppos.append(str(ppo))
            print('diy_cycle', diy_cycle)
            self.diy_cycles.append([diy_cycle])


    def get_diy_cycle(self, cycle, e1, e2, path, ra, mutate):
        cycle_list=[] #print temp

        # 1. get e1,e2 index
        while True:
            if self.exe.index(cycle[0]) == self.exe.index(e1):
                break
            item = cycle.pop(0)
            cycle.append(item)
            if print_flag:
                print('cycle str after change')
                for item in cycle:
                    print(item)
                    
        assert self.exe.index(cycle[0]) == self.exe.index(e1), 'e1 index must is 0'
        

        # 2. get other path
        e2_index = -1
        for i, event in enumerate(cycle):
            if self.exe.index(event) == self.exe.index(e2):
                e2_index = i
        assert e2_index!=-1, 'e2 must in cycle'

        other_path = []
        other_cycle = cycle[e2_index+1:]
        for i, em in enumerate(other_cycle):
            if i == len(other_cycle)-1:
                continue
            en = other_cycle[i+1]
            relations = self.get_relations_between_two_events(em, en)
            other_path.append((em, en, relations[0]))

        end_relation = self.get_relations_between_two_events(other_cycle[-1], e1)[0]
        start_relation = self.get_relations_between_two_events(e2, other_cycle[0])[0]
        print('start_relation', start_relation)
        print('end relation', end_relation)
        path_single_ppo = path_to_ppo(path)
        other_single_ppo = path_to_ppo(other_path, start_relation, end_relation)
        print('path_single_ppo', path_single_ppo)
        print('other_single_ppo', other_single_ppo)
        path_single_ppo.justify()
        other_single_ppo.justify()
        print('path_single_ppo justify', path_single_ppo)
        print('other_single_ppo justify', other_single_ppo)

        cycle = Cycle(path_single_ppo, other_single_ppo)

        return cycle.to_diy_format()

    def get_relations_between_two_events(self, ei: Event, ej: Event):
        return [relation for relation, func in self.relation_dict.items() if func(ei, ej)]

    def get_mem_types(self, event: Event):
        return [mem_type for mem_type, func in self.mem_type_dict.items() if func(event)]










