import time

from itertools import cycle

from src.slide.litmus import parse_litmus
from src.slide.synth.ppo_def import *

from z3 import Solver, Int, z3

from src.slide.utils.file_util import search_file, read_file
from src.slide.utils import cmd_util, file_util, dir_util, time_util
from src.slide import config
from src.slide.utils.herd_util import *


import os

class Cycle:
    def __init__(self, ppo :SinglePPO, other :SinglePPO):
        # other indicates other parts of the ring
        self.ppo = ppo
        self.other = other


    def __repr__(self):
        cycle = self.get_cycle()
        return ';'.join([item.get_ppo_name() for item in cycle])

    def get_cycle(self):
        # last relation is connect operator
        cycle = self.ppo.ppo + self.other.ppo
        while(True):
            item = cycle[len(cycle)-1]
            if type(item) == Rfe or type(item) == Fre or type(item) == Coe:
                break
            pop_item = cycle.pop()
            cycle.insert( 0,pop_item)
        return cycle

    def get_thread_ppos(self):
        # return list which has every thread's ppo and their connect relation
        thread_list = []
        connect_list = []
        cycle = self.get_cycle()
        for i, item in enumerate(cycle):
            if type(item) == Rfe or type(item) == Fre or type(item) == Coe:
                connect_list.append(i)

        for i, connect_item in enumerate(connect_list):
            if i == len(connect_list) - 1:
                thread_list.insert(0,cycle[:connect_list[0]])
            else:
                thread_list.append(cycle[connect_list[i]+1:connect_list[i+1]])
        connect_list = [cycle[item] for item in connect_list]
        return thread_list, connect_list

    def check(self):
        legal = True
        filter = [
            self.check_cycle_legal,
            self.check_cycle_diy_one7_rule
        ]
        for item in filter:
            if not item():
                legal = False
                break

        return legal

    def check_cycle_legal(self):
        cycle = self.get_cycle()
        print('check legal',cycle)
        for i, relation in enumerate(cycle):
            # if i == len(cycle) - 1:
            #     continue
            pre = i - 1 if i != 0 else len(cycle) - 1
            suc = i + 1 if i != len(cycle) - 1 else 0

            assert cycle[suc].relation_flag != relation.relation_flag, 'Two neighboring relationships cannot be the same'

            if relation.relation_flag:
                if not relation.check(cycle[pre], cycle[suc]):
                    return False
        print('check legal pass',cycle)
        return True

    def check_cycle_diy_one7_rule(self):

        cycle = self.get_cycle()

        solver = Solver()
        #1. init var
        var_list = []
        for i, relation in enumerate(cycle):
            if relation.relation_flag:
                var_list.append(None)
            else:
                var_list.append(Int('Relation' + str(i)))


        #2. check whether meet the diyone7 rule
        #2.1 add loc expression
        for i, relation in enumerate(cycle):
            if relation.relation_flag:
                pre = i - 1 if i != 0 else len(cycle) - 1
                suc = i + 1 if i != len(cycle) - 1 else 0
                condition = relation.address_expression(var_list[pre], var_list[suc])
                # print(condition)
                if condition is not None:
                    solver.add(condition)


        #2.2 solve
        result = solver.check()
        if result == z3.sat:
            # print('success')
            model = solver.model()
            # print('model', model)
            value_dict = {}
            relation_list = model.decls()

            #2.2.1 check unique location
            if relation_list is not None and len(relation_list) != 0:
                first = relation_list[0]
                first_value = int(str(model[first]))
                unique_location_flag = True
                for relation in relation_list:
                    if int(str(model[relation])) != first_value:
                        unique_location_flag = False
                        break
                if unique_location_flag:
                    return False
            return True
        else:
            # print('fail')
            return False

    def to_diy_format(self):

        cycle = self.get_cycle()

        translate_list=[]

        def get_cycle_item(relation, pre, suc):
            return relation.relation_expression(pre, suc)

        #translate all relation
        for i,relation in enumerate(cycle):
            pre = i - 1 if i != 0 else len(cycle) - 1
            pre = cycle[pre]
            suc = i + 1 if i != len(cycle) - 1 else 0
            suc = cycle[suc]
            if relation.relation_flag:  # relation
                translate_list.append(get_cycle_item(relation, pre, suc))
            if type(relation) == AMO:  # special
                translate_list.append('Rmw')

        translate_cycle = (' ').join(translate_list)
        return translate_cycle

    def pass_diy7_format(self, litmus_path):
        from src.slide.litmus.litmus_changer import match_cycle, get_max_register, InjectList, get_inject_list_by_thread
        ppo_no_Lr_and_Sc = self.ppo.get_no_Lr_and_Sc_ppo()
        other_no_Lr_and_Sc = self.other.get_no_Lr_and_Sc_ppo()
        if self.ppo == ppo_no_Lr_and_Sc and self.other == other_no_Lr_and_Sc:
            return True
        thread_list, connect_list = self.get_thread_ppos()
        cycle_no_Lr_and_Sc = Cycle(ppo_no_Lr_and_Sc, other_no_Lr_and_Sc)

        #create litmus test with Lr Sc 
        content = read_file(litmus_path)
        data = parse_litmus(content)
        print(data)

        max_reg_id = get_max_register(data)
        match_list = match_cycle(data,cycle_no_Lr_and_Sc)
        if match_list == []:
            return False
        print(f'match success: {match_list}')
        inject_list = InjectList()
        progs = data.progs
        thread_insts = []
        for prog in progs:
            thread_insts.append(prog.insts)

        # process
        for i in range(len(match_list)):
            inject_list.add_by_list(get_inject_list_by_thread(thread_list[i],thread_insts[match_list[i]],match_list[i],max_reg_id))
        
        data.mutate_new_litmus(inject_list, litmus_path)
        return True


class Diy7Generator:
    def __init__(self, time_limit = 1000):
        self.time_limit = time_limit
        self.ppo = None
        self.cycle_map = {}
        self.cycle_list = []
        self.before_cat_file = None
        self.after_cat_file = None
        self.i = 0

    def set_ppo(self, ppo :SinglePPO, before_cat_file, after_cat_file):
        self.ppo = ppo
        print('synth litmus test for ppo:',ppo)
        print('ppo',[type(item) for item in ppo.ppo])
        self.modify_ppo(ppo)
        # print('after modify ppo:',self.ppo)
        print('new_ppo',[type(item) for item in self.ppo.ppo])
        self.cycle_map = {} # str->1
        self.cycle_list = [] #(start,end,other)
        self.before_cat_file = before_cat_file
        self.after_cat_file = after_cat_file
        self.po_loc_ppo = None
        # self.get_po_loc_expend_ppo()

    def get_po_loc_expend_ppo(self):
        relation_list = []
        flag = False
        for ppo_item in self.ppo.ppo:
            relation_list.append(ppo_item)
            if type(ppo_item) == PoLoc:
                relation_list.extend(relation_list[-2:])
                flag = True
        if flag:
            self.po_loc_ppo = SinglePPO(relation_list)

    def modify_ppo(self, ppo : SinglePPO):
        relation_list = ppo.ppo
        # fix [X];...->[R];rmw;[X] to show X behavior
        while(True):
            flag = False
            new_relation_list = []
            for i, relation in enumerate(relation_list):
                if type(relation) == Sc and (i==0 or type(relation_list[i-2])!=Lr):
                    new_relation_list = relation_list[:i]+[Lr(),Rmw()]+relation_list[i:]
                    flag = True
            if flag:
                relation_list = new_relation_list
            else:
                break
        self.ppo = SinglePPO(relation_list)

    def init_cycle_list(self):

        # add start event
        start_relation_list = []
        start = self.ppo.start
        for pre_relation in jump_relation_list:
            if type(start) in ppo_can_add_relation_map[pre_relation]:
                start_relation_list.append([pre_relation()])

        for pre_relation in jump_relation_list:
            for first_ppo_unary_relation in ppo_add_relation_map[pre_relation]:
                for first_ppo_binary_relation in ppo_add_relation_map[first_ppo_unary_relation]:
                    if type(start) in ppo_can_add_relation_map[first_ppo_binary_relation]:
                        start_relation_list.append([pre_relation(),first_ppo_unary_relation(),first_ppo_binary_relation()])

        # print('start')
        # for item in start_relation_list:
        #     print(item)


        # add end event
        end_relation_list = []
        end = self.ppo.end
        for suc_relation in jump_relation_list:
            if suc_relation in ppo_can_add_relation_map[type(end)]:
                end_relation_list.append([suc_relation()])

        for suc_relation in jump_relation_list:
            for last_ppo_binary_relation in ppo_add_relation_map[type(end)]:  # add last_ppo
                for last_ppo_unary_relation in ppo_add_relation_map[last_ppo_binary_relation]:
                    if suc_relation in ppo_can_add_relation_map[last_ppo_unary_relation]:  # important
                        end_relation_list.append([last_ppo_binary_relation(), last_ppo_unary_relation(),suc_relation()])

        # print('end')
        # for item in end_relation_list:
        #     print(item)

        for start_list in start_relation_list:
            for end_list in end_relation_list:
                other_relation = ppo_add_relation_map[type(end_list[-1])][0] # because the end_relation in [fre,coe,rfe]
                self.cycle_list.append((start_list, end_list, [other_relation()]))

        # for item in self.cycle_list:
        #     print(item)


    def check(self, cycle : Cycle):

        # print(f'cycle{self.i}',cycle)
        # print('cycle diy7 str',cycle.to_diy_format())
        # self.i = self.i + 1
        # return False

        diy_str = cycle.to_diy_format()
        litmus_suite = [diy_str]
        print('use herd check start before pass transform', diy_str)
        # 1. create the litmus test
        test_dir = os.path.join(config.OUTPUT_DIR, f'test_ppo')
        dir_util.mk_dir_from_dir_path(test_dir)
        file_util.rm_files_with_suffix_in_dir(test_dir, '.litmus')
    
        new_test_paths = []
        for i, diy_cycle in enumerate(litmus_suite):
            new_test_path = os.path.join(test_dir, f"new_test.litmus")
            new_test_name = os.path.join(test_dir, f"new_test")
            # cmd = f"{config.DIYONE7_PATH} -arch RISC-V -name {new_test_name} {diy_cycle}"
            cmd = f"eval $(opam env);diyone7 -arch RISC-V -obs local -name {new_test_name} {diy_cycle}"
            print(f"diyone7 cmd: {cmd}")
            cmd_util.run_cmd(cmd)
            if os.path.exists(new_test_path):
                new_test_paths.append(new_test_path)
                if not cycle.pass_diy7_format(new_test_path):
                    return False
                content = read_file(new_test_path)
                print(content)

        print('use herd check start after pass transform', diy_str)
        flag = use_herd_test_ppo_by_litmus(new_test_paths, self.before_cat_file, self.after_cat_file)
        print('use herd check result: ', flag)
        return flag

    def get_cycle(self, start_list, end_list, other_list):
        other = SinglePPO(end_list+other_list+start_list)
        cycles = []
        cycle = Cycle(self.ppo, other)
        cycles.append(cycle)
        # if self.po_loc_ppo != None:
        #     cycles.append(Cycle(self.po_loc_ppo, other))
        return cycles

    def generate_litmus_test_legal(self):
        use_herd_check_time = 0
        use_static_check_time = 0
        mutate_time = 0
        while True:
            if mutate_time == self.time_limit:
                break
            mutate_time += 1
            if len(self.cycle_list) == 0:
                assert False, "this ppo can't create litmus test"
            start_list, end_list, other_list = self.cycle_list.pop(0)

            # generate
            relation = other_list[-1]
            
            # print('start')
            for binary_relation in other_add_relation_map[type(relation)]:
                for unary_relation in other_add_relation_map[binary_relation]:
                    new_start_list = copy.deepcopy(start_list)
                    new_end_list = copy.deepcopy(end_list)
                    new_other_list = copy.deepcopy(other_list)
                    new_other_list.append(binary_relation())
                    new_other_list.append(unary_relation())
                    cycles = self.get_cycle(new_start_list, new_end_list, new_other_list)
                    use_static_check_time += 1
                    for cycle in cycles:
                        if cycle.check():
                            use_herd_check_time += 1
                            if self.check(cycle):
                                test_dir = os.path.join(config.OUTPUT_DIR, f'test_ppo')
                                content = read_file(os.path.join(test_dir, f"new_test.litmus"))
                                litmus_name = str(cycle.ppo).replace('[', '').replace(']', '').replace(';', '_').replace('(','').replace(')','').replace('-','_')
                                litmus_name += '_'
                                litmus_name += str(cycle.to_diy_format()).replace(' ', '_').replace('.', '_').replace('-','_')
                                content = content.split('\n')
                                content[0] = f'RISCV {litmus_name}'
                                content = '\n'.join(content)
                                print(f'now find ppo:{self.ppo}, static time:{use_static_check_time}')
                                print(f'now find ppo:{self.ppo}, herd time:{use_herd_check_time}')
                                return cycle, content, litmus_name
                    self.cycle_list.append((start_list, end_list, other_list))
            # print('end')

        return None, None, None


if __name__ == '__main__':
    with open('output.txt', 'w') as f:
        sys.stdout = f



        mutateGenerator = Diy7Generator()

        old_cat_file_path = os.path.join(config.CAT_DIR, 'riscv-test.cat')
        new_cat_file_path = os.path.join(config.CAT_DIR, 'riscv-test1.cat')
        ppo = SinglePPO([R(),Po(),Lr(),Rmw(),Sc(),Po(),W()])
        mutateGenerator.set_ppo(ppo, old_cat_file_path, new_cat_file_path)
        mutateGenerator.init_cycle_list()
        cycle, litmus_file_content = mutateGenerator.generate_litmus_test_legal()
        print(cycle)
        print(cycle.ppo)
        print(cycle.to_diy_format())
        litmus_name = str(cycle.ppo).replace('[','').replace(']','').replace(';','_')
        litmus_name += '_'
        litmus_name += str(cycle.to_diy_format()).replace(' ','_').replace('.','_')
        print(litmus_name)
        print(litmus_file_content)

