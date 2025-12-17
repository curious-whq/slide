# MIT License
#
# Copyright (c) 2023 DehengYang (dehengyang@qq.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from multiprocessing import Manager, Process
from re import search
from typing import List

import networkx as nx
from math import ceil

from src.slide import config
from src.slide.analysis.model.mm import MemoryModel
from src.slide.analysis.model.rvwmo_without_ppo2 import RVWMO_WITHOUT_PPO2
from src.slide.comp.parse_result import parse_herd_log
from src.slide.litmus import parse_litmus
from src.slide.log import WARNING
from src.slide.synth.constraint import Constraint, get_violate_func_list
from src.slide.synth.ppo_def import PPOInitFlag
from src.slide.utils import file_util, list_util
from src.slide.utils.file_util import search_file, get_file_name_without_suffix_by_path
from src.slide.utils.herd_util import use_herd_test_ppo_by_litmus, herd_run_all_valid_litmus
from src.slide.utils.litmus7_util import litmus_run_until_match, litmus_run


def parallel_validate(rvwmo: MemoryModel, cur_mm: MemoryModel, litmus_test_suite: List, process_num=8):
    """
    Parallel execution of litmus tests
    """
    litmus_test_groups = []
    step = ceil(len(litmus_test_suite) / process_num)
    for i in range(0, len(litmus_test_suite), step):
        litmus_test_groups.append(litmus_test_suite[i:i + step])
    result_list = Manager().list()
    ps = []
    for litmus_test_group in litmus_test_groups:
        p = Process(target=validate, args=(rvwmo, cur_mm, litmus_test_group, result_list))
        ps.append(p)
        p.start()
    for p in ps:
        p.join()
    return result_list


def validate_use_herd(target_model_path: str, cur_mm_path: str, litmus_test_suite: List, result_list: List = [],
             early_exit=False):
    """
    A simple validation function
    :param rvwmo: cat file path
    :param cur_mm: synthesized model cat file path
    :param early_exit: exit once a failing test is identified
    :param verify_flag: True if the current phase the verification phase or the generation phase
    """
    validated = True
    failed_litmus_tests = []
    for litmus_file in litmus_test_suite:
        if use_herd_test_ppo_by_litmus([litmus_file], target_model_path, cur_mm_path):
            failed_litmus_tests.append(litmus_file)
            validated = False
            if early_exit:
                break

    result_list.append((failed_litmus_tests, validated))
    return failed_litmus_tests, validated

def validate_use_chip_log(chip_log_dict, cur_mm_path: str, litmus_test_suite: List):
    validated = True
    failed_litmus_tests = []
    herd_log = f'{config.INPUT_DIR}/herd/herd_result/synth_tmp.log'
    herd_run_all_valid_litmus(cur_mm_path, litmus_test_suite, herd_log, dependency_dir_path = None)
    rvwmo_log = f'{config.INPUT_DIR}/rvwmo_result/synth_tmp_rvwmo.log'
    herd_run_all_valid_litmus(cur_mm_path, litmus_test_suite, rvwmo_log, dependency_dir_path = None)
    herd_dict = {r.name: r.states for r in parse_herd_log(herd_log)}
    rvwmo_dict = {r.name: r.states for r in parse_herd_log(rvwmo_log)}

    for litmus_file in litmus_test_suite:
        litmus_name = get_file_name_without_suffix_by_path(litmus_file)
        print('validate:', litmus_name)
        if litmus_name not in chip_log_dict:
            # litmus_result = litmus_run(litmus_name, litmus_file, 1000)
            litmus_result = litmus_run_until_match(litmus_name, litmus_file, rvwmo_dict[litmus_name])
            chip_log_dict[litmus_name] = litmus_result
        print('herd:', herd_dict[litmus_name])
        print('chip:', chip_log_dict[litmus_name])
        if set(herd_dict[litmus_name])!= set(chip_log_dict[litmus_name]):
            failed_litmus_tests.append(litmus_file)
            validated = False
            print('failed:', litmus_file)
    return failed_litmus_tests, validated
    # return [], False


def validate(rvwmo: MemoryModel, cur_mm: MemoryModel, func_list, litmus_test_suite: List, result_list: List = [],
             early_exit=False, verify_flag=True):
    from src.slide.synth.pattern_synth import get_candidate_mm_by_func_list
    """
    A simple validation function
    :param rvwmo: target model (fewer states)
    :param cur_mm: synthesized model (more states)
    :param func_list: list of functions
    :param early_exit: exit once a failing test is identified
    :param verify_flag: True if the current phase the verification phase or the generation phase
    """
    validated = True
    any_ppo_all = []
    failed_litmus_tests = []
    for litmus_file in litmus_test_suite:
        litmus_instance = parse_litmus(file_util.read_file(litmus_file))
        print('validate litmus by tgt_mm')
        rvwmo.run(litmus_instance)  # chip (golden model) result
        print('validate litmus by cur_mm')
        cur_mm.run(litmus_instance)  # current model (to be synthesized) result

        a, b = set(rvwmo.states), set(cur_mm.states)
        a_minus_b = a - b  # should relax ppo
        # assert len(a_minus_b) == 0, "illegal states"
        if(verify_flag):
            if len(a_minus_b) > 0:
                WARNING("illegal states")
        b_minus_a = b - a  # should strengthen ppo
        # intersection = a & b  # should keep

        if rvwmo.states == cur_mm.states:  # passed
            continue
        # failed
        print(f'failed file: {litmus_file}')
        failed_litmus_tests.append(litmus_file)
        validated = False
        if early_exit:
            break
        
        for s in b_minus_a:  # traverse the extra states in cur_mm
            exe, ra = cur_mm.find_exe_by_state(s), cur_mm.find_ra_by_state(s)
            print('strengthen,s,exe',s,exe)
            any_ppo = [Constraint(e1, e2, ra, exe) for e1 in exe for e2 in exe if ra.po(e1, e2) and ra.gmo(e2, e1)]
            print('stengthen state,',s, ra, exe, any_ppo, False)
            any_ppo_all.append((s, ra, exe, any_ppo, False))  #strengthen


        # process /
        if not verify_flag:
            cur_mm.run(litmus_instance, complete_flag=True, base_model=RVWMO_WITHOUT_PPO2())
            print('complete exe list')
            print('get new mm')
            need_to_remove_func_name_list = []
            uninit_func_list = list(filter(lambda x:x[3]!=PPOInitFlag.Init, func_list))
            print('uninit_func_list')
            for index,gnode,func,init_flag in uninit_func_list:
                print(index,gnode,func,init_flag)
            for s in a_minus_b:
                exe, ra =cur_mm.find_exe_by_state(s), cur_mm.find_ra_by_state(s)
                remove_func_list = get_violate_func_list(ra, uninit_func_list)
                print('remove_func_list',remove_func_list)
                need_to_remove_func_name_list.extend(remove_func_list)
                any_ppo = [Constraint(e1, e2, ra, exe) for e1 in exe for e2 in exe if ra.po(e1, e2) and ra.gmo(e2, e1)]
                print('cur_mm,relax,s,exe',s,exe)
            # need_to_remove_func_name_list = list(set(need_to_remove_func_name_list))
            need_to_remove_func_name_list = [list(t) for t in
                                             set(tuple(sorted(x)) for x in need_to_remove_func_name_list)]
            for remove_list in need_to_remove_func_name_list:
                print('new_mm start')
                new_mm_func_list = list(filter(lambda x:x[0] not in remove_list, func_list))
                for name,gnode,func,init_flag in new_mm_func_list:
                    print(name,gnode,func,init_flag)
                print('new_mm end')
                new_mm = get_candidate_mm_by_func_list(new_mm_func_list)
                new_mm.run(litmus_instance)
                for s in a_minus_b:
                    assert s in new_mm.states, 'the new_mm must run the state'
                    exe, ra = new_mm.find_exe_by_state(s), new_mm.find_ra_by_state(s)
                    print('relax,s,exe',s,exe)
                    any_ppo = [Constraint(e1, e2, ra, exe) for e1 in exe for e2 in exe if ra.po(e1, e2) and ra.gmo(e2, e1)]
                    print('relax state,',s, ra, exe, any_ppo, True)
                    any_ppo_all.append((s, ra, exe, any_ppo, True))  #relax

        if early_exit:
            break

    result_list.append((failed_litmus_tests, any_ppo_all, validated))
    return failed_litmus_tests, any_ppo_all, validated



def validate_for_chip(cur_mm: MemoryModel, func_list, chip_state_dict, litmus_test_suite: List, result_list: List = [],
             early_exit=False, verify_flag=False):
    from src.slide.synth.pattern_synth import get_candidate_mm_by_func_list
    """
    A simple validation function for chip
    :param cur_mm: synthesized model (more states)
    :param func_list: list of functions
    :param chip_state_dict: dict of chip states
    :param early_exit: exit once a failing test is identified
    :param verify_flag: True if the current phase the generation phase or the verification phase
    """
    validated = True
    any_ppo_all = []
    failed_litmus_tests = []
    for i, litmus_file in enumerate(litmus_test_suite):
        litmus_instance = parse_litmus(file_util.read_file(litmus_file))
        litmus_name = get_file_name_without_suffix_by_path(litmus_file)
        chip_state=chip_state_dict[litmus_name] # chip (golden model) result
        print(f'run')
        cur_mm.run(litmus_instance) # current model (to be synthesized) result

        a, b = set(chip_state),set(cur_mm.states)
        a_minus_b = a - b  # should relax ppo
        # assert len(a_minus_b) == 0, "illegal states"
        if(verify_flag):
            if len(a_minus_b) > 0:
                WARNING("illegal states")
        b_minus_a = b - a  # should strengthen ppo
        # intersection = a & b  # should keep

        if len(b_minus_a) == 0:  # passed
            continue
        # failed
        print(f'failed file: {litmus_file}')
        print(f'b_minus_a: {b_minus_a}')
        failed_litmus_tests.append(litmus_file)
        validated = False
        if early_exit:
            break
        print(1)
        for s in b_minus_a:  # traverse the extra states in cur_mm
            exe, ra = cur_mm.find_exe_by_state(s), cur_mm.find_ra_by_state(s)
            
            print('strengthen,s,exe',s,exe)
            any_ppo = [Constraint(e1, e2, ra, exe) for e1 in exe for e2 in exe if ra.po(e1, e2) and ra.gmo(e2, e1)]
            print('stengthen state,',s, ra, exe, any_ppo, False)
            any_ppo_all.append((s, ra, exe, any_ppo, False))  #strengthen


            # plot
            # cur_mm.plot_execution(litmus_instance, None, exe, s, ra)
        if not verify_flag:
            cur_mm.run(litmus_instance, complete_flag=True, base_model=RVWMO_WITHOUT_PPO2())
            print('complete exe list')
            print('get new mm')
            need_to_remove_func_name_list = []
            uninit_func_list = list(filter(lambda x: x[3] != PPOInitFlag.Init, func_list))
            print('uninit_func_list')
            for index, gnode, func, init_flag in uninit_func_list:
                print(index, gnode, func, init_flag)
            for s in a_minus_b:
                exe, ra = cur_mm.find_exe_by_state(s), cur_mm.find_ra_by_state(s)
                remove_func_list = get_violate_func_list(ra, uninit_func_list) #[[ppo1,ppo2],[ppo3]]
                print('remove_func_list', remove_func_list)
                need_to_remove_func_name_list.extend(remove_func_list)
                any_ppo = [Constraint(e1, e2, ra, exe) for e1 in exe for e2 in exe if ra.po(e1, e2) and ra.gmo(e2, e1)]
                print('cur_mm,relax,s,exe', s, exe)

            need_to_remove_func_name_list = [list(t) for t in
                                                 set(tuple(sorted(x)) for x in need_to_remove_func_name_list)]
            # need_to_remove_func_name_list = list(set(need_to_remove_func_name_list))
            print('new_mm start')
            for remove_list in need_to_remove_func_name_list:
                new_mm_func_list = list(filter(lambda x: x[0] not in remove_list, func_list))
                for name, gnode, func, init_flag in new_mm_func_list:
                    print(name, gnode, func, init_flag)
                print('new_mm end')
                new_mm = get_candidate_mm_by_func_list(new_mm_func_list)
                new_mm.run(litmus_instance)
                for s in a_minus_b:
                    assert s in new_mm.states, 'the new_mm must run the state'
                    exe, ra = new_mm.find_exe_by_state(s), new_mm.find_ra_by_state(s)
                    print('relax,s,exe', s, exe)
                    any_ppo = [Constraint(e1, e2, ra, exe) for e1 in exe for e2 in exe if ra.po(e1, e2) and ra.gmo(e2, e1)]
                    print('relax state,', s, ra, exe, any_ppo, True)
                    any_ppo_all.append((s, ra, exe, any_ppo, True))  # relax

        if early_exit:
            break

    result_list.append((failed_litmus_tests, any_ppo_all, validated))
    return failed_litmus_tests, any_ppo_all, validated


def get_dif_detail(mm: MemoryModel, cur_log, tgt_log, dif_list, detail_file):
    config.init()
    config.set_var('reg_size', 64)
    result_list = []
    with open(detail_file, 'w') as f:
        for i, litmus_name in enumerate(dif_list):
            litmus_file = search_file(litmus_name, f'{config.INPUT_DIR}/litmus', '.litmus')
            litmus_instance = parse_litmus(file_util.read_file(litmus_file))

            mm.run(litmus_instance)
            print(f'run')

            cur_state, tgt_state = set(cur_log[litmus_name] ),set(tgt_log[litmus_name])

            tgt_minus_cur = tgt_state - cur_state
            cur_minus_tgt = cur_state - tgt_state

            item = {
                'name': litmus_name,
                'litmus': file_util.read_file(litmus_file),
                'state': [],
                'exe': [],
            }
            for state in tgt_minus_cur:
                exe, ra = mm.find_exe_by_state(state), mm.find_ra_by_state(state)
                item['state'].append(state)
                item['exe'].append(exe)
            result_list.append(item)
            f.write(f'now is {i}\n')
            f.write(item['name'])
            f.write('\n')
            f.write(item['litmus'])
            f.write('\n')
            for i, s in enumerate(item['state']):
                f.write(str(s))
                f.write('\n')
                for event in item['exe'][i]:
                    f.write(str(event))
                    f.write('\n')
            f.write('\n')
    return  result_list












