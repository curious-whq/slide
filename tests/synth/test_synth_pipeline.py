

import os
import re
import sys
import pytest

from src.slide.analysis.model.rvwmo_x import RVWMOX
from src.slide.litmus.load_litmus import get_litmus_by_policy, GetLitmusPolicy
from src.slide.synth.model.ppo_pool_rvwmo_x import get_rvwmo_modelx

sys.path.append("../src")
from src.slide import config
from src.slide.analysis.model.rvwmo import *
from src.slide.analysis.model.rvwmo_strong_ppo2 import *
from src.slide.litmus import parse_litmus
from src.slide.synth import transform, bottom_up, top_down
from src.slide.synth.bottom_up import Grammar, Spec
from src.slide.synth.pattern_synth import synth_by_test_pattern, synth_by_test_pattern_online
from src.slide.utils import dir_util, file_util, regex_util
from src.slide.validate.validate import validate, parallel_validate, validate_for_chip
from src.slide.ppo.ppo_parser import parse_to_gnode_tree
from src.slide.utils.file_util import *
from src.slide.comp.parse_result import parse_chip_log

# cur_dir = dir_util.get_cur_dir(__file__)
# input_dir = os.path.abspath(os.path.join(cur_dir, "../input"))
# litmus_dir = os.path.abspath(os.path.join(input_dir, "./litmus/non-mixed-size"))
#
# def find_test_suite_with_specific_ppo():
#     """
#     A function to explore which tests have the specific ppo (e.g., ppo2)
#     As we currently still do not implement the feature: identify the exact ppo rule between two
#     events, We have to add code like:
#         if 'ppo_r2' in str(ppo_rule):
#             pass
#     in "rel.py:PPO():wrapper()" function to detect if the specified ppo rule is identified.
#     """
#     litmus_test_suite = all_litmus_files
#     for litmus_file in litmus_test_suite:
#         if litmus_file.endswith('/MP+po+ctrl.litmus'):
#             continue
#         litmus_content = file_util.read_file(litmus_file)
#         litmus_instance = parse_litmus(litmus_content)
#         rvwmo = RVWMO(plot_enabled=False)
#         rvwmo.run(litmus_instance)
#
#
# def get_litmus_paths_from_names(all_litmus_files, litmus_names) -> List[str]:
#     """
#     Get litmus file paths from litmus file names
#     """
#     litmus_test_suite = []
#     for litmus_name in litmus_names:
#         if not litmus_name.endswith('.litmus'):
#             litmus_name += '.litmus'
#         match = False
#         for litmus_file in all_litmus_files:
#             if litmus_file.endswith(f'/{litmus_name}'):
#                 litmus_test_suite.append(litmus_file)
#                 match = True
#                 break
#     return litmus_test_suite
#
#
# def get_all_litmus_files() -> List[str]:
#     """
#     Get all litmus files (size: 3536) after filtering out the exceptional cases.
#     """
#     print(litmus_dir)
#     ori_litmus_files = file_util.list_files(litmus_dir, '.litmus')
#     exceptional_litmus_files = file_util._read_file_to_list_strip(
#         os.path.join(input_dir, 'herd/exceptional_aq_rl.txt'))
#
#     all_litmus_files = []
#     for litmus_file in ori_litmus_files:
#         litmus_name = file_util.get_file_name_from_path(litmus_file)
#         if litmus_name not in exceptional_litmus_files:
#             all_litmus_files.append(litmus_file)
#     return all_litmus_files
#
#
# def get_all_litmus_files_sorted_by_time(all_litmus_files):
#     """
#     parse litmus test info from pytest log file:
#     <li class="level test">
#         <span><em class="time">
#                 <div class="time">1.37 s</div>
#             </em><em class="status">passed</em>(MP+po+poarp+NEW-state_cnt54)</span>
#     </li>
#     """
#     log_file_string = file_util.read_file(
#         os.path.join(input_dir, 'litmus_tests_time_cost.log'))
#     time_name_tuples = regex_util.findall(
#         '<li class="level test">.*?<div class="time">(.*?)<\/div>.*?\((.*?)-state_.*?\)', log_file_string, re.S)
#
#     # unify the time unit (i.e., s) e.g., 1 m 0 s,210 ms,2.80 s
#     def get_time(time):
#         elements = time.split(' ')
#         if ' m ' in time:
#             time = float(elements[0]) * 60 + float(elements[2])
#         elif ' ms' in time:
#             time = float(elements[0]) / 1000
#         else:
#             time = float(elements[0])
#         return time
#
#     time_name_tuples = [(get_time(time), name.replace('_', '.')) for (time, name) in time_name_tuples]
#     time_name_tuples = sorted(time_name_tuples, key=lambda time_name_tuple: time_name_tuple[0])
#     total_time_cost = sum([time_name[0] for time_name in time_name_tuples]) / 60  # minutes
#     litmus_names = [time_name_tuple[1] for time_name_tuple in time_name_tuples]
#
#     all_litmus_files_sorted_by_time = get_litmus_paths_from_names(all_litmus_files, litmus_names)
#     assert len(all_litmus_files_sorted_by_time) == len(litmus_names), 'Some litmus tests are not found in ' \
#                                                                       'all_litmus_files.'
#     return all_litmus_files_sorted_by_time


# all_litmus_files = get_all_litmus_files()
# all_litmus_files_sorted_by_time = get_all_litmus_files_sorted_by_time(all_litmus_files)

all_litmus_files_sorted_by_time = get_litmus_by_policy(GetLitmusPolicy.SortByTime,{})
print(len(all_litmus_files_sorted_by_time))
class TestSynthPipeline:
    def test_bottom_up_enum_search(self):
        """
        size 10000 -->
        v0: 2min19s
        v1: 2min5s (size check)
        v2: 17s42ms (traversed_nodes reuse)
        v3: 11s244ms (avoid repeated generation of non_terminal_leaf_node)
        """
        # print('start')
        g = Grammar()
        spec = Spec()
        syntax_valid_patch_list = bottom_up.bottom_up_enum_search(g, spec, 2000)
        with open('synth/test_top_down.txt','w') as f:
            sys.stdout = f
            for p in syntax_valid_patch_list:
                print(p, p.get_size())

    def test_top_down_enum_search(self):
        """
        Test the top_down_enum_search() function
        5000 patches:
        original version: 72s
        optimization round1: 2s543ms (add number_of_non_terminal_leafs to speed up p_list.sort(key=lambda a:
        a.get_number_of_non_terminal_leafs()))
        optimization round2: 1s380ms (replace copy.deepcopy(root) by defining copy_tree(root))
        optimization round3: 1s687ms (remove p_list.sort(key=lambda a: a.get_number_of_non_terminal_leafs()) and use )
        optimization round4: 1s250ms (change the data structure of tried_p_list from list to set)
        10000 patches
        original version: 27min42s
        optimization round1: 37s787ms
        optimization round2: 21s373ms
        optimization round3: 4s196ms
        optimization round4: 2s550ms
        """
        g = Grammar()
        spec = Spec()
        syntax_valid_patch_list = top_down.top_down_enum_search(g, spec, 1000)
        with open('test_top_down.txt','w') as f:
            sys.stdout = f
            for p in syntax_valid_patch_list:
                print(p)

    def test_cal_enum_search_space_with_depth_limit(self):
        """
        depth 1: 1
        depth 2: 7
        depth 3: 175
        depth 4: 92071
            optimization round 1 (use deque() for p_list rather than list): ~2min
        """
        g = Grammar()
        spec = Spec()
        depth = 3
        top_down.cal_enum_search_space_with_depth_limit(g, spec, depth)

    def test_synthesis_pipeline_for_ppo4(self):
        """
        1. pre-validation (including test suite preparation, memory model initialization,
        and test suite execution)
        2. patch generation
        3. patch validation
        """
        # 1-1) prepare test suite
        config.init()
        config.set_var('reg_size', 64)
        litmus_names = ['ISA2+fence.rw.rw+ctrl+fence.rw.rw.litmus', 'MP+po+ctrl.litmus',
                        'MP+fence.rw.rw+ctrl.litmus', 'LB+fence.rw.rws.litmus']
        litmus_test_suite = get_litmus_paths_from_names(all_litmus_files, litmus_names) + all_litmus_files[:100]

        # 1-2) initialize the standard model
        rvwmo = RVWMO(plot_enabled=False)

        # 1-3) initialize cur_mm that lacks ppo4
        global_ppos = [ppo_r1, ppo_r2, ppo_r3, ppo_r5, ppo_r6, ppo_r7, ppo_r8, ppo_r9,
                       ppo_r10, ppo_r11, ppo_r12, ppo_r13]
        local_ppos = [p for p in global_ppos if p not in [ppo_r2, ppo_r3, ppo_r12]]
        cur_mm = RVWMO()
        # cur_mm.ppo_g = PPO(global_ppos)
        # cur_mm.ppo_l = PPO(local_ppos)

        # 1-4) pre validation
        failed_litmus_cnt, any_ppo_all, validated = validate(rvwmo, cur_mm, litmus_test_suite, False)
        print(f'pre-validation result (tests size: {len(litmus_test_suite)}): {validated}')

        # 2) generate patch.
        g = Grammar()
        spec = Spec()
        syntax_valid_patch_list = top_down.top_down_enum_search(g, spec, 500)
        # ppo_candidate = '[M];fence;[M]'
        # patch = parse_to_gnode_tree(ppo_candidate)
        for patch in syntax_valid_patch_list:
            # get fence ppo
            if str(patch) != 'fence':
                continue

            python_func_string = transform.transform(patch)
            # CAUTION: must has globals() para, otherwise the ppo_candidate_func won't be recognized
            exec(python_func_string, globals())

            cur_mm.ppo_g = PPO(global_ppos + [ppo_candidate_func])  # add to globals
            if not ('rf' in str(patch) or 'rsw' in str(patch) or 'co' in str(patch)):
                cur_mm.ppo_l = PPO(local_ppos + [ppo_candidate_func])  # add to locals if not include
                # 'rf' 'rsw' 'co'

            # 3) validation
            states_compare, validated = validate(rvwmo, cur_mm, litmus_test_suite)
            print(f'the patch is validated (tests size: {len(litmus_test_suite)}): {validated}')



    def test_collect_data_by_diff_log(self):
        # init
        config.init()
        config.set_var('reg_size', 64)
        with open('output.txt','w') as wf:
            sys.stdout = wf
            litmus_to_ppo_map = {}
            ppo_to_litmus_map = {}
            litmus_list = []
            with open(f'{config.INPUT_DIR}/chip_execution_logs/chip_dif_not_exceed_two_1000000.txt','r') as f:
                list = f.readlines()
                litmus_list = [item.strip() for item in list]
                f.close()
            # print(litmus_list)
            litmus_test_suite = []
            # for item in litmus_test_suite:
            #     print(item)
            rvwmo = RVWMO()
            with open(f'{config.INPUT_DIR}/chip_execution_logs/chip_dif_not_exceed_two_1000000_litmus.txt','w') as f:
                for i,name in enumerate(litmus_list):
                    file_name = search_file(name, 'input/litmus', '.litmus')
                    print(i,file_name)
                    litmus_test_suite.append(file_name)
                    if file_name == None:
                        continue
                    f.write(file_name+'\n')
                f.close() 

            # with open(f'{config.INPUT_DIR}/chip_execution_logs/chip_dif_litmus.txt','r') as f:
            #     list = f.readlines()
            #     litmus_test_suite = [item.strip() for item in list]
            #     f.close() 
            print(litmus_test_suite)
            chip_log_path = f'{config.INPUT_DIR}/chip_execution_logs/chip_log_1000000.txt'
            chip_log = {r.name: r.states for r in parse_chip_log(chip_log_path)}
            print(len(chip_log.keys()))
            # collect state
            print('start collect')
            for i in range(775,len(litmus_test_suite)):
                print(i)
                litmus_test = litmus_test_suite[i]
            # for i,litmus_test in enumerate(litmus_test_suite):
                assert litmus_list[i] in chip_log, 'the litmus test must in chip log'
                print(f'collect {i}')
                if i <= 774:
                    continue
                print(1)
                num = i
                litmus = litmus_list[i]
                # get
                _, any_ppo_all, _ = validate_for_chip(rvwmo, [chip_log[litmus_list[i]]], [litmus_test])
                
                # collect
                tried_candidate_ppos = []
                for _,_,_, any_ppo,relax_flag in any_ppo_all:
                    for constraint in any_ppo:
                        for j, candidate_ppo in enumerate(constraint.candidate_ppos):
                            if candidate_ppo in tried_candidate_ppos:
                                continue
                            tried_candidate_ppos.append(candidate_ppo)
                
                litmus_to_ppo_map[litmus_list[i]]=tried_candidate_ppos
                for ppo in tried_candidate_ppos:
                    if ppo in ppo_to_litmus_map:
                        ppo_to_litmus_map[ppo].append(litmus_list[i])
                    else:
                        ppo_to_litmus_map[ppo] = [litmus_list[i]]
                
                
                list = tried_candidate_ppos
                print(f'add litmus_to_ppo {list}')
                with open('collect.txt','a') as data_f:
                    data_f.write(f'litmus to ppo {num}: ')
                    data_f.write(f"litmus {litmus}:{len(list)}\n")
                    for item in list:
                        data_f.write(f'   {item}\n')
                    data_f.close()

            #write_to_file
            with open('collect_data_total.txt','w') as f:
                f.write('ppo to litmus\n ')
                for key in ppo_to_litmus_map:
                    list = ppo_to_litmus_map[key]
                    f.write(f"{key}:{len(list)},  {' ;'.join(list)}\n")
                
                f.write('------------------------------------------\n')
                f.write('litmus to ppo\n')
                for key in litmus_to_ppo_map:
                    list = litmus_to_ppo_map[key]
                    f.write(f"litmus {key}:{len(list)}\n")
                    for item in list:
                        f.write(f'   {item}\n')
                f.close()

    def test_diyone7_based_pipeline_for_ppo1(self):
        """
        This is to explore the pattern of ppo1
        """
        with open('output.txt','w') as f:
            original_stdout=sys.stdout
            sys.stdout=f
            config.init()
            config.set_var('reg_size', 64)

            litmus_test_suite = all_litmus_files_sorted_by_time
            litmus_test_suite = [litmus_test for litmus_test in litmus_test_suite if
                                'addr-rfi' in litmus_test or 'data-rfi' in litmus_test][:100]
            passed_tests = []  # all_litmus_files_sorted_by_time[:10]
            litmus_test_suite = [
                                 'input/litmus/diy7/ppo1_WW.litmus',
                                #  'input/litmus/non-mixed-size/CO/CoWW.litmus',
                                 'input/litmus/non-mixed-size/CO/CoRW2.litmus',
                                #  'input/litmus/non-mixed-size/RELAX/Rfi/MP+rfi-addr+ctrl-rfi-ctrlfenceis.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-data+data-rfi-ctrlfencei.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-addr+data-rfi-ctrlfencei.litmus',
                                # 'input/litmus/non-mixed-size/HAND/ISA11.litmus',
                                # 'input/litmus/non-mixed-size/AMO_X0_2_THREAD/R+popar+poarp+NEW.litmus',
                                # 'input/litmus/non-mixed-size/CO/CoRR.litmus',
                                # diy7 created in shell manually
                                # 'input/litmus/manual/LB+addr+data-po-ctrl.litmus',

                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-addr.litmus'
                                ] + passed_tests
            rvwmo = RVWMO(plot_enabled=False)

            # 1-3) initialize cur_mm that lacks ppo12
            global_ppos = [ppo_r11, ppo_r2, ppo_r3, ppo_r4, ppo_r5, ppo_r6, ppo_r7, ppo_r8,
                        ppo_r9, ppo_r10, ppo_r12, ppo_r13]
            local_ppos = [p for p in global_ppos if p not in [ppo_r2, ppo_r3, ppo_r12]]
            cur_mm = RVWMO()
            cur_mm.ppo_g = PPO(global_ppos)
            cur_mm.ppo_l = PPO(local_ppos)

            # 1-4) pre validation
            # parallel approach
            # result_list = parallel_validate(rvwmo, cur_mm, litmus_test_suite)
            # failed_litmus_tests, any_ppo_all, validated = validate(rvwmo, cur_mm, litmus_test_suite)
            # print('failed_litmus_tests:', len(failed_litmus_tests))
            print(1)

            cat_file_name = 'riscv-remove-ppo-1.cat'
            patches = synth_by_test_pattern_online_for_one_ppo(rvwmo, cur_mm, global_ppos, litmus_test_suite,cat_file_name)
            print('patches',patches)
            sys.stdout=original_stdout
            f.close()

    def test_diyone7_based_pipeline_for_ppo2_All(self):
        """
        This is to explore the pattern of ppo2
        """
        with open('output.txt','w') as f:
            original_stdout=sys.stdout
            sys.stdout=f
            config.init()
            config.set_var('reg_size', 64)

            litmus_test_suite = all_litmus_files_sorted_by_time
            litmus_test_suite = [litmus_test for litmus_test in litmus_test_suite if
                                'addr-rfi' in litmus_test or 'data-rfi' in litmus_test][:100]
            passed_tests = []  # all_litmus_files_sorted_by_time[:10]
            litmus_test_suite = [
                                # 'input/litmus/diy7/PPO2_remove_R_po_R.litmus' ,
                                # 'input/litmus/non-mixed-size/SAFE/ISA2+pos+ctrl+addr.litmus',
                                # 'input/litmus/non-mixed-size/CO/CoRR.litmus',
                                #  'input/litmus/diy7/PPO2_remove_R_po_R.litmus' ,
                                # 'input/litmus/non-mixed-size/HAND/RSW.litmus',
                                #  'input/litmus/non-mixed-size/HAND/MP+fence.w.w+fri-rfi-addr.litmus',
                                 'input/litmus/manual/test_PPO2_amo.litmus'
                                #  'input/litmus/non-mixed-size/SAFE/ISA2+pos+ctrl+addr.litmus',
                                #  'input/litmus/non-mixed-size/RELAX/Rfi/MP+rfi-addr+ctrl-rfi-ctrlfenceis.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-data+data-rfi-ctrlfencei.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-addr+data-rfi-ctrlfencei.litmus',
                                # 'input/litmus/non-mixed-size/HAND/ISA11.litmus',
                                # 'input/litmus/non-mixed-size/AMO_X0_2_THREAD/R+popar+poarp+NEW.litmus',
                                # 'input/litmus/non-mixed-size/CO/CoRR.litmus',
                                # diy7 created in shell manually
                                # 'input/litmus/manual/LB+addr+data-po-ctrl.litmus',

                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-addr.litmus'
                                ] + passed_tests
            rvwmo = RVWMO(plot_enabled=False)

            # 1-3) initialize cur_mm that lacks ppo12
            global_ppos = [ppo_r1, ppo_r12, ppo_r3, ppo_r4, ppo_r5, ppo_r6, ppo_r7, ppo_r8, ppo_r9,
                        ppo_r10, ppo_r11, ppo_r13, ppo_r2_strong]
            local_ppos = [p for p in global_ppos if p not in [ppo_r2, ppo_r3, ppo_r12]]
            cur_mm = RVWMO()
            cur_mm.ppo_g = PPO(global_ppos)
            cur_mm.ppo_l = PPO(local_ppos)

            # 1-4) pre validation
            # parallel approach
            # result_list = parallel_validate(rvwmo, cur_mm, litmus_test_suite)
            # failed_litmus_tests, any_ppo_all, validated = validate(rvwmo, cur_mm, litmus_test_suite)
            # print('failed_litmus_tests:', len(failed_litmus_tests))
            print(1)
            cat_file_name = 'riscv-remove-ppo-2.cat'
            
            patches = synth_by_test_pattern_online_for_one_ppo(rvwmo, cur_mm, global_ppos, litmus_test_suite, cat_file_name)
            print('patches',patches)
            sys.stdout=original_stdout
            f.close()

    def test_diyone7_based_pipeline_for_ppo3(self):
        """
        This is to explore the pattern of ppo3
      """
        with open('output.txt','w') as f:
            original_stdout=sys.stdout
            sys.stdout=f
            config.init()
            config.set_var('reg_size', 64)

            litmus_test_suite = all_litmus_files_sorted_by_time
            litmus_test_suite = [litmus_test for litmus_test in litmus_test_suite if
                                'addr-rfi' in litmus_test or 'data-rfi' in litmus_test][:100]
            passed_tests = []  # all_litmus_files_sorted_by_time[:10]
            litmus_test_suite = [
                                # 'input/litmus/non-mixed-size/HAND/ForwardAMO.litmus',
                                # 'input/litmus/non-mixed-size/HAND/ForwardSc.litmus',
                                # 'input/litmus/manual/test_PPO3_sc.litmus',
                                 'input/litmus/manual/test_PPO3.litmus'
                                ] + passed_tests
            rvwmo = RVWMO(plot_enabled=False)

            # 1-3) initialize cur_mm that lacks ppo12
            global_ppos = [ppo_r1, ppo_r2, ppo_r4, ppo_r5, ppo_r6, ppo_r7, ppo_r8,
                        ppo_r9, ppo_r10, ppo_r11, ppo_r12, ppo_r13]
            local_ppos = [p for p in global_ppos if p not in [ppo_r2, ppo_r3, ppo_r12]]
            cur_mm = RVWMO()
            cur_mm.ppo_g = PPO(global_ppos)
            cur_mm.ppo_l = PPO(local_ppos)

            # 1-4) pre validation
            # parallel approach
            # result_list = parallel_validate(rvwmo, cur_mm, litmus_test_suite)
            # failed_litmus_tests, any_ppo_all, validated = validate(rvwmo, cur_mm, litmus_test_suite)
            # print('failed_litmus_tests:', len(failed_litmus_tests))
            print(1)
            cat_file_name = 'riscv-remove-ppo-3.cat'
            patches = synth_by_test_pattern_online_for_one_ppo(rvwmo, cur_mm, global_ppos, litmus_test_suite, cat_file_name)
            print('patches',patches)
            sys.stdout=original_stdout
            f.close()


    def test_diyone7_based_pipeline_for_ppo4(self):
        """
        This is to explore the pattern of ppo4
      """
        with open('output.txt','w') as f:
            original_stdout=sys.stdout
            sys.stdout=f
            config.init()
            config.set_var('reg_size', 64)

            litmus_test_suite = all_litmus_files_sorted_by_time
            litmus_test_suite = [litmus_test for litmus_test in litmus_test_suite if
                                'addr-rfi' in litmus_test or 'data-rfi' in litmus_test][:100]
            passed_tests = []  # all_litmus_files_sorted_by_time[:10]
            litmus_test_suite = [
                                'input/litmus/non-mixed-size/HAND/PPOLDSTLD01.litmus',
                                'input/litmus/non-mixed-size/HAND/C-Will01-Bad.litmus',
                                'input/litmus/non-mixed-size/HAND/C-Will02+HEAD.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA-MP-DEP-SUCCESS-SWAP-SIMPLE.litmus',
                                'input/litmus/non-mixed-size/HAND/RSW+W.litmus',
                                'input/litmus/non-mixed-size/HAND/ForwardAMO.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA10+BIS.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA-DEP-CTRL.litmus',
                                'input/litmus/non-mixed-size/HAND/PPODA.litmus',
                                'input/litmus/non-mixed-size/HAND/ForwardSc.litmus',
                                'input/litmus/non-mixed-size/HAND/C-Will03.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA14+BIS.litmus',
                                'input/litmus/non-mixed-size/HAND/PPOAA.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA-MP-DEP-ADDR-LR-FAIL.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA-MP-DEP-SUCCESS-SWAP.litmus',
                                'input/litmus/non-mixed-size/HAND/PPOLDSTLD02.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA-DEP-WR-ADDR.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA14.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA13+BIS.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA-MP-DEP-SUCCESS-SUCCESS.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA14+TER.litmus',
                                'input/litmus/non-mixed-size/HAND/C-Will02.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA09+BIS.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA14+NEW.litmus',
                                'input/litmus/non-mixed-size/HAND/RDW.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA-S-DEP-ADDR-SUCCESS.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA-MP-DEP-SUCCESS.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA09.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA-S-DEP-DATA-SUCCESS.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA-DEP-ADDR.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA-MP-DEP-ADDR-LR-SUCCESS.litmus',
                                ] + passed_tests
            rvwmo = RVWMO(plot_enabled=False)

            # 1-3) initialize cur_mm that lacks ppo12
            global_ppos = [ppo_r1, ppo_r2, ppo_r3, ppo_r5, ppo_r6, ppo_r7, ppo_r8,
                        ppo_r9, ppo_r10, ppo_r11, ppo_r12, ppo_r13]
            local_ppos = [p for p in global_ppos if p not in [ppo_r2, ppo_r3, ppo_r12]]
            cur_mm = RVWMO()
            cur_mm.ppo_g = PPO(global_ppos)
            cur_mm.ppo_l = PPO(local_ppos)

            # 1-4) pre validation
            # parallel approach
            # result_list = parallel_validate(rvwmo, cur_mm, litmus_test_suite)
            # failed_litmus_tests, any_ppo_all, validated = validate(rvwmo, cur_mm, litmus_test_suite)
            # print('failed_litmus_tests:', len(failed_litmus_tests))
            print(1)
            patches = synth_by_test_pattern_online_for_one_ppo(rvwmo, cur_mm, global_ppos, litmus_test_suite)
            print('patches',patches)
            sys.stdout=original_stdout
            f.close()

    def test_diyone7_based_pipeline_for_ppo5(self):
        """
        This is to explore the pattern of ppo5
      """
        with open('output.txt','w') as f:
            original_stdout=sys.stdout
            sys.stdout=f
            config.init()
            config.set_var('reg_size', 64)

            litmus_test_suite = all_litmus_files_sorted_by_time
            litmus_test_suite = [litmus_test for litmus_test in litmus_test_suite if
                                'addr-rfi' in litmus_test or 'data-rfi' in litmus_test][:100]
            passed_tests = []  # all_litmus_files_sorted_by_time[:10]
            litmus_test_suite = [
                                # 'input/litmus/non-mixed-size/HAND/ISA11.litmus',
                                'input/litmus/non-mixed-size/HAND/ISA-OLD+BIS.litmus',
                                # 'input/litmus/manual/test_PPO5.litmus',
                            #    'input/litmus/non-mixed-size/AMO_X0_2_THREAD/LB+poarps+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/R+popar+poarp+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/MP+popar+poarp+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/S+poarps+NEW.litmus',
                                # 'input/litmus/non-mixed-size/AMO_X0_2_THREAD/MP+poarar+poarp+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/LB+popar+poarp+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/2+2W+popar+poarp+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/MP+poarps+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/LB+poarp+poarar+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/SB+popar+poarp+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/SB+poarp+poarar+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/R+poarps+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/R+poarp+popar+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/LB+poaqps+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/MP+poarp+poarar+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/2+2W+poarp+poarar+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/R+poarar+poarp+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/S+poarar+poarp+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/S+popar+poarp+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/R+poarp+poarar+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/2+2W+poarps+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/MP+poarp+popar+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/SB+poarps+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/S+poarp+popar+NEW.litmus',
                            #     'input/litmus/non-mixed-size/AMO_X0_2_THREAD/S+poarp+poarar+NEW.litmus',
                            #     'input/litmus/non-mixed-size/HAND/LR-SC-NOT-FENCE.litmus',
                            #     'input/litmus/non-mixed-size/HAND/Luc02+BIS.litmus',
                                # 'input/litmus/non-mixed-size/HAND/C-Will02+HEAD.litmus',
                                # 'input/litmus/non-mixed-size/HAND/Luc01.litmus',
                                # 'input/litmus/non-mixed-size/HAND/Luc03+BIS.litmus',
                                # 'input/litmus/non-mixed-size/HAND/2+2Swap+Acqs.litmus',
                                # 'input/litmus/non-mixed-size/HAND/C-Will03.litmus',
                                # 'input/litmus/non-mixed-size/HAND/ISA03+SIMPLE.litmus',
                                # 'input/litmus/non-mixed-size/HAND/AMO-FENCE.litmus',
                                # 'input/litmus/non-mixed-size/HAND/ISA13.litmus',
                                # 'input/litmus/non-mixed-size/HAND/ISA13+BIS.litmus',
                                # 'input/litmus/non-mixed-size/HAND/Luc02.litmus',
                                # 'input/litmus/non-mixed-size/HAND/C-Will01-Bad.litmus',
                                # 'input/litmus/non-mixed-size/HAND/ISA11+BIS.litmus',
                                # 'input/litmus/non-mixed-size/HAND/Luc01+BIS.litmus',
                                # 'input/litmus/non-mixed-size/HAND/Luc03.litmus',
                                # 'input/litmus/non-mixed-size/HAND/ISA03.litmus',
                                # 'input/litmus/non-mixed-size/HAND/ISA03+SB01.litmus',
                                ] + passed_tests
            rvwmo = RVWMO(plot_enabled=False)

            # 1-3) initialize cur_mm that lacks ppo12
            global_ppos = [ppo_r1, ppo_r2, ppo_r3, ppo_r4, ppo_r6, ppo_r7, ppo_r8,
                        ppo_r9, ppo_r10, ppo_r11, ppo_r12, ppo_r13]
            local_ppos = [p for p in global_ppos if p not in [ppo_r2, ppo_r3, ppo_r12]]
            cur_mm = RVWMO()
            cur_mm.ppo_g = PPO(global_ppos)
            cur_mm.ppo_l = PPO(local_ppos)

            # 1-4) pre validation
            # parallel approach
            # result_list = parallel_validate(rvwmo, cur_mm, litmus_test_suite)
            # failed_litmus_tests, any_ppo_all, validated = validate(rvwmo, cur_mm, litmus_test_suite)
            # print('failed_litmus_tests:', len(failed_litmus_tests))
            print(1)
            cat_file_name = 'riscv-remove-ppo-5.cat'
            patches = synth_by_test_pattern_online_for_one_ppo(rvwmo, cur_mm, global_ppos, litmus_test_suite, cat_file_name)
            print('patches',patches)
            sys.stdout=original_stdout
            f.close()

    def test_diyone7_based_pipeline_for_ppo6(self):
        """
        This is to explore the pattern of ppo6
      """
        with open('output.txt','w') as f:
            original_stdout=sys.stdout
            sys.stdout=f
            config.init()
            config.set_var('reg_size', 64)

            litmus_test_suite = all_litmus_files_sorted_by_time
            litmus_test_suite = [litmus_test for litmus_test in litmus_test_suite if
                                'addr-rfi' in litmus_test or 'data-rfi' in litmus_test][:100]
            passed_tests = []  # all_litmus_files_sorted_by_time[:10]
            litmus_test_suite = [
                                #  'input/litmus/non-mixed-size/RELAX/Rfi/MP+rfi-addr+ctrl-rfi-ctrlfenceis.litmus',
                                 'input/litmus/non-mixed-size/AMO_X0_2_THREAD/2+2W+poprl+porlrl+NEW.litmus',
                                 'input/litmus/non-mixed-size/AMO_X0_2_THREAD/S+porlrl+poprl+NEW.litmus',
                                #  'input/litmus/non-mixed-size/RELAX/Rfi/MP+rfi-addr+ctrl-rfi-ctrlfenceis.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-data+data-rfi-ctrlfencei.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-addr+data-rfi-ctrlfencei.litmus',
                                # 'input/litmus/non-mixed-size/HAND/ISA11.litmus',
                                # 'input/litmus/non-mixed-size/AMO_X0_2_THREAD/R+popar+poarp+NEW.litmus',
                                # 'input/litmus/non-mixed-size/CO/CoRR.litmus',
                                # diy7 created in shell manually
                                # 'input/litmus/manual/LB+addr+data-po-ctrl.litmus',

                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-addr.litmus'
                                ] + passed_tests
            rvwmo = RVWMO(plot_enabled=False)

            # 1-3) initialize cur_mm that lacks ppo12
            global_ppos = [ppo_r1, ppo_r2, ppo_r3, ppo_r4, ppo_r5, ppo_r7, ppo_r8,
                        ppo_r9, ppo_r10, ppo_r11, ppo_r12, ppo_r13]
            local_ppos = [p for p in global_ppos if p not in [ppo_r2, ppo_r3, ppo_r12]]
            cur_mm = RVWMO()
            cur_mm.ppo_g = PPO(global_ppos)
            cur_mm.ppo_l = PPO(local_ppos)

            # 1-4) pre validation
            # parallel approach
            # result_list = parallel_validate(rvwmo, cur_mm, litmus_test_suite)
            # failed_litmus_tests, any_ppo_all, validated = validate(rvwmo, cur_mm, litmus_test_suite)
            # print('failed_litmus_tests:', len(failed_litmus_tests))
            print(1)
            cat_file_name = 'riscv-remove-ppo-6.cat'
            patches = synth_by_test_pattern_online_for_one_ppo(rvwmo, cur_mm, global_ppos, litmus_test_suite, cat_file_name)
            print('patches',patches)
            sys.stdout=original_stdout
            f.close()

    def test_diyone7_based_pipeline_for_ppo7(self):
        """
        This is to explore the pattern of ppo7
      """
        with open('output.txt','w') as f:
            original_stdout=sys.stdout
            sys.stdout=f
            config.init()
            config.set_var('reg_size', 64)

            litmus_test_suite = all_litmus_files_sorted_by_time
            litmus_test_suite = [litmus_test for litmus_test in litmus_test_suite if
                                'addr-rfi' in litmus_test or 'data-rfi' in litmus_test][:100]
            passed_tests = []  # all_litmus_files_sorted_by_time[:10]
            # litmus_test_suite = [
            #                      'tests/input/litmus/manual/test_PPO7.litmus',
            #                     ]
            litmus_test_suite = [os.path.join(config.INPUT_DIR,'litmus/manual/test_PPO7.litmus')]
            litmus_test_suite = litmus_test_suite + passed_tests
            rvwmo = RVWMO(plot_enabled=False)

            # 1-3) initialize cur_mm that lacks ppo12
            global_ppos = [ppo_r1, ppo_r2, ppo_r3, ppo_r4, ppo_r5, ppo_r6, ppo_r8,
                        ppo_r9, ppo_r10, ppo_r11, ppo_r12, ppo_r13]
            local_ppos = [p for p in global_ppos if p not in [ppo_r2, ppo_r3, ppo_r12]]
            cur_mm = RVWMO()
            cur_mm.ppo_g = PPO(global_ppos)
            cur_mm.ppo_l = PPO(local_ppos)

            # 1-4) pre validation
            # parallel approach
            # result_list = parallel_validate(rvwmo, cur_mm, litmus_test_suite)
            # failed_litmus_tests, any_ppo_all, validated = validate(rvwmo, cur_mm, litmus_test_suite)
            # print('failed_litmus_tests:', len(failed_litmus_tests))
            print(1)
            cat_file_name = 'riscv-remove-ppo-7.cat'
            patches = synth_by_test_pattern_online_for_one_ppo(rvwmo, cur_mm, global_ppos, litmus_test_suite, cat_file_name)
            print('patches',patches)
            sys.stdout=original_stdout
            f.close()

    def test_diyone7_based_pipeline_for_ppoX(self):
        """
        This is to explore the pattern of ppo7
      """
        with open('output.txt', 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            config.init()
            config.set_var('reg_size', 64)

            litmus_test_suite = all_litmus_files_sorted_by_time
            litmus_test_suite = [litmus_test for litmus_test in litmus_test_suite if
                                 'addr-rfi' in litmus_test or 'data-rfi' in litmus_test][:100]
            passed_tests = []  # all_litmus_files_sorted_by_time[:10]
            # litmus_test_suite = [
            #                      'tests/input/litmus/manual/test_PPO7.litmus',
            #                     ]
            litmus_test_suite = [os.path.join(config.INPUT_DIR, 'litmus/non-mixed-size/ATOMICS/RELAX/PodRWPX/LB+addr+popx.litmus')]
            litmus_test_suite = litmus_test_suite + passed_tests
            rvwmo = RVWMOX(plot_enabled=False)


            cur_mm = get_rvwmo_modelx()

            cat_file_name = os.path.join(config.CAT_DIR,'riscv-X.cat')
            patches,_ = synth_by_test_pattern_online(rvwmo, cur_mm, litmus_test_suite,
                                                               cat_file_name)
            print('patches', patches)
            sys.stdout = original_stdout
            f.close()

    def test_diyone7_based_pipeline_for_ppo8(self):
        """
        This is to explore the pattern of ppo8
      """
        with open('output.txt','w') as f:
            original_stdout=sys.stdout
            sys.stdout=f
            config.init()
            config.set_var('reg_size', 64)

            litmus_test_suite = all_litmus_files_sorted_by_time
            litmus_test_suite = [litmus_test for litmus_test in litmus_test_suite if
                                'addr-rfi' in litmus_test or 'data-rfi' in litmus_test][:100]
            passed_tests = []  # all_litmus_files_sorted_by_time[:10]
            litmus_test_suite = [
                                 'input/litmus/manual/test_PPO8.litmus',
                                ] + passed_tests
            rvwmo = RVWMO(plot_enabled=False)

            # 1-3) initialize cur_mm that lacks ppo12
            global_ppos = [ppo_r1, ppo_r2, ppo_r3, ppo_r4, ppo_r5, ppo_r6, ppo_r7,
                        ppo_r9, ppo_r10, ppo_r11, ppo_r12, ppo_r13]
            local_ppos = [p for p in global_ppos if p not in [ppo_r2, ppo_r3, ppo_r12]]
            cur_mm = RVWMO()
            cur_mm.ppo_g = PPO(global_ppos)
            cur_mm.ppo_l = PPO(local_ppos)

            # 1-4) pre validation
            # parallel approach
            # result_list = parallel_validate(rvwmo, cur_mm, litmus_test_suite)
            # failed_litmus_tests, any_ppo_all, validated = validate(rvwmo, cur_mm, litmus_test_suite)
            # print('failed_litmus_tests:', len(failed_litmus_tests))
            print(1)
            cat_file_name = 'riscv-remove-ppo-8.cat'
            patches = synth_by_test_pattern_online_for_one_ppo(rvwmo, cur_mm, global_ppos, litmus_test_suite, cat_file_name)
            print('patches',patches)
            sys.stdout=original_stdout
            f.close()

    def test_diyone7_based_pipeline_for_ppo9(self):
        """
        This is to explore the pattern of ppo9
        """
        with open('output.txt','w') as f:
            original_stdout=sys.stdout
            sys.stdout=f
            config.init()
            config.set_var('reg_size', 64)

            litmus_test_suite = all_litmus_files_sorted_by_time
            litmus_test_suite = [litmus_test for litmus_test in litmus_test_suite if
                                'addr-rfi' in litmus_test or 'data-rfi' in litmus_test][:100]
            passed_tests = []  # all_litmus_files_sorted_by_time[:10]
            litmus_test_suite = [
                                #  'input/litmus/non-mixed-size/RELAX/Rfi/MP+rfi-addr+ctrl-rfi-ctrlfenceis.litmus',
                                 'input/litmus/non-mixed-size/SAFE/ISA2+pos+ctrl+addr.litmus',
                                 'input/litmus/non-mixed-size/RELAX/Rfi/MP+rfi-addr+ctrl-rfi-ctrlfenceis.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-data+data-rfi-ctrlfencei.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-addr+data-rfi-ctrlfencei.litmus',
                                # 'input/litmus/non-mixed-size/HAND/ISA11.litmus',
                                # 'input/litmus/non-mixed-size/AMO_X0_2_THREAD/R+popar+poarp+NEW.litmus',
                                # 'input/litmus/non-mixed-size/CO/CoRR.litmus',
                                # diy7 created in shell manually
                                # 'input/litmus/manual/LB+addr+data-po-ctrl.litmus',

                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-addr.litmus'
                                ] + passed_tests
            rvwmo = RVWMO(plot_enabled=False)

            # 1-3) initialize cur_mm that lacks ppo12
            global_ppos = [ppo_r1, ppo_r2, ppo_r3, ppo_r4, ppo_r5, ppo_r6, ppo_r7, ppo_r8,
                        ppo_r10, ppo_r11, ppo_r12, ppo_r13]
            local_ppos = [p for p in global_ppos if p not in [ppo_r2, ppo_r3, ppo_r12]]
            cur_mm = RVWMO()
            cur_mm.ppo_g = PPO(global_ppos)
            cur_mm.ppo_l = PPO(local_ppos)

            # 1-4) pre validation
            # parallel approach
            # result_list = parallel_validate(rvwmo, cur_mm, litmus_test_suite)
            # failed_litmus_tests, any_ppo_all, validated = validate(rvwmo, cur_mm, litmus_test_suite)
            # print('failed_litmus_tests:', len(failed_litmus_tests))
            print(1)
            cat_file_name = 'riscv-remove-ppo-9.cat'
            patches = synth_by_test_pattern_online_for_one_ppo(rvwmo, cur_mm, global_ppos, litmus_test_suite, cat_file_name)
            print('patches',patches)
            sys.stdout=original_stdout
            f.close()

    def test_diyone7_based_pipeline_for_ppo10(self):
        """
        This is to explore the pattern of ppo10
        """
        with open('output.txt','w') as f:
            original_stdout=sys.stdout
            sys.stdout=f
            config.init()
            config.set_var('reg_size', 64)

            litmus_test_suite = all_litmus_files_sorted_by_time
            litmus_test_suite = [litmus_test for litmus_test in litmus_test_suite if
                                'addr-rfi' in litmus_test or 'data-rfi' in litmus_test][:100]
            passed_tests = []  # all_litmus_files_sorted_by_time[:10]
            litmus_test_suite = [
                                #  'input/litmus/non-mixed-size/RELAX/Rfi/MP+rfi-addr+ctrl-rfi-ctrlfenceis.litmus',
                                 'input/litmus/non-mixed-size/RELAX/Coi-Rfi/LB+data+ctrl-wsi-rfi-data.litmus',
                                #  'input/litmus/non-mixed-size/RELAX/Rfi/MP+rfi-addr+ctrl-rfi-ctrlfenceis.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-data+data-rfi-ctrlfencei.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-addr+data-rfi-ctrlfencei.litmus',
                                # 'input/litmus/non-mixed-size/HAND/ISA11.litmus',
                                # 'input/litmus/non-mixed-size/AMO_X0_2_THREAD/R+popar+poarp+NEW.litmus',
                                # 'input/litmus/non-mixed-size/CO/CoRR.litmus',
                                # diy7 created in shell manually
                                # 'input/litmus/manual/LB+addr+data-po-ctrl.litmus',

                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-addr.litmus'
                                ] + passed_tests
            rvwmo = RVWMO(plot_enabled=False)

            # 1-3) initialize cur_mm that lacks ppo12
            global_ppos = [ppo_r1, ppo_r2, ppo_r3, ppo_r4, ppo_r5, ppo_r6, ppo_r7, ppo_r8,
                        ppo_r9, ppo_r11, ppo_r12, ppo_r13]
            local_ppos = [p for p in global_ppos if p not in [ppo_r2, ppo_r3, ppo_r12]]
            cur_mm = RVWMO()
            cur_mm.ppo_g = PPO(global_ppos)
            cur_mm.ppo_l = PPO(local_ppos)

            # 1-4) pre validation
            # parallel approach
            # result_list = parallel_validate(rvwmo, cur_mm, litmus_test_suite)
            # failed_litmus_tests, any_ppo_all, validated = validate(rvwmo, cur_mm, litmus_test_suite)
            # print('failed_litmus_tests:', len(failed_litmus_tests))
            print(1)
            cat_file_name = 'riscv-remove-ppo-10.cat'
            patches = synth_by_test_pattern_online_for_one_ppo(rvwmo, cur_mm, global_ppos, litmus_test_suite, cat_file_name)
            print('patches',patches)
            sys.stdout=original_stdout
            f.close()

    def test_diyone7_based_pipeline_for_ppo11(self):
        """
        This is to explore the pattern of ppo11
        """
        with open('output.txt','w') as f:
            original_stdout=sys.stdout
            sys.stdout=f
            config.init()
            config.set_var('reg_size', 64)

            litmus_test_suite = all_litmus_files_sorted_by_time
            litmus_test_suite = [litmus_test for litmus_test in litmus_test_suite if
                                'addr-rfi' in litmus_test or 'data-rfi' in litmus_test][:100]
            passed_tests = []  # all_litmus_files_sorted_by_time[:10]
            litmus_test_suite = [
                                #  'input/litmus/non-mixed-size/RELAX/Rfi/MP+rfi-addr+ctrl-rfi-ctrlfenceis.litmus',
                                 'input/litmus/non-mixed-size/RELAX/Coi-Rfi/LB+ctrl+addr-wsi-rfi-ctrl.litmus',
                                #  'input/litmus/non-mixed-size/RELAX/Rfi/MP+rfi-addr+ctrl-rfi-ctrlfenceis.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-data+data-rfi-ctrlfencei.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-addr+data-rfi-ctrlfencei.litmus',
                                # 'input/litmus/non-mixed-size/HAND/ISA11.litmus',
                                # 'input/litmus/non-mixed-size/AMO_X0_2_THREAD/R+popar+poarp+NEW.litmus',
                                # 'input/litmus/non-mixed-size/CO/CoRR.litmus',
                                # diy7 created in shell manually
                                # 'input/litmus/manual/LB+addr+data-po-ctrl.litmus',

                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-addr.litmus'
                                ] + passed_tests
            rvwmo = RVWMO(plot_enabled=False)

            # 1-3) initialize cur_mm that lacks ppo12
            global_ppos = [ppo_r1, ppo_r2, ppo_r3, ppo_r4, ppo_r5, ppo_r6, ppo_r7, ppo_r8,
                        ppo_r9, ppo_r10, ppo_r12, ppo_r13]
            local_ppos = [p for p in global_ppos if p not in [ppo_r2, ppo_r3, ppo_r12]]
            cur_mm = RVWMO()
            cur_mm.ppo_g = PPO(global_ppos)
            cur_mm.ppo_l = PPO(local_ppos)

            # 1-4) pre validation
            # parallel approach
            # result_list = parallel_validate(rvwmo, cur_mm, litmus_test_suite)
            # failed_litmus_tests, any_ppo_all, validated = validate(rvwmo, cur_mm, litmus_test_suite)
            # print('failed_litmus_tests:', len(failed_litmus_tests))
            print(1)
            cat_file_name = 'riscv-remove-ppo-11.cat'
            patches = synth_by_test_pattern_online_for_one_ppo(rvwmo, cur_mm, global_ppos, litmus_test_suite, cat_file_name)
            print('patches',patches)
            sys.stdout=original_stdout
            f.close()

    def test_diyone7_based_pipeline_for_ppo91011(self):
        """
        This is to explore the pattern of ppo91011
        """
        with open('output.txt','w') as f:
            original_stdout=sys.stdout
            sys.stdout=f
            config.init()
            config.set_var('reg_size', 64)

            litmus_test_suite = all_litmus_files_sorted_by_time
            litmus_test_suite = [litmus_test for litmus_test in litmus_test_suite if
                                'addr-rfi' in litmus_test or 'data-rfi' in litmus_test][:100]
            passed_tests = []  # all_litmus_files_sorted_by_time[:10]
            litmus_test_suite = [
                                #  'input/litmus/non-mixed-size/RELAX/Rfi/MP+rfi-addr+ctrl-rfi-ctrlfenceis.litmus',
                                 'input/litmus/non-mixed-size/RELAX/Coi-Rfi/LB+ctrl+addr-wsi-rfi-ctrl.litmus',
                                 'input/litmus/non-mixed-size/RELAX/Coi-Rfi/LB+data+ctrl-wsi-rfi-data.litmus',
                                 'input/litmus/non-mixed-size/SAFE/ISA2+pos+ctrl+addr.litmus',
                                 'input/litmus/non-mixed-size/RELAX/Rfi/MP+rfi-addr+ctrl-rfi-ctrlfenceis.litmus',
                                #  'input/litmus/non-mixed-size/RELAX/Rfi/MP+rfi-addr+ctrl-rfi-ctrlfenceis.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-data+data-rfi-ctrlfencei.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-addr+data-rfi-ctrlfencei.litmus',
                                # 'input/litmus/non-mixed-size/HAND/ISA11.litmus',
                                # 'input/litmus/non-mixed-size/AMO_X0_2_THREAD/R+popar+poarp+NEW.litmus',
                                # 'input/litmus/non-mixed-size/CO/CoRR.litmus',
                                # diy7 created in shell manually
                                # 'input/litmus/manual/LB+addr+data-po-ctrl.litmus',

                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-addr.litmus'
                                ] + passed_tests
            rvwmo = RVWMO(plot_enabled=False)

            # 1-3) initialize cur_mm that lacks ppo12
            global_ppos = [ppo_r1, ppo_r2, ppo_r3, ppo_r4, ppo_r5, ppo_r6, ppo_r7, ppo_r8,
                           ppo_r12, ppo_r13]
            local_ppos = [p for p in global_ppos if p not in [ppo_r2, ppo_r3, ppo_r12]]
            cur_mm = RVWMO()
            cur_mm.ppo_g = PPO(global_ppos)
            cur_mm.ppo_l = PPO(local_ppos)

            # 1-4) pre validation
            # parallel approach
            # result_list = parallel_validate(rvwmo, cur_mm, litmus_test_suite)
            # failed_litmus_tests, any_ppo_all, validated = validate(rvwmo, cur_mm, litmus_test_suite)
            # print('failed_litmus_tests:', len(failed_litmus_tests))
            print(1)
            patches = synth_by_test_pattern_online_for_one_ppo(rvwmo, cur_mm, global_ppos, litmus_test_suite)
            print('patches',patches)
            sys.stdout=original_stdout
            f.close()
    
    def test_diyone7_based_pipeline_for_ppo12(self):
        """
        This is to explore the pattern of ppo12
      """
        with open('output.txt','w') as f:
            original_stdout=sys.stdout
            sys.stdout=f
            config.init()
            config.set_var('reg_size', 64)

            litmus_test_suite = all_litmus_files_sorted_by_time
            litmus_test_suite = [litmus_test for litmus_test in litmus_test_suite if
                                'addr-rfi' in litmus_test or 'data-rfi' in litmus_test][:100]
            passed_tests = []  # all_litmus_files_sorted_by_time[:10]
            litmus_test_suite = [
                                'input/litmus/non-mixed-size/HAND/PPOAA.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr+data-rfi-data.litmus',
                                #  'input/litmus/non-mixed-size/RELAX/Rfi/MP+rfi-addr+ctrl-rfi-ctrlfenceis.litmus',
                                #  'input/litmus/non-mixed-size/RELAX/Fri-Rfi/LB+data+addr-fri-rfi-addr.litmus',
                                #  'input/litmus/non-mixed-size/RELAX/Coi-Rfi/LB+data+addr-wsi-rfi-ctrl.litmus',
                                #  'input/litmus/non-mixed-size/RELAX/Rfi/MP+rfi-addr+ctrl-rfi-ctrlfenceis.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-data+data-rfi-ctrlfencei.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-addr+data-rfi-ctrlfencei.litmus',
                                # 'input/litmus/non-mixed-size/HAND/ISA11.litmus',
                                # 'input/litmus/non-mixed-size/AMO_X0_2_THREAD/R+popar+poarp+NEW.litmus',
                                # 'input/litmus/non-mixed-size/CO/CoRR.litmus',
                                # diy7 created in shell manually
                                # 'input/litmus/manual/LB+addr+data-po-ctrl.litmus',

                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-data.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-addr.litmus',
                                # 'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-addr.litmus'
                                ] + passed_tests
            rvwmo = RVWMO(plot_enabled=False)

            # 1-3) initialize cur_mm that lacks ppo12
            global_ppos = [ppo_r1, ppo_r2, ppo_r3, ppo_r4, ppo_r5, ppo_r6, ppo_r7, ppo_r8,
                        ppo_r9, ppo_r10, ppo_r11, ppo_r13]
            local_ppos = [p for p in global_ppos if p not in [ppo_r2, ppo_r3, ppo_r12]]
            cur_mm = RVWMO()
            cur_mm.ppo_g = PPO(global_ppos)
            cur_mm.ppo_l = PPO(local_ppos)

            # 1-4) pre validation
            # parallel approach
            # result_list = parallel_validate(rvwmo, cur_mm, litmus_test_suite)
            # failed_litmus_tests, any_ppo_all, validated = validate(rvwmo, cur_mm, litmus_test_suite)
            # print('failed_litmus_tests:', len(failed_litmus_tests))
            cat_file_name = 'riscv-remove-ppo-12.cat'
            print(1)
            patches = synth_by_test_pattern_online_for_one_ppo(rvwmo, cur_mm, global_ppos, litmus_test_suite, cat_file_name)
            print('patches',patches)
            sys.stdout=original_stdout
            f.close()

    def test_diyone7_based_pipeline_for_ppo13(self):
        """
        This is to explore the pattern of ppo13
      """
        with open('output.txt','w') as f:
            original_stdout=sys.stdout
            sys.stdout=f
            config.init()
            config.set_var('reg_size', 64)

            litmus_test_suite = all_litmus_files_sorted_by_time
            litmus_test_suite = [litmus_test for litmus_test in litmus_test_suite if
                                'addr-rfi' in litmus_test or 'data-rfi' in litmus_test][:100]
            passed_tests = []  # all_litmus_files_sorted_by_time[:10]
            litmus_test_suite = [
                                 'input/litmus/non-mixed-size/RELAX/Coi-Rfi/LB+data+addr-wsi-rfi-ctrl.litmus',
                                #  'input/litmus/non-mixed-size/RELAX/Fri-Rfi/LB+data+addr-fri-rfi-addr.litmus'
                                ] + passed_tests
            rvwmo = RVWMO(plot_enabled=False)

            # 1-3) initialize cur_mm that lacks ppo12
            global_ppos = [ppo_r1, ppo_r2, ppo_r3, ppo_r4, ppo_r5, ppo_r6, ppo_r7, ppo_r8,
                        ppo_r9, ppo_r10, ppo_r11, ppo_r12]
            local_ppos = [p for p in global_ppos if p not in [ppo_r2, ppo_r3, ppo_r12]]
            cur_mm = RVWMO()
            cur_mm.ppo_g = PPO(global_ppos)
            cur_mm.ppo_l = PPO(local_ppos)

            # 1-4) pre validation
            # parallel approach
            # result_list = parallel_validate(rvwmo, cur_mm, litmus_test_suite)
            # failed_litmus_tests, any_ppo_all, validated = validate(rvwmo, cur_mm, litmus_test_suite)
            # print('failed_litmus_tests:', len(failed_litmus_tests))
            print(1)
            cat_file_name = 'riscv-remove-ppo-13.cat'
            patches = synth_by_test_pattern_online_for_one_ppo(rvwmo, cur_mm, global_ppos, litmus_test_suite, cat_file_name)
            print('patches',patches)
            sys.stdout=original_stdout
            f.close()


    def test_synthesis_pipeline_for_ppo12(self):
        # 1-1) prepare test suite
        config.init()
        config.set_var('reg_size', 64)
        litmus_names = ['MP+fence.rw.w+addr-rfi-addr.litmus']
        # ['MP+fence.rw.w+addr-rfi-addr.litmus', 'LB+ctrl+data-rfi-ctrl.litmus']
        # ['PPOLDSTLD01.litmus', 'ISA14+TER.litmus']

        litmus_test_suite = get_litmus_by_policy(GetLitmusPolicy,{'name_list':litmus_names})
        # + all_litmus_files[:100]

        # FIXME: PPOLDSTLD01.litmus'(index:1686), 'ISA14+TER.litmus' (index: 1684) are not exposed when running
        #  all_litmus_files[1000:2000]. The reason should be further explored.
        # for i, litmus in enumerate(all_litmus_files):
        #     if 'ISA14+TER.litmus' in litmus:
        #         pass

        # litmus_test_suite = all_litmus_files[3000:]
        # # time-consuming litmus tests are often adjacent in the list, so use random.shuffle to shuffle the list
        # random.shuffle(litmus_test_suite)
        """
        1~1000: ~2min
        1000~2000: some litmus tests are very time-consuming (more than 200s)  32min23s
        (['input/litmus/non-mixed-size/HAND/PPOAA.litmus', 'input/litmus/non-mixed-size/HAND/PPOLDSTLD02.litmus'],
        [], False)]
        2000~2500: 9min10s
        2500~3000: 15min25s
        [(['input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-data+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-addr+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+ctrl+data-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-addrs.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+rfi-ctrl+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrlfencei+ctrlfencei-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrl+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+rfi-data+data-rfi-ctrl.litmus'], [], False), 
        (['input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+rfi-addr+data-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr+data-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrls.litmus'], [], False), 
        (['input/litmus/non-mixed-size/RELAX/Rfi/LB+data+data-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-addr+data-rfi-data.litmus'], [], False), 
        (['input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-addr+ctrl-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-ctrl+data-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+rfi-ctrl+data-rfi-data.litmus'], [], False), 
        (['input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-data+data-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-addr+ctrl-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/MP+fence.rw.w+data-rfi-addr.litmus'], [], False), 
        (['input/litmus/non-mixed-size/RELAX/Rfi/MP+fence.rw.w+addr-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-addr+ctrl-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-data+ctrlfencei-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+ctrl+data-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data+data-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-addr+ctrlfencei-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-data+data-rfi-ctrlfencei.litmus'], [], False), 
        (['input/litmus/non-mixed-size/RELAX/Rfi/LB+data+data-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+ctrl+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-ctrlfencei+data-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+rfi-ctrl+data-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-addr+data-rfi-ctrlfencei.litmus'], [], False), 
        (['input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.r.rw+data-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+rfi-data+data-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-data+ctrl-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+ctrlfencei+data-rfi-ctrl.litmus'], [], False)]
        3000~3536: 18min30s
        [(['input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrlfencei+ctrl-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.r.rw+data-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-data+ctrl-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.rw+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrlfencei+ctrl-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+ctrlfencei+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+rfi-addr+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-data+ctrlfencei-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+rfi-ctrlfencei+data-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-ctrlfencei+data-rfi-addr.litmus'], [], False), 
        (['input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-addr+data-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-data+data-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+rfi-addr+data-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-addr+ctrlfencei-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.rw+data-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.rw+data-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/MP+fence.rw.rw+addr-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrl+ctrlfencei-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+ctrlfencei+data-rfi-addr.litmus'], [], False), 
        (['input/litmus/non-mixed-size/RELAX/Rfi/LB+addr+data-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-addr+data-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/MP+fence.w.w+addr-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-addr+data-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-data+data-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.rw+data-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrl+ctrlfencei-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/MP+fence.w.w+data-rfi-addr.litmus'], [], False), 
        (['input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrlfencei+ctrl-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-data+ctrl-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-addr+ctrlfencei-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrl+ctrl-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.r.rw+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrlfencei+ctrlfencei-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.rw+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-ctrlfencei+data-rfi-ctrlfencei.litmus'], [], False), 
        (['input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrl+ctrl-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrl+ctrl-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.r.rw+data-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-ctrl+data-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+ctrl+data-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-ctrl+data-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+rfi-addr+data-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+rfi-data+data-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+ctrlfencei+data-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrlfencei+ctrlfencei-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+rfi-ctrl+data-rfi-addr.litmus'], [], False), 
        (['input/litmus/non-mixed-size/RELAX/Rfi/LB+data+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrlfencei+ctrlfencei-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-addr+ctrl-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-data+ctrl-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrl+ctrl-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr+data-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+rfi-ctrlfencei+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-addr+data-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-data+data-rfi-ctrl.litmus'], [], False), 
        (['input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-ctrlfencei+data-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.rw+data-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+rfi-data+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-addr+ctrlfencei-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+fence.rw.w+data-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrlfencei+ctrl-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+addr-rfi-ctrl+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.rw+data-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.w+data-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrl+ctrlfencei-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrlfenceis.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+fence.w.w+data-rfi-ctrl.litmus'], [], False), 
        (['input/litmus/non-mixed-size/RELAX/Rfi/S+fence.rw.rw+data-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-ctrl+ctrlfencei-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/MP+fence.rw.rw+data-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-data+ctrlfencei-rfi-ctrlfencei.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+rfi-ctrlfencei+data-rfi-ctrl.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/S+rfi-ctrlfencei+data-rfi-addr.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-data+ctrlfencei-rfi-data.litmus', 
        'input/litmus/non-mixed-size/RELAX/Rfi/LB+data-rfi-datas.litmus'], [], False)]
        """

        # 1-2) initialize the standard model
        rvwmo = RVWMO(plot_enabled=False)

        # 1-3) initialize cur_mm that lacks ppo4
        global_ppos = [ppo_r1, ppo_r2, ppo_r3, ppo_r4, ppo_r5, ppo_r6, ppo_r7, ppo_r8, ppo_r9,
                       ppo_r10, ppo_r11, ppo_r13]
        local_ppos = [p for p in global_ppos if p not in [ppo_r2, ppo_r3, ppo_r12]]
        cur_mm = RVWMO()
        cur_mm.ppo_g = PPO(global_ppos)
        cur_mm.ppo_l = PPO(local_ppos)

        # 1-4) pre validation
        # parallel approach
        result_list = parallel_validate(rvwmo, cur_mm, litmus_test_suite)
        print(result_list)
        return

        # sequential approach
        # failed_litmus_cnt, any_ppo_all, validated = validate(rvwmo, cur_mm, litmus_test_suite)
        # print(
        #     f'pre-validation result (tests size: {len(litmus_test_suite)}): {validated}, failed_litmus_cnt: '
        #     f'{failed_litmus_cnt}')
        # return

        # 2) generate patch.
        g = Grammar()
        spec = Spec()
        syntax_valid_patch_list = synth.top_down_enum_search(g, spec, 100)
        ppo_candidate = '[M];(addr|data);[W];rfi;[R]'
        patch = parse_to_gnode_tree(ppo_candidate)
        syntax_valid_patch_list.insert(0, patch)
        for patch in syntax_valid_patch_list:
            python_func_string = transform.transform(patch)
            print(f'patch: {patch}, python_func_string: {python_func_string}')

            # CAUTION: must has globals() para, otherwise the ppo_candidate_func won't be recognized
            exec(python_func_string, globals())

            cur_mm.ppo_g = PPO(global_ppos + [ppo_candidate_func])  # add to globals
            if not ('rf' in str(patch) or 'rsw' in str(patch) or 'co' in str(patch)):
                cur_mm.ppo_l = PPO(local_ppos + [ppo_candidate_func])  # add to locals if not include
                # 'rf' 'rsw' 'co'

            # 3) validation
            states_compare, validated = validate(rvwmo, cur_mm, litmus_test_suite, True)
            print(f'the patch is validated (tests size: {len(litmus_test_suite)}): {validated}')

            if validated:
                break
