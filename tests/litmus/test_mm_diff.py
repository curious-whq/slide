
#
"""Test for Litmus"""
from typing import List
#pytest -s analysis/test_mm.py::TestMemoryModel::test_MP_fence_w_w_fri_rfi_addr
import pytest
import sys
import os
sys.path.append("../src")
from src.slide import config
from src.slide.analysis.model import *
from src.slide.comp.parse_result import parse_herd_log,parse_chip_log
from src.slide.litmus import parse_litmus
from src.slide.utils.file_util import *
from datetime import datetime
from src.slide.synth.ppo_dict import *
all_litmus_files = list_files('../input/litmus', '.litmus')
output_dir = f'{os.getcwd()}/output/'

exceptional_aqrl = read_file('../input/herd/exceptional_aq_rl.txt').split('\n')

TARGET_MEMORY_MODEL = 'rvwmo'
NOW_MEMORY_MODEL_PPO1 = 'rvwmo_remove_ppo_1'
NOW_MEMORY_MODEL_PPO3 = 'rvwmo_remove_ppo_3'
NOW_MEMORY_MODEL_PPO4 = 'rvwmo_remove_ppo_4'
NOW_MEMORY_MODEL_PPO6 = 'rvwmo_remove_ppo_6'
NOW_MEMORY_MODEL_PPO7 = 'rvwmo_remove_ppo_7'
NOW_MEMORY_MODEL_PPO8 = 'rvwmo_remove_ppo_8'
NOW_MEMORY_MODEL_PPO9 = 'rvwmo_remove_ppo_9'
NOW_MEMORY_MODEL_PPO10 = 'rvwmo_remove_ppo_10'
NOW_MEMORY_MODEL_PPO11 = 'rvwmo_remove_ppo_11'
NOW_MEMORY_MODEL_PPO12 = 'rvwmo_remove_ppo_12'
NOW_MEMORY_MODEL_PPO13 = 'rvwmo_change_ppo13'
NOW_MEMORY_MODEL_PPO2 = 'rvwmo_remove_ppo_2'
NOW_MEMORY_MODEL_CHANGE_PPO2 = 'rvwmo_change_ppo2'
NOW_MEMORY_MODEL_PPO5 = 'rvwmo_remove_ppo_5'
NOW_MEMORY_MODEL_TSO = 'rvtso'

target_log_path = f'{config.INPUT_DIR}/herd/herd_results_{TARGET_MEMORY_MODEL}.log'

target_herd_logs = {r.name: r.states for r in parse_herd_log(target_log_path)}

class TestMemoryModelDiff:

    # @staticmethod
    # def herd_run_all_valid_litmus(model_file_path, litmus_suite_path, output_path, dependency_dir_path = None):
    #     output_file = output_path
    #     file_util.clear_file(output_file)
    #
    #     index=0
    #     start_time = datetime.now()
    #
    #     for litmus_file in litmus_suite_path:
    #         cmd = f"herd7 {litmus_file} -model {model_file_path}"
    #         if dependency_dir_path:
    #             cmd = f"herd7 {litmus_file} -model {model_file_path} -I {dependency_dir_path}"
    #
    #         output = cmd_util.run_cmd_with_output(cmd).strip()
    #         # print('output',output)
    #         # print(cmd)
    #         index+=1
    #         print(index,cmd)
    #         assert output, f'[ERROR] cmd has empty output: {cmd}'
    #
    #         time_util.update_start_time()
    #         file_util.write_str_to_file(output_file, f'[INFO] cmd: {cmd}, time cost: {time_util.cal_time_cost()}\n\n', True)
    #         file_util.write_str_to_file(output_file, f'{output}\n\n', True)
    #
    #     file_util.write_str_to_file(output_file,
    #                                 f'[INFO] total time cost: {time_util.get_time_cost(start_time, datetime.now())}\n', True)
    #

    # def test_cycle_generator_PPO12_data(self):
    #     ppo = PPOItem([R(), DataD(), W(), Rfi(), R()])
    #     cat_file = 'riscv-remove-ppo-12.cat'
    #     ppo_string = '[M];data;[W];rfi;[R]'
    #     ppo_index = 14
    #     mutateGenerator = MutateGenerator()
    #     mutateGenerator.set_ppo(ppo, cat_file, ppo_string, ppo_index)
    #     start = datetime.now()
    #     litmus_suite = mutateGenerator.generate_cycle_legal()
    #     print(f'time: {datetime.now()-start},the litmus is {litmus_suite}')
    #
    #
    # def test_cycle_generator_PPO12_addr(self):
    #     ppo = PPOItem([R(), AddrD(), W(), Rfi(), R()])
    #     cat_file = 'riscv-remove-ppo-12.cat'
    #     ppo_string = '[M];addr;[W];rfi;[R]'
    #     ppo_index = 15
    #     mutateGenerator = MutateGenerator()
    #     mutateGenerator.set_ppo(ppo, cat_file, ppo_string, ppo_index)
    #     start = datetime.now()
    #     litmus_suite = mutateGenerator.generate_cycle_legal()
    #     print(f'time: {datetime.now()-start},the litmus is {litmus_suite}')
    #
    # def test_cycle_generator_PPO1(self):
    #     input_cat_file_name = 'riscv-add-ppo-20.cat'
    #     input_cat_file = f'{config.CAT_DIR}/change/{input_cat_file_name}'
    #     output_cat_file_name = 'riscv-add-ppo-21.cat'
    #     output_cat_file = f'{config.CAT_DIR}/change/{output_cat_file_name}'
    #     ppo = '[W];po-loc;[W]'
    #     mutateGenerator = MutateGenerator()
    #     ppo = ppo.replace('[','')
    #     ppo = ppo.replace(']','')
    #     ppo_obj = get_ppo_item_by_str(ppo)
    #     mutateGenerator.set_ppo(ppo_obj, ppo, 20, input_cat_file, output_cat_file)
    #     start = datetime.now()
    #     litmus_suite = mutateGenerator.generate_cycle_legal()
    #     print(f'time: {datetime.now()-start},the litmus is {litmus_suite}')
    #
    #
    # def test_cycle_generator_PPO2_po_loc_no_W(self):
    #     ppo = PPOItem([R(), PoLoc(), W(), Rfi(), R()])
    #     cat_file = 'riscv-complete.cat'
    #     ppo_string = '[R];po-loc;[W];rfi;[R]'
    #     ppo_index = 16
    #     mutateGenerator = MutateGenerator()
    #     mutateGenerator.set_ppo(ppo, cat_file, ppo_string, ppo_index)
    #     start = datetime.now()
    #     litmus_suite = mutateGenerator.generate_cycle_legal()
    #     print(f'time: {datetime.now()-start},the litmus is {litmus_suite}')
    #
    #
    # def test_cycle_generator_PPO2_rsw(self):
    #     ppo = PPOItem([R(), Rsw(), R()])
    #     cat_file = 'riscv-complete.cat'
    #     ppo_string = 'rsw'
    #     ppo_index = 17
    #     mutateGenerator = MutateGenerator()
    #     mutateGenerator.set_ppo(ppo, cat_file, ppo_string, ppo_index)
    #     start = datetime.now()
    #     litmus_suite = mutateGenerator.generate_cycle_legal()
    #     print(f'time: {datetime.now()-start},the litmus is {litmus_suite}')
    #
    #
    # def test_cycle_generator_PPO13_any(self):
    #     ppo = PPOItem([R(), AddrD(), W(), Po(), W(), Po() ,R()])
    #     cat_file = 'riscv-complete.cat'
    #     ppo_string = '[R];addr;[R];po-loc;[R]'
    #     ppo_index = 18
    #     mutateGenerator = MutateGenerator()
    #     mutateGenerator.set_ppo(ppo, cat_file, ppo_string, ppo_index)
    #     start = datetime.now()
    #     litmus_suite = mutateGenerator.generate_cycle_legal()
    #     print(f'time: {datetime.now()-start},the litmus is {litmus_suite}')
    #
    #
    # def test_cycle_generator_PPO3_AMO(self):
    #     ppo = PPOItem([AMO(), Rfi(), R()])
    #     cat_file = 'riscv-remove-ppo-3.cat'
    #     ppo_string = '[AMO];rfi;[R]'
    #     ppo_index = 19
    #     mutateGenerator = MutateGenerator()
    #     mutateGenerator.set_ppo(ppo, cat_file, ppo_string, ppo_index)
    #     start = datetime.now()
    #     litmus_suite = mutateGenerator.generate_cycle_legal()
    #     print(f'time: {datetime.now()-start},the litmus is {litmus_suite}')
    #
    # def test_cycle_generator_PPO_any(self):
    #     input_cat_file_name = 'riscv-add-ppo-0.cat'
    #     input_cat_file = f'{config.CAT_DIR}/change/{input_cat_file_name}'
    #     output_cat_file_name = 'riscv-add-ppo-1.cat'
    #     output_cat_file = f'{config.CAT_DIR}/change/{output_cat_file_name}'
    #     ppo = '[R];data;[W];rfi;[R]'
    #     mutateGenerator = MutateGenerator()
    #     ppo = ppo.replace('[','')
    #     ppo = ppo.replace(']','')
    #     ppo_obj = get_ppo_item_by_str(ppo)
    #     mutateGenerator.set_ppo(ppo_obj, ppo, 20, input_cat_file, output_cat_file)
    #     start = datetime.now()
    #     litmus_suite = mutateGenerator.generate_cycle_legal()
    #     print(f'time: {datetime.now()-start},the litmus is {litmus_suite}')
    #
    # def test_cycle_generator_ppo(self):
    #     ppos = []
    #     ppo_file = f'{os.getcwd()}/input/chip_execution_logs/chip_ppo.txt'
    #     ppo_file_out = f'{os.getcwd()}/input/chip_execution_logs/chip_ppo_to_litmus.txt'
    #     with open(ppo_file,'r') as f:
    #         datas = f.readlines()
    #         ppos = [item.strip() for item in datas]
    #         f.close()
    #     litmus_suite = []
    #
    #     with open(ppo_file_out,'a') as f:
    #         for i, ppo in enumerate(ppos):
    #             if i<= 339:
    #                 continue
    #             ppo_string = ppo
    #             ppo = ppo.replace('[','').replace(']','')
    #             ppo = get_ppo_item_by_str(ppo)
    #             input_cat_file = f'{os.getcwd()}/input/CAT/riscv-complete.cat'
    #             ppo_index = 19
    #             new_cat_file_path = create_cat_file(ppo_index+1,[ppo_string],input_cat_file)
    #             mutateGenerator = MutateGenerator()
    #             mutateGenerator.set_ppo(ppo, ppo, 20, input_cat_file, new_cat_file_path)
    #             start = datetime.now()
    #             print(ppo)
    #             litmus_test = mutateGenerator.generate_cycle_legal()
    #             litmus_suite.append(litmus_test)
    #             print(f'time: {datetime.now()-start},the litmus is {litmus_test}')
    #             f.write(ppo_string)
    #             f.write('    ')
    #             f.write(('  ').join(litmus_test))
    #             f.write('\n')
    #
    # def test_diyone7_PPO12(self):
    #
    #     # 0. init the ppo which want to create the litmus test
    #     ppo=PPOItem([R(), DataD(), W(), Rfi(), R()])
    #     mutateGenerator = MutateGenerator()
    #     mutateGenerator.set_ppo(ppo)
    #     litmus_suite = mutateGenerator.generate_cycle_legal()
    #     print(len(litmus_suite))
    #     litmus_suite = litmus_suite[:10]
    #     # litmus_suite = [
    #     #     'Rfe DpDatadW Rfi Fence.rw.rwdRR Fre Fence.rw.rwdWW',
    #     #     'Rfe DpDatadW Rfi DpAddrdR Fre Fence.rw.rwdWW',
    #     # ]
    #     input_cat_file = f'{os.getcwd()}/input/CAT/riscv-remove-ppo-12.cat'
    #     output_cat_file = f'{os.getcwd()}/input/CAT/change/riscv-add-ppo-12.cat'
    #     index = 14
    #     ppo_string = '[M];data;[W];rfi;[R]'
    #
    #
    #     # 1. create the cat file
    #     with open(input_cat_file,'r') as f:
    #         lines = f.readlines()
    #
    #     lines.insert(lines.index('let ppo =\n') + 1, 'r14\n |')
    #     insert_i = -1
    #     for i,line in enumerate(lines):
    #         if 'and r12' in line:
    #             insert_i = i
    #             break
    #     lines.insert( insert_i+1,f'and r{index} = {ppo_string}\n')
    #     with open(output_cat_file,'w') as f:
    #         f.writelines(lines)
    #
    #
    #     # 2. create the litmus test
    #     test_dir = os.path.join(config.OUTPUT_DIR, 'test_ppo')
    #     dir_util.mk_dir_from_dir_path(test_dir)
    #     file_util.rm_files_with_suffix_in_dir(test_dir, '.litmus')
    #
    #     cnt = index
    #     new_test_paths = []
    #     for i, diy_cycle in enumerate(litmus_suite):
    #         new_test_path = os.path.join(test_dir, f"new_test_{i}_{cnt}.litmus")
    #         new_test_name = os.path.join(test_dir, f"new_test_{i}_{cnt}")
    #         # cmd = f"{config.DIYONE7_PATH} -arch RISC-V -name {new_test_name} {diy_cycle}"
    #         cmd = f"diyone7 -arch RISC-V -obs local -name {new_test_name} {diy_cycle}"
    #         print(f"diyone7 cmd: {cmd}")
    #         cmd_util.run_cmd(cmd)
    #         if os.path.exists(new_test_path):
    #             new_test_paths.append(new_test_path)
    #
    #
    #     # 3. run the litmus test at before and after cat in herd
    #     now_log_path_ppo_add = f'{os.getcwd()}/input/herd/herd_result/ppo-add-{index}.log'
    #     now_log_path_ppo_remove = f'{os.getcwd()}/input/herd/herd_result/ppo-remove-{index}.log'
    #
    #     self.herd_run_all_valid_litmus(output_cat_file, new_test_paths, now_log_path_ppo_add, dependency_dir_path = None)
    #     self.herd_run_all_valid_litmus(input_cat_file, new_test_paths, now_log_path_ppo_remove, dependency_dir_path = None)
    #
    #     # 4. Compare the results and find the litmus test with errors
    #
    #     now_herd_logs_ppo_add = {r.name: r.states for r in parse_herd_log(now_log_path_ppo_add)}
    #     now_herd_logs_ppo_remove = {r.name: r.states for r in parse_herd_log(now_log_path_ppo_remove)}
    #
    #     different_litmus_tests=[]
    #     for litmus in now_herd_logs_ppo_add:
    #         if litmus not in now_herd_logs_ppo_remove:
    #             continue
    #         if  now_herd_logs_ppo_remove[litmus]!=now_herd_logs_ppo_add[litmus]:
    #             different_litmus_tests.append(litmus)
    #     print(different_litmus_tests)  # litmus test path list

    def test_tso_Diff(self):
        litmus_suite=[]
        now_log_path_tso = f'{os.getcwd()}/input/herd/herd_results_{NOW_MEMORY_MODEL_TSO}.log'
        now_herd_logs_tso = {r.name: r.states for r in parse_herd_log(now_log_path_tso)}
        print(len(all_litmus_files))
        for litmus in all_litmus_files:
            litmus_file=litmus
            litmus=litmus.split('/')[-1].split('.')[0]
            if litmus not in now_herd_logs_tso or litmus not in target_herd_logs:
                continue
            if now_herd_logs_tso[litmus]!=target_herd_logs[litmus]:
                litmus_suite.append(litmus_file)
        for litmus in litmus_suite:
            print("\'"+litmus+"\',")


    def test_reason_litmus_file(self):
        exceed_two_thread_path = f'{os.getcwd()}/input/chip_execution_logs/exceed_two_threads.log'
        exceed_two_thread_list = []
        with open(exceed_two_thread_path,'r') as f:
            list = f.readlines()
            exceed_two_thread_list = [item.strip() for item in list]
            f.close()
        access_path = f'{os.getcwd()}/input/chip_execution_logs/exceed_4_access.txt'
        access_list = []
        with open(access_path, 'r') as f:
            list = f.readlines()
            access_list = [item.strip() for item in list]
            f.close()


    def test_chip_log(self):
        # chip_log_path = f'{os.getcwd()}/input/chip_execution_logs/banana/chip_log_banana_100000.txt'
        # chip_log = {r.name: r.states for r in parse_chip_log(chip_log_path)}
        # rvwmo_log_path = f'{os.getcwd()}/input/herd/herd_results_rvwmo_banana.log'
        chip_log_path = f'{os.getcwd()}/input/chip_execution_logs/C910/chip_log.txt'
        chip_log = {r.name: r.states for r in parse_chip_log(chip_log_path)}
        rvwmo_log_path = f'{os.getcwd()}/input/herd/herd_results_rvwmo_banana.log'
        rvwmo_log = {r.name: r.states for r in parse_herd_log(rvwmo_log_path)}
        exceed_two_thread_path = f'{os.getcwd()}/input/chip_execution_logs/exceed_two_threads.log'
        exceed_two_thread_list = []
        state_difs= {}
        with open(exceed_two_thread_path,'r') as f:
            list = f.readlines()
            exceed_two_thread_list = [item.strip() for item in list]
            f.close()
        fence_i_path = f'{os.getcwd()}/input/chip_execution_logs/fence_file.txt'
        fence_i_list = []
        with open(fence_i_path,'r') as f:
            list = f.readlines()
            fence_i_list = [item.strip() for item in list]
            f.close()
        access_path = f'{os.getcwd()}/input/chip_execution_logs/exceed_4_access.txt'
        access_list = []
        with open(access_path,'r') as f:
            list = f.readlines()
            access_list = [item.strip() for item in list]
            f.close()
        amo_path = f'{os.getcwd()}/input/chip_execution_logs/amo_file.txt'
        amo_list = []
        with open(amo_path,'r') as f:
            list = f.readlines()
            amo_list = [item.strip() for item in list]
            f.close()
        litmus_list = []
        litmus_state_list = []
        print(len(chip_log.keys()))
        for key in chip_log:
            if key in rvwmo_log:
                if key in exceed_two_thread_list:
                    continue
                if f'{key}.litmus' in fence_i_list:
                    continue
                if key in access_list:
                    continue
                if key in amo_list:
                    continue
                if rvwmo_log[key] != chip_log[key]:
                    if len(rvwmo_log[key]) - len(chip_log[key]) > 0:
                        litmus_list.append(key)
                        litmus_state_list.append(f"{key}, {len(rvwmo_log[key])},   {len(chip_log[key])}")
                        state_dif = len(rvwmo_log[key]) - len(chip_log[key])
                        if state_dif in state_difs:
                            state_difs[state_dif] = state_difs[state_dif] + 1
                        else:
                            state_difs[state_dif] = 1

            # print(key,chip_log[key])
        with open(f'{os.getcwd()}/input/chip_execution_logs/C910/chip_dif_not_exceed_two_100000.txt','w') as f:
            for i,litmus in enumerate(litmus_list):
                f.write(litmus)
                f.write('\n')
        with open(f'{os.getcwd()}/input/chip_execution_logs/C910/chip_dif_state_not_exceed_two_100000.txt','w') as f:
            f.write('rwvmo')
            f.write(',  ')
            f.write('chip\n')
            for litmus_state in litmus_state_list:
                f.write(litmus_state)
                f.write('\n')
            # print(i,litmus)

        with open(f'{os.getcwd()}/input/chip_execution_logs/C910/chip_dif_not_exceed_two_statistics_100000.txt','w') as f:
            f.write('state     num\n')
            for state_dif in state_difs:
                f.write(f'{state_dif},{state_difs[state_dif]}')
                f.write('\n')


    def test_chip_model_diff(self):
        diff_file = f'{os.getcwd()}/input/chip_execution_logs/chip_dif.txt'
        banana_chip_diff_file = f'{os.getcwd()}/input/chip_execution_logs/banana/chip_dif_not_exceed_two_100000.txt'
        C910_chip_diff_file = f'{os.getcwd()}/input/chip_execution_logs/C910/chip_dif_not_exceed_two_100000.txt'

        with open(banana_chip_diff_file, 'r') as f:
            banana_chip_diff_list = f.readlines()
            banana_chip_diff_list = [item.strip() for item in banana_chip_diff_list]
        
        with open(C910_chip_diff_file, 'r') as f:
            C910_chip_diff_list = f.readlines()
            C910_chip_diff_list = [item.strip() for item in C910_chip_diff_list]

        with open(diff_file, 'w') as wf:
            wf.write('banana - C910')
            wf.write('\n')
            for banana_item in banana_chip_diff_list:
                if banana_item not in C910_chip_diff_list:
                    wf.write(banana_item)
                    wf.write('\n')

            wf.write('C910 - banana')
            wf.write('\n')
            for C910_item in C910_chip_diff_list:
                if C910_item not in banana_chip_diff_list: 
                    wf.write(C910_item)
                    wf.write('\n')

    def test_PPO2_Diff(self):
        litmus_suite=[]
        now_log_path_ppo2 = f'{config.INPUT_DIR}/herd/herd_results_{NOW_MEMORY_MODEL_CHANGE_PPO2}.log'
        now_herd_logs_ppo2 = {r.name: r.states for r in parse_herd_log(now_log_path_ppo2)}
        print(len(all_litmus_files))
        for litmus in all_litmus_files:
            litmus_file=litmus
            litmus=litmus.split('/')[-1].split('.')[0]
            if litmus not in now_herd_logs_ppo2 or litmus not in target_herd_logs:
                continue
            if now_herd_logs_ppo2[litmus]!=target_herd_logs[litmus]:
                litmus_suite.append(litmus_file)
        print(litmus_suite)
            # assert now_herd_logs[]    
        # 2023.12.9 22:37: 44s
        # 2023.12.9 23:40: 8s
        # assert run_litmus('3.LB+ctrls') == herd_logs['3.LB+ctrls']
    
    def test_PPO1_Diff(self):
        litmus_suite=[]
        now_log_path_ppo1 = f'{os.getcwd()}/input/herd/herd_results_{NOW_MEMORY_MODEL_PPO1}.log'
        now_herd_logs_ppo1 = {r.name: r.states for r in parse_herd_log(now_log_path_ppo1)}
        print(len(all_litmus_files))
        for litmus in all_litmus_files:
            litmus=litmus.split('/')[-1].split('.')[0]
            if litmus not in now_herd_logs_ppo1 or litmus not in target_herd_logs:
                continue
            if now_herd_logs_ppo1[litmus]!=target_herd_logs[litmus]:
                litmus_suite.append(litmus)
        print(litmus_suite)

    def test_PPO3_Diff(self):
        litmus_suite=[]
        now_log_path_ppo3 = f'{os.getcwd()}/input/herd/herd_results_{NOW_MEMORY_MODEL_PPO3}.log'
        now_herd_logs_ppo3 = {r.name: r.states for r in parse_herd_log(now_log_path_ppo3)}
        print(len(all_litmus_files))
        for litmus in all_litmus_files:
            litmus=litmus.split('/')[-1].split('.')[0]
            if litmus not in now_herd_logs_ppo3 or litmus not in target_herd_logs:
                continue
            if now_herd_logs_ppo3[litmus]!=target_herd_logs[litmus]:
                litmus_suite.append(litmus)
        print(litmus_suite)
    
    def test_PPO3_change_Diff(self):
        litmus_suite=[]
        now_log_path_ppo3 = f'{os.getcwd()}/input/herd/herd_results_rvwmo_change_ppo3.log'
        now_herd_logs_ppo3 = {r.name: r.states for r in parse_herd_log(now_log_path_ppo3)}
        print(len(all_litmus_files))
        for litmus in all_litmus_files:
            litmus=litmus.split('/')[-1].split('.')[0]
            if litmus not in now_herd_logs_ppo3 or litmus not in target_herd_logs:
                continue
            if now_herd_logs_ppo3[litmus]!=target_herd_logs[litmus]:
                litmus_suite.append(litmus)
        print(litmus_suite)
    
    def test_PPO4_Diff(self):
        litmus_suite=[]
        now_log_path_ppo4 = f'{os.getcwd()}/input/herd/herd_results_{NOW_MEMORY_MODEL_PPO4}.log'
        now_herd_logs_ppo4 = {r.name: r.states for r in parse_herd_log(now_log_path_ppo4)}
        print(len(all_litmus_files))
        for litmus in all_litmus_files:
            litmus_file=litmus
            litmus=litmus.split('/')[-1].split('.')[0]
            if litmus not in now_herd_logs_ppo4 or litmus not in target_herd_logs:
                continue
            if now_herd_logs_ppo4[litmus]!=target_herd_logs[litmus]:
                litmus_suite.append(litmus_file)
        for litmus in litmus_suite:
            print("\'"+litmus+"\',")
    
    def test_PPO5_change_Diff(self):
        litmus_suite=[]
        now_log_path_ppo5 = f'{os.getcwd()}/input/herd/herd_results_rvwmo_change_ppo5.log'
        now_herd_logs_ppo5 = {r.name: r.states for r in parse_herd_log(now_log_path_ppo5)}
        print(len(all_litmus_files))
        for litmus in all_litmus_files:
            litmus=litmus.split('/')[-1].split('.')[0]
            if litmus not in now_herd_logs_ppo5 or litmus not in target_herd_logs:
                continue
            if now_herd_logs_ppo5[litmus]!=target_herd_logs[litmus]:
                litmus_suite.append(litmus)
        print(litmus_suite)

    def test_PPO6_Diff(self):
        litmus_suite=[]
        now_log_path_ppo6 = f'{os.getcwd()}/input/herd/herd_results_{NOW_MEMORY_MODEL_PPO6}.log'
        now_herd_logs_ppo6 = {r.name: r.states for r in parse_herd_log(now_log_path_ppo6)}
        print(len(all_litmus_files))
        for litmus in all_litmus_files:
            litmus_file=litmus
            litmus=litmus.split('/')[-1].split('.')[0]
            if litmus not in now_herd_logs_ppo6 or litmus not in target_herd_logs:
                continue
            if now_herd_logs_ppo6[litmus]!=target_herd_logs[litmus]:
                litmus_suite.append(litmus_file)
        for litmus in litmus_suite:
            print("\'"+litmus+"\',")
    
    def test_PPO7_Diff(self):
        litmus_suite=[]
        now_log_path_ppo7 = f'{os.getcwd()}/input/herd/herd_results_{NOW_MEMORY_MODEL_PPO7}.log'
        now_herd_logs_ppo7 = {r.name: r.states for r in parse_herd_log(now_log_path_ppo7)}
        print(len(all_litmus_files))
        for litmus in all_litmus_files:
            litmus_file=litmus
            litmus=litmus.split('/')[-1].split('.')[0]
            if litmus not in now_herd_logs_ppo7 or litmus not in target_herd_logs:
                continue
            if now_herd_logs_ppo7[litmus]!=target_herd_logs[litmus]:
                litmus_suite.append(litmus_file)
        for litmus in litmus_suite:
            print("\'"+litmus+"\',")

    def test_PPO8_Diff(self):
        litmus_suite=[]
        now_log_path_ppo8 = f'{os.getcwd()}/input/herd/herd_results_{NOW_MEMORY_MODEL_PPO8}.log'
        now_herd_logs_ppo8 = {r.name: r.states for r in parse_herd_log(now_log_path_ppo8)}
        print(len(all_litmus_files))
        for litmus in all_litmus_files:
            litmus_file=litmus
            litmus=litmus.split('/')[-1].split('.')[0]
            if litmus not in now_herd_logs_ppo8 or litmus not in target_herd_logs:
                continue
            if now_herd_logs_ppo8[litmus]!=target_herd_logs[litmus]:
                litmus_suite.append(litmus_file)
        for litmus in litmus_suite:
            print("\'"+litmus+"\',")
    
    def test_PPO9_Diff(self):
        litmus_suite=[]
        now_log_path_ppo9 = f'{os.getcwd()}/input/herd/herd_results_{NOW_MEMORY_MODEL_PPO9}.log'
        now_herd_logs_ppo9 = {r.name: r.states for r in parse_herd_log(now_log_path_ppo9)}
        print(len(all_litmus_files))
        for litmus in all_litmus_files:
            litmus_file=litmus
            litmus=litmus.split('/')[-1].split('.')[0]
            if litmus not in now_herd_logs_ppo9 or litmus not in target_herd_logs:
                continue
            if now_herd_logs_ppo9[litmus]!=target_herd_logs[litmus]:
                litmus_suite.append(litmus_file)
        for litmus in litmus_suite:
            print("\'"+litmus+"\',")
    
    def test_PPO10_Diff(self):
        litmus_suite=[]
        now_log_path_ppo10 = f'{os.getcwd()}/input/herd/herd_results_{NOW_MEMORY_MODEL_PPO10}.log'
        now_herd_logs_ppo10 = {r.name: r.states for r in parse_herd_log(now_log_path_ppo10)}
        print(len(all_litmus_files))
        for litmus in all_litmus_files:
            litmus_file=litmus
            litmus=litmus.split('/')[-1].split('.')[0]
            if litmus not in now_herd_logs_ppo10 or litmus not in target_herd_logs:
                continue
            if now_herd_logs_ppo10[litmus]!=target_herd_logs[litmus]:
                litmus_suite.append(litmus_file)
        for litmus in litmus_suite:
            print("\'"+litmus+"\',")
        # print(litmus_suite)
    
    def test_PPO11_Diff(self):
        litmus_suite=[]
        now_log_path_ppo11 = f'{os.getcwd()}/input/herd/herd_results_{NOW_MEMORY_MODEL_PPO11}.log'
        now_herd_logs_ppo11 = {r.name: r.states for r in parse_herd_log(now_log_path_ppo11)}
        print(len(all_litmus_files))
        for litmus in all_litmus_files:
            litmus_file=litmus
            litmus=litmus.split('/')[-1].split('.')[0]
            if litmus not in now_herd_logs_ppo11 or litmus not in target_herd_logs:
                continue
            if now_herd_logs_ppo11[litmus]!=target_herd_logs[litmus]:
                litmus_suite.append(litmus_file)
        for litmus in litmus_suite:
            print("\'"+litmus+"\',")
        # print(litmus_suite)

    def test_PPO12_Diff(self):
        litmus_suite=[]
        now_log_path_ppo12 = f'{os.getcwd()}/input/herd/herd_results_{NOW_MEMORY_MODEL_PPO12}.log'
        now_herd_logs_ppo12 = {r.name: r.states for r in parse_herd_log(now_log_path_ppo12)}
        print(len(all_litmus_files))
        for litmus in all_litmus_files:
            litmus=litmus.split('/')[-1].split('.')[0]
            if litmus not in now_herd_logs_ppo12 or litmus not in target_herd_logs:
                continue
            if now_herd_logs_ppo12[litmus]!=target_herd_logs[litmus]:
                litmus_suite.append(litmus)
        print(litmus_suite)

    def test_PPO12_change_Diff(self):
        litmus_suite=[]
        now_log_path_ppo12 = f'{os.getcwd()}/input/herd/herd_results_rvwmo_change_ppo12.log'
        now_herd_logs_ppo12 = {r.name: r.states for r in parse_herd_log(now_log_path_ppo12)}
        print(len(all_litmus_files))
        for litmus in all_litmus_files:
            litmus=litmus.split('/')[-1].split('.')[0]
            if litmus not in now_herd_logs_ppo12 or litmus not in target_herd_logs:
                continue
            if now_herd_logs_ppo12[litmus]!=target_herd_logs[litmus]:
                litmus_suite.append(litmus)
        print(litmus_suite)

    def test_PPO13_Diff(self):
        litmus_suite=[]
        now_log_path_ppo13 = f'{config.INPUT_DIR}/herd/herd_results_{NOW_MEMORY_MODEL_PPO13}.log'
        now_herd_logs_ppo13 = {r.name: r.states for r in parse_herd_log(now_log_path_ppo13)}
        print(len(all_litmus_files))
        for litmus in all_litmus_files:
            litmus_file=litmus
            litmus=litmus.split('/')[-1].split('.')[0]
            if litmus not in now_herd_logs_ppo13 or litmus not in target_herd_logs:
                continue
            if now_herd_logs_ppo13[litmus]!=target_herd_logs[litmus]:
                litmus_suite.append(litmus_file)
        for litmus in litmus_suite:
            print("\'"+litmus+"\',")
    

    def test_PPO5_Diff(self):
        litmus_suite=[]
        now_log_path_ppo5 = f'{os.getcwd()}/input/herd/herd_results_{NOW_MEMORY_MODEL_PPO5}.log'
        now_herd_logs_ppo5 = {r.name: r.states for r in parse_herd_log(now_log_path_ppo5)}
        print(len(all_litmus_files))
        for litmus in all_litmus_files:
            litmus_file=litmus
            litmus=litmus.split('/')[-1].split('.')[0]
            if litmus not in now_herd_logs_ppo5 or litmus not in target_herd_logs:
                continue
            if now_herd_logs_ppo5[litmus]!=target_herd_logs[litmus]:
                litmus_suite.append(litmus_file)
        for litmus in litmus_suite:
            print("\'"+litmus+"\',")
    