
"""Test for Litmus"""
import os.path
from datetime import datetime
import sys

sys.path.append('../src')
from src.slide.prog.inst import AmoInst, LoadInst, StoreInst
from src.slide import config
from src.slide.utils import dir_util, file_util, cmd_util, time_util
from src.slide.litmus import parse_litmus
from src.slide.utils.file_util import *
cur_dir = dir_util.get_cur_dir(__file__)
input_dir = os.path.abspath(os.path.join(cur_dir, "../input"))

all_litmus_files = file_util.list_files(os.path.join(input_dir, 'litmus/non-mixed-size'), '.litmus')
exceptional_litmus_files = file_util._read_file_to_list_strip(os.path.join(input_dir, 'herd/exceptional_aq_rl.txt'))


class TestHerd:
    def test_herd_run_mem_models(self):
        # for i in range(7,8):
        memory_model_dir = os.path.join(input_dir, 'CAT')
        # memory_model_dir = os.path.join(input_dir,'herd/memory_models')
            # memory_models_dict = {'rvwmo_remove_ppo_'+str(i): os.path.join(memory_model_dir, 'riscv-remove-ppo-'+str(i)+'.cat')}
        memory_models_dict = {
                            #   'sc': os.path.join(memory_model_dir, 'sc.cat'),
                            #   'rvtso': os.path.join(memory_model_dir, 'riscv-tso.cat'),
                            #   'rvwmo': os.path.join(memory_model_dir, 'riscv.cat'),
                            # 'rvwmo_change_ppo12': os.path.join(memory_model_dir, 'riscv-complete_change_ppo12.cat'),
                            # 'rvwmo_change_ppo3': os.path.join(memory_model_dir, 'riscv-complete_change_ppo3.cat'),
                            # 'rvwmo_change_ppo5': os.path.join(memory_model_dir, 'riscv-complete_change_ppo5.cat')
                            'rvwmo_change_ppo2': os.path.join(memory_model_dir, 'riscv-complete_change_ppo2.cat'),
                            # 'rvwmo_banana': os.path.join(memory_model_dir, 'riscv-complete-banana.cat')
                              }
        for model_name, model_file_path in memory_models_dict.items():
            assert os.path.exists(model_file_path)
            self.herd_run_all_valid_litmus(model_name, model_file_path)

    def test_get_litmus_path_list(self):
        output_file = os.path.join(input_dir, f'chip_execution_logs/banana/litmus_list.txt')
        file_util.clear_file(output_file)
        with open(f'{os.getcwd()}/input/chip_execution_logs/banana/chip_dif_not_exceed_two_10000.txt','r') as f:
            lines = f.readlines()
            print(lines)
            for line in lines:
                print(line)
                litmus_name = line.strip()
                file = search_file(litmus_name, 'input/litmus', '.litmus')
                file_util.write_str_to_file(output_file, f'{file}\n', True)


    def test_get_two_thread_litmus(self):
        exceed_two_thread_output_file = os.path.join(input_dir, f'chip_execution_logs/exceed_two_thread.txt')
        file_util.clear_file(exceed_two_thread_output_file)
        access_output_file = os.path.join(input_dir, f'chip_execution_logs/exceed_4_access.txt')
        file_util.clear_file(access_output_file)
        fence_file = os.path.join(input_dir, f'chip_execution_logs/fence_file.txt')
        file_util.clear_file(fence_file)
        amo_file = os.path.join(input_dir, f'chip_execution_logs/amo_file.txt')
        file_util.clear_file(amo_file)
        index=0 
        with open('rwvmo.txt', 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            for litmus_file in all_litmus_files:
                litmus_name = file_util.get_file_name_from_path(litmus_file)
                if litmus_name not in exceptional_litmus_files:
                    content = read_file(litmus_file)
                    litmus =  parse_litmus(content)
                    if (len(litmus.progs)>2):
                        file_util.write_str_to_file(exceed_two_thread_output_file, f'{litmus_name}\n', True)

                    for prog in litmus.progs:
                        count = 0
                        for inst in prog.insts:
                            print(type(inst))
                            if isinstance(inst,AmoInst):
                                file_util.write_line_to_file(amo_file, f'{litmus_name}\n', True)
                            if isinstance(inst,LoadInst) or isinstance(inst,StoreInst) or isinstance(inst,AmoInst):
                                count+=1
                                if count > 4:
                                    file_util.write_str_to_file(access_output_file, f'{litmus_name}\n', True)
                                    break
                            
                else:
                    file_util.write_str_to_file(exceed_two_thread_output_file, f'{litmus_name}\n', True)

        with open('output_fence.i.txt', 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            for litmus_file in all_litmus_files:
                litmus_name = file_util.get_file_name_from_path(litmus_file)
                if 'fencei' in litmus_name or 'fence.i' in litmus_name or '.tso' in litmus_name:
                    file_util.write_str_to_file(fence_file, f'{litmus_name}.litmus\n', True)


    @staticmethod
    def herd_run_all_valid_litmus(model_name, model_file_path, dependency_dir_path = None):
        output_file = os.path.join(input_dir, f'herd/herd_results_{model_name}.log')
        file_util.clear_file(output_file)

        index=0 
        start_time = datetime.now()
        with open('rwvmo.txt', 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            for litmus_file in all_litmus_files:
                litmus_name = file_util.get_file_name_from_path(litmus_file)
                if litmus_name not in exceptional_litmus_files:
                    print(litmus_file)
                    # continue
                    cmd = f"herd7 {litmus_file} -model {model_file_path}"
                    if dependency_dir_path:
                        cmd = f"herd7 {litmus_file} -model {model_file_path} -I {dependency_dir_path}"
                        
                    output = cmd_util.run_cmd_with_output(cmd).strip()
                    # print('output',output)
                    # print(cmd)
                    index+=1
                    print(index,cmd)
                    assert output, f'[ERROR] cmd has empty output: {cmd}'

                    time_util.update_start_time()
                    file_util.write_str_to_file(output_file, f'[INFO] cmd: {cmd}, time cost: {time_util.cal_time_cost()}\n\n', True)
                    file_util.write_str_to_file(output_file, f'{output}\n\n', True)

        file_util.write_str_to_file(output_file,
                                    f'[INFO] total time cost: {time_util.get_time_cost(start_time, datetime.now())}\n', True)
