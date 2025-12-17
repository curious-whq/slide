import os
import csv
import subprocess
import sys
from collections import defaultdict
from itertools import product

from numpy.testing.print_coercion_tables import print_new_cast_table

from src.slide import config
from src.slide.comp.parse_result import parse_herd_log
from src.slide.litmus import parse_litmus
from src.slide.litmus.load_litmus import get_litmus_by_policy, GetLitmusPolicy
from src.slide.utils.file_util import read_file
from src.slide.utils.herd_util import herd_run_all_valid_litmus

class TestMemSynthFormat():
    def __init__(self, path_list):
        self.path_list = path_list


    def filter_inst_type(self, path_list):
        filter_litmus_list = []
        litmus_dict = {}
        # names = ['li','sw','sc.w','sc.w.aq','sc.w.rl','sc.w.aq.rl',
        #          'amoswap.w','amoswap.w.aq','amoswap.w.rl','amoswap.w.aq.rl',
        #          'lw','lr.w','lr.w.aq','lr.w.rl','lr.w.aq.rl',
        #          'xor','add','ori','fence']
        names = ['li','sw',
                 'lw',
                 'xor','add','ori','fence']
        regs_name = ['sc.w','sc.w.aq','sc.w.rl','sc.w.aq.rl',
                     'amoswap.w','amoswap.w.aq','amoswap.w.rl','amoswap.w.aq.rl',
                 'lw','lr.w','lr.w.aq','lr.w.rl','lr.w.aq.rl']
        litmus_dir = os.path.join(config.OUTPUT_DIR, 'litmus_align_memsynth')

        for path in path_list:
            regs = []
            litmus = parse_litmus(read_file(path))
            flag = True
            for i,prog in enumerate(litmus.progs):
                for inst in prog.insts:
                    name = str(inst.name).strip()
                    if name not in names:
                        # print(name)
                        flag = False
                    elif name in regs_name:
                        if str(inst.rd) == 'x0':
                            continue
                        regs.append(f'{i}:{inst.rd}')
            if flag:
                # regs.extend(litmus.vars)
                # state_str = ' /\\ '.join(f'{reg}=1' for reg in regs)
                # new_line = f'exists ( {state_str} )'
                # litmus_name = path.split('/')[-1].split('.litmus')[0]
                # with open(path, 'r') as f:
                #     lines = f.readlines()
                # cut_index = next((i for i, line in enumerate(lines) if 'exists' in line), len(lines))
                # new_lines = lines[:cut_index]
                # print('old line')
                # print(lines[cut_index:cut_index+2])
                # print('new line')
                # print(new_line)
                # new_lines.append(new_line)
                # new_path = os.path.join(litmus_dir, f'{litmus_name}.litmus')
                # with open(new_path, 'w') as f:
                #     f.writelines(new_lines)
                # filter_litmus_list.append(new_path)
                filter_litmus_list.append(path)
                litmus_dict[path] = regs
        for path in litmus_dict:
            print('path', path, litmus_dict[path])
        return filter_litmus_list, litmus_dict

    def filter_dif_litmus(self, path_list):
        dif_log = os.path.join(config.HERD_LOG_DIR_PATH, 'dif_litmus.log')
        with open(dif_log, 'r') as f:
            litmus_suite = f.readlines()
            litmus_suite = [litmus.strip() for litmus in litmus_suite]
        filter_litmus_list = []
        for path in path_list:
            litmus_name = path.split('/')[-1].split('.')[0]
            if litmus_name in litmus_suite:
                filter_litmus_list.append(path)

        return filter_litmus_list
    def filter_sep_litmus(self, path_list):
        sep_litmus = [
            'ISA-MP-DEP-SUCCESS-SWAP',
            'ISA-MP-DEP-SUCCESS-SWAP-SIMPLE',
            'AMO-FENCE',
            'ISA03+SB01',
            'ISA14'
        ]

        new_path_list = []
        for path in path_list:
            litmus_name = path.split('/')[-1].split('.')[0]
            if litmus_name not in sep_litmus:
                new_path_list.append(path)
        return new_path_list

    def filter_litmus_list(self):

        path_list = self.path_list
        # print(len(path_list))
        path_list = self.filter_dif_litmus(path_list)
        path_list = self.filter_sep_litmus(path_list)
        path_list, litmus_dict = self.filter_inst_type(path_list)
        # print(len(path_list))
        # for path in path_list:
        #     print(path)
        return path_list, litmus_dict

    def make_new_litmus_file(self, litmus_suite, litmus_dict):
        path = os.path.join(config.OUTPUT_DIR, 'mem_litmus/herd_complete_litmus.log')
        litmus_dir = os.path.join(config.OUTPUT_DIR, 'litmus_align_memsynth')
        init_litmus_dir = os.path.join(config.OUTPUT_DIR, 'litmus_align_memsynth_init')
        cat_path = os.path.join(config.CAT_DIR, 'riscv-complete.cat')
        herd_run_all_valid_litmus(cat_path, litmus_suite, path)
        state_dict = {r.name: r.states for r in parse_herd_log(path)}
        pos_list = []
        new_litmus_list = []
        for litmus_path in litmus_suite:
            litmus_name = litmus_path.split('/')[-1].split('.litmus')[0]
            states = state_dict[litmus_name]
            var_dict = defaultdict(list)
            new_var_list = []
            for state in states:
                pos_list.append(state.state)
                for var in state.state: #LitmusState.state
                    # print(var)
                    # print(state.state[var])
                    for var in state.state:
                        var_dict[var].append(state.state[var])
                for var in litmus_dict[litmus_path]:
                    if var not in var_dict.keys() and var not in new_var_list:
                        new_var_list.append(var)
        #
        #
        #     # 1. create new litmus test

            state_str = ''
            for var in var_dict:
                state_str += f'{var}={var_dict[var][0]}' + ' /\\ '
            state_str = state_str.strip('/\\ ')
            if len(new_var_list) > 0:
                state_str += ' /\\ '
                state_str += ' /\\ '.join(f'{reg}=1' for reg in new_var_list)
            state_str = state_str.strip().replace(']','').replace('[','')
            new_line = f'exists ( {state_str} )'
            print('new_line', new_line)
            with open(litmus_path, 'r') as f:
                lines = f.readlines()
            cut_index = next((i for i, line in enumerate(lines) if 'exists' in line), len(lines))
            new_lines = lines[:cut_index]
            new_lines.append(new_line)
            new_path = os.path.join(init_litmus_dir, f'{litmus_name}.litmus')
            with open(new_path, 'w') as f:
                f.writelines(new_lines)
            new_litmus_list.append(new_path)
        for litmus_path in new_litmus_list:
            print(litmus_path)
        herd_run_all_valid_litmus(cat_path, new_litmus_list, path)
        state_dict = {r.name: r.states for r in parse_herd_log(path)}
        for litmus_path in new_litmus_list:
            pos_list = []
            litmus_name = litmus_path.split('/')[-1].split('.litmus')[0]
            print(litmus_name)
            states = state_dict[litmus_name]
            print(states)
            var_dict = defaultdict(list)
            new_var_list = []
            for state in states:
                pos_list.append(state.state)
                for var in state.state:  # LitmusState.state
                    # print(var)
                    # print(state.state[var])
                    for var in state.state:
                        var_dict[var].append(state.state[var])


            # 2. Remove duplicates from each list while preserving the original order
            var_dict = {k: list(dict.fromkeys(v)) for k, v in var_dict.items()}

            # 3. product
            keys = list(var_dict.keys())
            value_lists = [var_dict[k] for k in keys]

            cartesian_product = list(product(*value_lists))

            # get dict
            result_dicts = [dict(zip(keys, values)) for values in cartesian_product]
            # print(result_dicts)
            print(litmus_name)
            print('pos dict')
            for item in pos_list:
                print(item)
            print('neg dict')
            neg_list = []
            for item in result_dicts:
                if item not in pos_list:
                    neg_list.append(item)
            # for item in neg_list:
            #     print(item)
            # to str dict
            formatted_results = [
                ' /\\ '.join(f'{k}={v}' for k, v in d.items())
                for d in result_dicts
            ]
            flag_dict = {}
            for i, item in enumerate(result_dicts):
                if item in pos_list:
                    flag = 1
                else:
                    flag = 0
                state_str = ' /\\ '.join(f'{k}={v}' for k, v in item.items())
                line = state_str.replace('[', '').replace(']','')
                new_line = f'exists ( {line} )'
                print(f'exists ( {line} )')
                with open(litmus_path, 'r') as f:
                    lines = f.readlines()
                cut_index = next((i for i, line in enumerate(lines) if 'exists' in line), len(lines))
                new_lines = lines[:cut_index]
                new_lines.append(new_line)
                result_line = f'Result '
                if flag == 1:
                    result_line = result_line + 'RVWMOherd \n'
                else:
                    result_line = result_line + 'RVWMONot \n'
                new_lines = [result_line]+new_lines
                with open(os.path.join(litmus_dir, f'{litmus_name}_{i}.litmus'), 'w') as f:
                    f.writelines(new_lines)
                flag_dict[f'{litmus_name}_{i}']=flag
            with open(os.path.join(init_litmus_dir, f'flag.txt'), 'a+') as f:
                for k, v in flag_dict.items():
                    f.write(f'{os.path.join(litmus_dir,k)}.litmus,{v}')
                    f.write('\n')



    def run(self):
        path_list, litmus_dict = self.filter_litmus_list()
        self.make_new_litmus_file(path_list, litmus_dict)


    # def run(self):
    #     path_list = self.filter_litmus_list()
    #
    #     print(len(path_list))
    #     prefix_to_remove = 'litmus/RISCV/'
    #     with open('RISCV-expected.csv' , 'r', newline='') as csvfile:
    #         reader = csv.reader(csvfile)
    #         processed_rows = {}
    #
    #         for row in reader:
    #             if row:
    #                 row[0] = row[0].removeprefix(prefix_to_remove)
    #                 processed_rows[row[0]]=row[1]
    #     # for row in processed_rows:
    #     #     print(row)
    #     results = {}
    #     for path in path_list:
    #         # print(path)
    #         litmus = path.split('/')[-1]
    #         if litmus in processed_rows:
    #             results[path] = processed_rows[litmus]
    #     print(len(results))
    #     # for id in results:
    #     #     print(id)
    #     #     print(results[id])
    #     with open('litmus.txt', 'w') as f:
    #         for path in results:
    #             f.write(path)
    #             f.write(',')
    #             f.write(results[path])
    #             f.write('\n')

if __name__ == '__main__':
    filter_litmus_list = [
        os.path.join(config.INPUT_DIR, 'chip_execution_logs/exceed_two_threads.txt'),
        os.path.join(config.INPUT_DIR, 'chip_execution_logs/exceed_4_access.txt'),
        os.path.join(config.INPUT_DIR, 'chip_execution_logs/exceptional.txt'),
        os.path.join(config.INPUT_DIR, 'chip_execution_logs/fence.i_ctrlfencei.txt'),
    ]
    litmus_files = get_litmus_by_policy(GetLitmusPolicy.FilterByFile, {
        'file_list': filter_litmus_list})
    test=TestMemSynthFormat(litmus_files)
    test.run()