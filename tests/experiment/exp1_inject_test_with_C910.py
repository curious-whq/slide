import argparse
import json
import os

import numpy as np
from matplotlib import pyplot as plt

from src.slide import config
from src.slide.comp.parse_result import parse_chip_log
from src.slide.litmus.litmus_transformer import get_transform_litmus_list, TransMode
from src.slide.litmus.load_litmus import get_litmus_by_policy, GetLitmusPolicy
from src.slide.utils.file_util import search_file, write_list_to_file, write_line_to_file
from src.slide.utils.litmus7_util import litmus_run_get_log

litmus_suite = [
    '2+2W',
    'LB',
    'SB',
    'MP',
    'R',
    'S',
    'ISA2',
    'WRC',
    'PPO12-2',
    'PPOCA'
]



# class TestInject:

run_time = 10000
exp1_result_dir = os.path.join(config.TEST_DIR,'results/exp1_log_with_C910')
exp1_log_path = os.path.join(config.TEST_DIR, 'results/exp1_inject_test_with_C910.txt')
exp1_png_path = os.path.join(config.TEST_DIR, 'results/exp1_inject_with_C910.png')

def delete_history():
    print(f'deleting exp1_result_dir: {exp1_result_dir}')
    for filename in os.listdir(exp1_result_dir):
        file_path = os.path.join(exp1_result_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    if os.path.exists(exp1_log_path):
        print(f'deleting exp1_log: {exp1_log_path}')
        os.remove(exp1_log_path)
    if os.path.exists(exp1_png_path):
        print(f'deleting exp1_png_path: {exp1_png_path}')
        os.remove(exp1_png_path)

def inject():
    results = []

    litmus_files = [os.path.join(config.TEST_DIR, f'experiment/exp1_litmus/{item}.litmus') for item in litmus_suite]
    has_run_litmus_files = []
    # run_log_path = os.path.join(config.TEST_DIR, 'experiment/inject_run.txt')
    for litmus_file_path in litmus_files:
        # 1. get result
        litmus_name = os.path.splitext(os.path.basename(litmus_file_path))[0]
        base_log_path = litmus_run_get_log(litmus_name, litmus_file_path, run_time=run_time)
        if base_log_path is None:
            #show the chip don't support
            continue
        with open(os.path.join(exp1_result_dir,f'{os.path.splitext(os.path.basename(litmus_file_path))[0]}_base.log'), 'w') as wf:
            with open(base_log_path, 'r') as f:
                lines = f.readlines()
                wf.writelines(lines)
        base_result = {r.name: r for r in parse_chip_log(base_log_path)}[litmus_name]
        base_states = base_result.get_state_list_by_num()
        # 2. get state which want to transform
        total_num = sum(state.num for state in base_states)
        # threshold
        # transform_state = None
        # for state in base_states:
        #     if state.num * 2000 > total_num:
        #         transform_state = state
        #         break
        transform_state = base_states[0]
        print(transform_state)
        trans_litmus_suite = get_transform_litmus_list(litmus_name, litmus_file_path, [transform_state], TransMode.DIRECT)
        print(trans_litmus_suite)
        # 3. get new result
        trans_result = None
        for i, trans_litmus_path in enumerate(trans_litmus_suite):
            trans_log_path = litmus_run_get_log(litmus_name, trans_litmus_path, run_time=run_time)
            if trans_log_path is None:
                continue
            with open(os.path.join(exp1_result_dir,
                                   f'{os.path.splitext(os.path.basename(litmus_file_path))[0]}_trans{i}.log'), 'w') as wf:
                with open(trans_log_path, 'r') as f:
                    lines = f.readlines()
                    wf.writelines(lines)
            if trans_result is None:
                trans_result = {r.name: r for r in parse_chip_log(trans_log_path)}[litmus_name]
            else:
                trans_result.union({r.name: r for r in parse_chip_log(trans_log_path)}[litmus_name])
        if trans_result is None:
            results.append({
                'litmus_name': litmus_name,
                'state' : transform_state,
                'result' : 'no transformation found'
            })
            continue
        trans_states = trans_result.states
        for trans_state in trans_states:
            print(trans_state.get_num_str())

        result = {
            'litmus_name' : litmus_name,
            'state' : str(transform_state),
            'before_num' : transform_state.num,
            'before_sum' : total_num,
            'after_num' : 0 if transform_state not in trans_states else trans_states[trans_states.index(transform_state)].num,
            'after_sum' : sum(state.num for state in trans_states),
        }
        result['ratio'] = (result['after_num'] / result['before_num'])/(result['after_sum']/result['before_sum'])
        results.append(result)
        with open(exp1_log_path, 'a+') as f:
            f.write(json.dumps(result))
            f.write('\n')


    for result in results:
        print(result)


def static():
    all_files = []
    for root, dirs, files in os.walk(exp1_result_dir):
        for file in files:
            all_files.append(os.path.join(root, file))
    base_log = {}
    trans_log = {}
    for file in all_files:
        file_name = os.path.basename(file).split('.')[0]
        litmus_name = file_name.split('_')[0]
        if('trans' in file_name):
            trans_log.setdefault(litmus_name, []).append(file)
        else:
            base_log.setdefault(litmus_name, []).append(file)

    result = {}
    for litmus_name in litmus_suite:
        base_list = base_log.setdefault(litmus_name, [])
        trans_list = trans_log.setdefault(litmus_name, [])
        base_result = None
        trans_result = None
        for base_file in base_list:
            if base_result is None:
                base_result = {r.name: r for r in parse_chip_log(base_file)}[litmus_name]
            else:
                base_result.union({r.name: r for r in parse_chip_log(base_file)}[litmus_name])
        for trans_file in trans_list:
            if trans_result is None:
                trans_result = {r.name: r for r in parse_chip_log(trans_file)}[litmus_name]
            else:
                trans_result.union({r.name: r for r in parse_chip_log(trans_file)}[litmus_name])
        base_states = base_result.get_state_list_by_num()
        min_state = base_states[0]

        base_num = sum(state.num for state in base_states)
        before_num = min_state.num

        trans_num= sum(state.num for state in trans_result.states)
        after_num = 0 if min_state not in trans_result.states else trans_result.states[trans_result.states.index(min_state)].num
        result[litmus_name] = (after_num / before_num)/(trans_num /base_num)


    x = np.arange(len(litmus_suite))

    np.random.seed(42)
    base_litmus_list = [1] * len(litmus_suite)
    up_litmus_list = [result[litmus_name] for litmus_name in litmus_suite]

    plt.figure(figsize=(10, 6))

    bar_width = 0.2

    plt.bar(x - 0.5 * bar_width, base_litmus_list, bar_width, label="Base", color="#FFF59D")
    plt.bar(x + 0.5 * bar_width, up_litmus_list, bar_width, label="LitmusTrans", color="#A5D6A7")

    plt.xticks(x, litmus_suite, rotation=45, ha="right", fontsize=14)
    plt.ylabel("Outcome Occurrences", fontsize=16)
    plt.legend(fontsize=14)
    plt.subplots_adjust(bottom=0.2)

    plt.savefig(exp1_png_path, bbox_inches='tight', pad_inches=0.05)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Experiment 1 with ssh parameters.")
    parser.add_argument('--hostname', type=str, required=True, help='hostname')
    parser.add_argument('--port', type=int, required=True, help='port')
    parser.add_argument('--username', type=str, required=True, help='username')
    parser.add_argument('--password', type=str, required=True, help='password')
    parser.add_argument('--hostpath', type=str, required=True, help='hostpath')
    args = parser.parse_args()
    config.HOSTPATH = args.hostpath
    config.USERNAME = args.username
    config.PASSWORD = args.password
    config.HOSTNAME = args.hostname
    config.PORT = args.port
    bar = "=" * 60
    header = f"Experiment 1: inject_test with C910"
    centered_header = header.center(60)
    print(f"\n{bar}\n{centered_header}\n{bar}\n")
    delete_history()
    inject()
    static()










