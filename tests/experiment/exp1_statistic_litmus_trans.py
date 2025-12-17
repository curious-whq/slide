import os
import sys
import time

import numpy as np
from matplotlib import pyplot as plt

from src.slide import config
from src.slide.analysis import RVWMO
from src.slide.comp.parse_result import parse_herd_log, parse_chip_log, parse_chip_log_not_trans
from src.slide.litmus.load_litmus import get_litmus_by_policy, GetLitmusPolicy, sort_litmus_by_weight
from src.slide.synth.model.ppo_pool_rvwmo import get_rvwmo_model
from src.slide.synth.model.ppo_pool_rvwmo_init import get_rvwmo_init_model
from src.slide.synth.pattern_synth import synth_by_test_pattern_online
from src.slide.utils.file_util import search_file, read_file, get_file_name_without_suffix_by_path
from src.slide.utils.herd_util import herd_run_all_valid_litmus, get_model_diff
from src.slide.utils.litmus7_util import litmus_run_until_match, integrate_chip_log


exp1_litmus_trans_results_path = os.path.join(config.TEST_DIR,'results/exp1_litmus_trans_results.png')

filter_litmus_list = [
    os.path.join(config.INPUT_DIR,'chip_execution_logs/exceed_two_threads.txt'),
    os.path.join(config.INPUT_DIR,'chip_execution_logs/exceed_4_access.txt'),
    os.path.join(config.INPUT_DIR,'chip_execution_logs/exceptional.txt'),
    os.path.join(config.INPUT_DIR,'chip_execution_logs/fence.i_ctrlfencei.txt'),
    os.path.join(config.INPUT_DIR, 'chip_execution_logs/exceed_two_threads_has_X.txt'),
    os.path.join(config.INPUT_DIR, 'chip_execution_logs/X_exceed_four.txt')

]

def delete_history():
    if os.path.exists(exp1_litmus_trans_results_path):
        print(f'deleting exp1_litmus_trans_results_path: {exp1_litmus_trans_results_path}')
        os.remove(exp1_litmus_trans_results_path)

class TestStatistic:

    def get_litmus_trans_up(self, tgt_log):
        litmus_state_add_num_dict = {}
        filter_litmus_suite = []
        for filter_litmus_path in filter_litmus_list:
            with open(filter_litmus_path, 'r') as f:
                filter_litmus_suite.extend(f.readlines())
        filter_litmus_suite = [item.strip() for item in filter_litmus_suite]
        print(len(filter_litmus_suite))
        chip_dict = {r.name: r.states for r in parse_chip_log(tgt_log)}
        chip_no_trans_dict =  {r.name: r.states for r in parse_chip_log_not_trans(tgt_log)}
        litmus_files = get_litmus_by_policy(GetLitmusPolicy.FilterByFile, {
            'file_list': filter_litmus_list})
        print(len(litmus_files))
        size = 0
        for litmus in litmus_files:
            litmus = get_file_name_without_suffix_by_path(litmus)
            if litmus not in chip_dict:
                # print(litmus)
                continue
            if litmus not in chip_no_trans_dict:
                continue
            if len(chip_dict[litmus])> len(chip_no_trans_dict[litmus]):
                size += 1
                print(litmus)
                print(f'base:{len(chip_no_trans_dict[litmus])}, {set(chip_no_trans_dict[litmus])}')
                print(f'trans:{len(chip_dict[litmus])}, {set(chip_dict[litmus])}')
                dif_num = len(chip_dict[litmus]) - len(chip_no_trans_dict[litmus])
                litmus_state_add_num_dict[dif_num] = litmus_state_add_num_dict.get(dif_num, 0) + 1

        print(size)
        for size in litmus_state_add_num_dict:
            print(f'{size}:{litmus_state_add_num_dict[size]}')
        return litmus_state_add_num_dict

    def draw(self, state_dict):
        x_labels = state_dict.keys()
        x = np.arange(len(x_labels))

        np.random.seed(42)
        up_state_num_list = state_dict.values()

        plt.figure(figsize=(10, 6))

        bar_width = 0.5

        plt.bar(x , up_state_num_list, bar_width, color="#A5D6A7")

        plt.xticks(x, x_labels, ha="right", fontsize=14)
        plt.ylabel("Number of Litmus Tests ", fontsize=16)
        plt.xlabel("Number of Newly Triggered States", fontsize=16)
        # plt.ylim(0, )
        # plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.subplots_adjust(bottom=0.2)

        plt.savefig(exp1_litmus_trans_results_path, bbox_inches='tight', pad_inches=0.05)


if __name__ == '__main__':
    bar = "=" * 60
    header = f"Experiment 1: statistic_litmus_trans"
    centered_header = header.center(60)
    print(f"\n{bar}\n{centered_header}\n{bar}\n")
    delete_history()
    test = TestStatistic()
    tgt_log = os.path.join(config.INPUT_DIR, 'chip_execution_logs/C910/chip_log.txt')

    state_dict = test.get_litmus_trans_up(tgt_log)
    test.draw(state_dict)