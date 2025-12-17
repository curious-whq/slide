import os
import sys
import time

from src.slide import config
from src.slide.analysis import RVWMO
from src.slide.litmus.load_litmus import get_litmus_by_policy, GetLitmusPolicy, sort_litmus_by_weight
from src.slide.synth.model.ppo_pool_rvwmo_init import get_rvwmo_init_model
from src.slide.synth.model.ppo_pool_rvwmo_sc_per_location import get_rvwmo_sc_per_location_model
from src.slide.synth.pattern_synth import synth_by_test_pattern_online
from src.slide.utils.file_util import search_file
from src.slide.utils.herd_util import herd_run_all_valid_litmus, get_model_diff

filter_litmus_list = [
    os.path.join(config.INPUT_DIR,'chip_execution_logs/exceed_two_threads.txt'),
    os.path.join(config.INPUT_DIR,'chip_execution_logs/exceed_4_access.txt'),
    os.path.join(config.INPUT_DIR,'chip_execution_logs/exceptional.txt'),
    os.path.join(config.INPUT_DIR,'chip_execution_logs/fence.i_ctrlfencei.txt'),
]

exp_2_output_log_path = os.path.join(config.TEST_DIR,'results/exp2_synth_rvwmo.log')
exp_2_output_result_file_path = os.path.join(config.TEST_DIR,'results/exp2_synth_rvwmo_result.txt')
exp_2_test_ppo_path = os.path.join(config.TEST_DIR,'results/exp2_synth_rvwmo_tested_ppo.txt')

def delete_history():
    if os.path.exists(exp_2_output_log_path):
        print(f'deleting exp_2_output_log_path: {exp_2_output_log_path}')
        os.remove(exp_2_output_log_path)
    if os.path.exists(exp_2_output_result_file_path):
        print(f'deleting exp_2_output_result_file_path: {exp_2_output_result_file_path}')
        os.remove(exp_2_output_result_file_path)
    if os.path.exists(exp_2_test_ppo_path):
        print(f'deleting exp_2_test_ppo_path: {exp_2_test_ppo_path}')
        os.remove(exp_2_test_ppo_path)


class TestSynth:

    def get_dif_litmus_datasets(self, src_cat, tgt_cat, src_log, tgt_log, dif_log):
        litmus_files = get_litmus_by_policy(GetLitmusPolicy.FilterByFile, {
            'file_list': filter_litmus_list})
        if not os.path.exists(src_log):
            herd_run_all_valid_litmus(src_cat, litmus_files, src_log)
        if not os.path.exists(tgt_log):
            herd_run_all_valid_litmus(tgt_cat, litmus_files, tgt_log)
        litmus_suite = []
        if not os.path.exists(dif_log):
            dif_litmus_suite = get_model_diff(src_log, tgt_log)
            litmus_suite = sort_litmus_by_weight(dif_litmus_suite)
            with open (dif_log, 'w') as f:
                for litmus in litmus_suite:
                    f.write(litmus)
                    f.write('\n')
        else:
            with open(dif_log, 'r') as f:
                litmus_suite = f.readlines()
                litmus_suite = [litmus.strip ()for litmus in litmus_suite]

        return litmus_suite


    def synth(self):
        start = time.time()
        src_cat = os.path.join(config.CAT_DIR,'riscv-init.cat')
        tgt_cat = os.path.join(config.CAT_DIR,'riscv-complete.cat')
        src_log = os.path.join(config.HERD_LOG_DIR_PATH,'riscv-init.log')
        tgt_log = os.path.join(config.HERD_LOG_DIR_PATH,'riscv-complete.log')
        dif_log = os.path.join(config.HERD_LOG_DIR_PATH,'dif_litmus_init1.log')
        litmus_test_suite = self.get_dif_litmus_datasets(src_cat, tgt_cat, src_log, tgt_log, dif_log)
        # print(len(litmus_test_suite))
        litmus_test_suite = [search_file(litmus, f'{config.INPUT_DIR}/litmus', '.litmus') for litmus in litmus_test_suite]

        # synth
        with open(exp_2_output_log_path, 'w') as f:
            ori = sys.stdout
            sys.stdout = f
            config.init()
            config.set_var('reg_size',64)

            rvwmo = RVWMO()
            cur_mm = get_rvwmo_init_model()
            rvwmo_cat = os.path.join(config.CAT_DIR, 'riscv-complete.cat')
            patch, ppo_list = synth_by_test_pattern_online(rvwmo, cur_mm, litmus_test_suite, rvwmo_cat)
            time_cost = time.time() - start
            sys.stdout = ori
            f.close()

        with open(exp_2_output_result_file_path, 'w') as f:
            ori = sys.stdout
            sys.stdout = f
            if patch is not None:
                for _,ppo,_,_ in patch:
                    print(ppo)
            print('time: ',time_cost)
            sys.stdout = ori
            f.close()

        with open(exp_2_test_ppo_path, 'w') as f:
            ori = sys.stdout
            sys.stdout = f
            for ppo in ppo_list:
                print(ppo)
            sys.stdout = ori
            f.close()


if __name__ == '__main__':
    bar = "=" * 60
    header = f"Experiment 2: synth RVWMO"
    centered_header = header.center(60)
    print(f"\n{bar}\n{centered_header}\n{bar}\n")
    delete_history()
    test = TestSynth()
    test.synth()