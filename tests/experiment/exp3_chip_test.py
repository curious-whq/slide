import os
import sys
import time

from src.slide import config
from src.slide.analysis import RVWMO
from src.slide.comp.parse_result import parse_herd_log, parse_chip_log
from src.slide.litmus.load_litmus import get_litmus_by_policy, GetLitmusPolicy, sort_litmus_by_weight
from src.slide.synth.model.ppo_pool_rvwmo import get_rvwmo_model
from src.slide.synth.model.ppo_pool_rvwmo_init import get_rvwmo_init_model
from src.slide.synth.pattern_synth import synth_by_test_pattern_online
from src.slide.utils.file_util import search_file, read_file
from src.slide.utils.herd_util import herd_run_all_valid_litmus, get_model_diff
from src.slide.utils.litmus7_util import litmus_run_until_match, integrate_chip_log

filter_litmus_list = [
    os.path.join(config.INPUT_DIR,'chip_execution_logs/exceed_two_threads.txt'),
    os.path.join(config.INPUT_DIR,'chip_execution_logs/exceed_4_access.txt'),
    os.path.join(config.INPUT_DIR,'chip_execution_logs/exceptional.txt'),
    os.path.join(config.INPUT_DIR,'chip_execution_logs/fence.i_ctrlfencei.txt'),
    os.path.join(config.INPUT_DIR,'chip_execution_logs/X_exceed_four.txt'),
    os.path.join(config.INPUT_DIR, 'chip_execution_logs/exceed_two_threads_has_X.txt'),
    os.path.join(config.INPUT_DIR,'chip_execution_logs/C910_except_litmus.txt'),
]
exp_3_output_log_path = os.path.join(config.TEST_DIR,'results/exp3_synth_C910.log')
exp_3_output_result_file_path = os.path.join(config.TEST_DIR,'results/exp3_synth_C910_result.txt')
exp_3_test_ppo_path = os.path.join(config.TEST_DIR,'results/exp3_synth_C910_tested_ppo.txt')

def delete_history():
    if os.path.exists(exp_3_output_log_path):
        print(f'deleting exp_3_output_log_path: {exp_3_output_log_path}')
        os.remove(exp_3_output_log_path)
    if os.path.exists(exp_3_output_result_file_path):
        print(f'deleting exp_3_output_result_file_path: {exp_3_output_result_file_path}')
        os.remove(exp_3_output_result_file_path)
    if os.path.exists(exp_3_test_ppo_path):
        print(f'deleting exp_3_test_ppo_path: {exp_3_test_ppo_path}')
        os.remove(exp_3_test_ppo_path)


class TestChip:

    def get_dif_litmus_datasets_for_chips(self, src_cat, src_log, tgt_log, dif_log):

        filter_litmus_suite = []
        for filter_litmus_path in filter_litmus_list:
            with open(filter_litmus_path, 'r') as f:
                filter_litmus_suite.extend(f.readlines())
        filter_litmus_suite = [item.strip() for item in filter_litmus_suite]
        print(len(filter_litmus_suite))
        if not os.path.exists(dif_log):
            dif_litmus_suite = get_model_diff(src_log, tgt_log, mode1= 'herd', mode2= 'chip')
            dif_litmus_suite = [item for item in dif_litmus_suite if item not in filter_litmus_suite]
            litmus_suite = sort_litmus_by_weight(dif_litmus_suite)
            model_1_dict = {r.name: r.states for r in parse_herd_log(src_log)}
            model_2_dict = {r.name: r.states for r in parse_chip_log(tgt_log)}
            with open (dif_log, 'w') as f:
                for litmus in litmus_suite:
                    f.write(litmus)
                    f.write('\n')
            with open (dif_log.replace('.log', '.txt'), 'w') as f:
                for litmus in litmus_suite:
                    f.write(litmus)
                    f.write('\n')
                    litmus_path = search_file(litmus, f'{config.INPUT_DIR}/litmus', '.litmus')
                    litmus_content = read_file(litmus_path)
                    if litmus in model_1_dict:
                        f.write(str(model_1_dict[litmus]))
                        f.write('\n')
                    if litmus in model_2_dict:
                        f.write(str(model_2_dict[litmus]))
                        f.write('\n')
                    f.write(litmus_content)
                    f.write('\n')


        else:
            with open(dif_log, 'r') as f:
                litmus_suite = f.readlines()
                litmus_suite = [litmus.strip ()for litmus in litmus_suite]
        return litmus_suite




    def synth(self):
        start = time.time()
        src_cat = os.path.join(config.CAT_DIR,'riscv-complete.cat')
        src_log = os.path.join(config.INPUT_DIR,'herd/herd_results_rvwmo.log')
        tgt_log = os.path.join(config.INPUT_DIR,'chip_execution_logs/C910/chip_log.txt')
        dif_log = os.path.join(config.HERD_LOG_DIR_PATH,'dif_litmus_C910_RVWMO1.log')
        litmus_test_suite = self.get_dif_litmus_datasets_for_chips(src_cat, src_log, tgt_log, dif_log)
        # print(len(litmus_test_suite))
        litmus_test_suite = [search_file(litmus, f'{config.INPUT_DIR}/litmus', '.litmus') for litmus in litmus_test_suite]

        # synth
        with open(exp_3_output_log_path, 'w') as f:
            ori = sys.stdout
            sys.stdout = f
            config.init()
            config.set_var('reg_size',64)

            cur_mm = get_rvwmo_model()
            patch, ppo_list = synth_by_test_pattern_online(None, cur_mm, litmus_test_suite, None, mode='chip', chip_log_path = tgt_log)
            time_cost = time.time() - start
            sys.stdout = ori
            f.close()

        with open(exp_3_output_result_file_path, 'w') as f:
            ori = sys.stdout
            sys.stdout = f
            if patch is not None:
                for _,ppo,_,_ in patch:
                    print(ppo)
            print('time:', time_cost)
            sys.stdout = ori
            f.close()

        with open(exp_3_test_ppo_path, 'w') as f:
            ori = sys.stdout
            sys.stdout = f
            for ppo in ppo_list:
                print(ppo)
            sys.stdout = ori
            f.close()


if __name__ == '__main__':
    bar = "=" * 60
    header = f"Experiment 3: synth C910"
    centered_header = header.center(60)
    print(f"\n{bar}\n{centered_header}\n{bar}\n")
    delete_history()
    test = TestChip()
    test.synth()
