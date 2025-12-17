import itertools
import os
import sys
import time

from src.slide import config
from src.slide.analysis import RVWMO
from src.slide.analysis.model.rvwmo_variant import RVWMO_variant
from src.slide.litmus.load_litmus import get_litmus_by_policy, GetLitmusPolicy, sort_litmus_by_weight


from src.slide.synth.model.ppo_pool_rvwmo import get_rvwmo_model
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
exp4_synth_litmus_dir = os.path.join(config.OUTPUT_DIR, 'complete_litmus_rvwmo')
exp5_used_litmus_dir = os.path.join(config.OUTPUT_DIR, 'complete_litmus_rvwmo_variant')

exp_5_output_log_dir = os.path.join(config.TEST_DIR,'results/exp5_result')
exp5_png_path = os.path.join(config.TEST_DIR, 'results/exp5_result.png')


def delete_history():
    print(f'deleting exp5 exp_5_output_log_dir : {exp_5_output_log_dir}')
    for filename in os.listdir(exp_5_output_log_dir):
        file_path = os.path.join(exp_5_output_log_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    if os.path.exists(exp5_png_path):
        os.remove(exp5_png_path)

def pre():
    assert len(os.listdir(exp4_synth_litmus_dir)) != 0, "must pass exp4 synth rvwmo post"

rvwmo_model = [
    '[M];po-loc;[W]',
    r'([R];po-loc-no-w;[R]) \ rsw',
    '[AMO|X];rfi;[R]',
    'fence',
    '[AQ];po;[M]',
    '[M];po;[RL]',
    '[RCsc];po;[RCsc]',
    'rmw',
    '[M];addr;[M]',
    '[M];data;[W]',
    '[M];ctrl;[W]',
    '[M];(addr|data);[W];rfi;[R]',
    '[M];addr;[M];po;[W]',
]

model = {
    'RW' : ['[R];po;[W]'],
    'AMO_X' : ['[AMO];po;[M]', '[M];po;[AMO]', '[X];po;[M]', '[M];po;[X]'],
    'fence' : ['[M];fencerel(Fence.r.r);[M]', '[M];fencerel(Fence.r.w);[M]', '[M];fencerel(Fence.r.rw);[M]',
               '[M];fencerel(Fence.w.r);[M]', '[M];fencerel(Fence.w.w);[M]', '[M];fencerel(Fence.w.rw);[M]',
               '[M];fencerel(Fence.rw.r);[M]', '[M];fencerel(Fence.rw.w);[M]', '[M];fencerel(Fence.rw.rw);[M]',
               '[M];fencerel(Fence.tso);[M]'],
    'SC' : ['[M];po;[M]'],
    'TSO' : ['[R];po;[M]', '[W];po;[W]'],
    'strong_ppo12' : ['[M];(addr|data);[W];po;[R]'],
    'strong_ppo13' : ['[M];addr;[M];po;[R]']
}

cat_dir = os.path.join(config.CAT_DIR,'memory_model_variant/variant')
log_dir = os.path.join(config.CAT_DIR,'memory_model_variant/variant/log')


def create_cat_file(ppo_list, cat_file):
    input_cat_def_file = f'{config.INPUT_DIR}/CAT/change/riscv-defs.cat'
    input_cat_file = f'{config.INPUT_DIR}/CAT/change/riscv.cat'
    ppo_list = rvwmo_model + ppo_list
    with open(input_cat_def_file, 'r') as f:
        lines = f.readlines()

    with open(input_cat_file, 'r') as f:
        cat_lines = f.readlines()

    with open(cat_file, 'w') as f:
        f.writelines(lines)
        var_str = '\n'
        for i, ppo in enumerate(ppo_list):
            var_str += 'let ' if i == 0 else 'and '
            var_str += f'r{i + 1} = {ppo} \n'

        ppo_str = '\nlet ppo = \n'
        for i, ppo in enumerate(ppo_list):
            ppo_str += f' ' if i == 0 else '| '
            ppo_str += f'r{i + 1}\n'

        f.write(var_str)
        f.write(ppo_str)
        f.writelines(cat_lines)

    return cat_file

class TestSynth:

    def get_model(self):
        new_model = {}
        keys = list(model.keys())
        for r in range(1, len(keys) + 1):
            for key_combo in itertools.combinations(keys, r):
                if 'TSO' in key_combo and len(key_combo) > 1:
                    continue
                if 'SC' in key_combo and len(key_combo) > 1:
                    continue
                new_key = '_'.join(key_combo)
                new_patterns = []
                for k in key_combo:
                    new_patterns.extend(model[k])
                new_model[new_key] = new_patterns
        for k in new_model:
            print(k)
            print(new_model[k])
        for k in new_model:
            cat_path = os.path.join(cat_dir, f'{k}.cat')
            create_cat_file(new_model[k], cat_path)


    def get_dif_litmus_datasets(self, src_cat, tgt_cat, src_log, tgt_log, dif_log):
        litmus_files = get_litmus_by_policy(GetLitmusPolicy.FilterByFile, {
            'file_list': filter_litmus_list})
        for root, dirs, files in os.walk(exp4_synth_litmus_dir):
            for file in files:
                file_path = os.path.join(root, file)
                litmus_files.append(file_path)
        for root, dirs, files in os.walk(exp5_used_litmus_dir):
            for file in files:
                file_path = os.path.join(root, file)
                litmus_files.append(file_path)
        if not os.path.exists(src_log):
            herd_run_all_valid_litmus(src_cat, litmus_files, src_log)
        if not os.path.exists(tgt_log):
            herd_run_all_valid_litmus(tgt_cat, litmus_files, tgt_log)
        litmus_suite = []
        if not os.path.exists(dif_log):
            dif_litmus_suite = get_model_diff(src_log, tgt_log)
            print(len(dif_litmus_suite))
            litmus_suite = sort_litmus_by_weight(dif_litmus_suite,litmus_dir = [f'{config.INPUT_DIR}/litmus',exp4_synth_litmus_dir,exp5_used_litmus_dir])
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
        for root, dirs, files in os.walk(cat_dir):
            for file in files:
                start = time.time()
                file_path = os.path.join(root, file)
                src_cat = os.path.join(config.CAT_DIR,'riscv-complete.cat')
                tgt_cat = file_path
                model_name =  os.path.splitext(os.path.basename(file_path))[0]
                src_log = os.path.join(config.HERD_LOG_DIR_PATH,'variant/log/riscv-complete.log')
                tgt_log = os.path.join(config.HERD_LOG_DIR_PATH,f'variant/log/{model_name}.log')
                dif_log = os.path.join(config.HERD_LOG_DIR_PATH,f'variant/dif/dif_{model_name}_rvwmo.log')
                litmus_test_suite = self.get_dif_litmus_datasets(src_cat, tgt_cat, src_log, tgt_log, dif_log)
                # # print(len(litmus_test_suite))
                test_suite = []
                for litmus in litmus_test_suite:
                    file_path = search_file(litmus, f'{config.INPUT_DIR}/litmus', '.litmus')
                    if file_path is not None:
                        test_suite.append(file_path)
                        continue
                    file_path = search_file(litmus, exp4_synth_litmus_dir, '.litmus')
                    if file_path is not None:
                        test_suite.append(file_path)
                        continue
                    file_path = search_file(litmus, exp5_used_litmus_dir, '.litmus')
                    if file_path is not None:
                        test_suite.append(file_path)
                        continue
                litmus_test_suite = test_suite
                exp_5_output_log_path = os.path.join(exp_5_output_log_dir,f'log_{model_name}.log')
                exp_5_output_result_file_path = os.path.join(exp_5_output_log_dir,f'result_{model_name}.txt')
                exp_5_test_ppo_path = os.path.join(exp_5_output_log_dir,f'tested_ppo_{model_name}.txt')
                header = f"Experiment 5: synth {model_name}"
                centered_header = header.center(60)
                print(f"\n{bar}\n{centered_header}\n{bar}\n")
                model_list = []
                for key in model.keys():
                    if key in model_name:
                        model_list.append(key)
                # synth
                with open(exp_5_output_log_path, 'w') as f:
                    ori = sys.stdout
                    sys.stdout = f
                    config.init()
                    config.set_var('reg_size',64)

                    tgt_mm = RVWMO_variant(plot_enabled=False, variant_array = model_list)

                    cur_mm = get_rvwmo_model()
                    patch, ppo_list = synth_by_test_pattern_online(tgt_mm, cur_mm, litmus_test_suite, tgt_cat)
                    time_cost = time.time() - start
                    sys.stdout = ori
                    f.close()

                with open(exp_5_output_result_file_path, 'w') as f:
                    ori = sys.stdout
                    sys.stdout = f
                    if patch is not None:
                        for _,ppo,_,_ in patch:
                            print(ppo)
                    print('time: ',time_cost)
                    sys.stdout = ori
                    f.close()

                with open(exp_5_test_ppo_path, 'w') as f:
                    ori = sys.stdout
                    sys.stdout = f
                    for ppo in ppo_list:
                        print(ppo)
                    sys.stdout = ori
                    f.close()



if __name__ == '__main__':
    bar = "=" * 60
    header = f"Experiment 5: synth RVWMO variant"
    centered_header = header.center(60)
    delete_history()
    print(f"\n{bar}\n{centered_header}\n{bar}\n")
    test = TestSynth()
    # test.get_model()
    test.synth()
