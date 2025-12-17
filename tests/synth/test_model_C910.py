import os
import sys

from src.slide import config
from src.slide.analysis import RVWMO
from src.slide.analysis.model.C910 import C910
from src.slide.analysis.model.rvwmo import ppo_r11
from src.slide.comp.parse_result import parse_herd_log
from src.slide.litmus.load_litmus import get_litmus_by_policy, GetLitmusPolicy
from src.slide.synth.model.ppo_pool_C910 import get_C910_model
from src.slide.synth.pattern_synth import synth_by_test_pattern_online
# from src.slide.synth.pattern_synth import synth_by_test_pattern_online_for_one_ppo
from src.slide.utils.file_util import read_file, search_file
from src.slide.utils.herd_util import herd_run_all_valid_litmus

C910_dif_litmus_dir = os.path.join(config.INPUT_DIR,'dif/C910')


mm_suite={
    'C910':{
        'cat': os.path.join(config.CAT_DIR,'riscv-C910.cat'),
        'log': os.path.join(config.INPUT_DIR, 'herd/herd_results_C910.log')
    },
    'C910_without_AMO':{
        'cat': os.path.join(config.CAT_DIR,'riscv-C910-without-AMOAMO.cat'),
        'log': os.path.join(config.INPUT_DIR, 'herd/herd_results_C910_without_AMOAMO.log')
    },
    'C910_without_AMOW':{
        'cat': os.path.join(config.CAT_DIR,'riscv-C910-withoutAMOW.cat'),
        'log': os.path.join(config.INPUT_DIR, 'herd/herd_results_C910_without_AMOW.log')
    },
    'C910_without_PX':{
        'cat': os.path.join(config.CAT_DIR,'riscv-C910-without-PX.cat'),
        'log': os.path.join(config.INPUT_DIR, 'herd/herd_results_C910_without_PX.log')
    },
    'C910_without_RAMO':{
        'cat': os.path.join(config.CAT_DIR,'riscv-C910-without-RAMO.cat'),
        'log': os.path.join(config.INPUT_DIR, 'herd/herd_results_C910_without_RAMO.log')
    },
    'C910_without_WAMO':{
        'cat': os.path.join(config.CAT_DIR,'riscv-C910-without-WAMO.cat'),
        'log': os.path.join(config.INPUT_DIR, 'herd/herd_results_C910_without_WAMO.log')
    },
    'C910_without_XP':{
        'cat': os.path.join(config.CAT_DIR,'riscv-C910-without-XP.cat'),
        'log': os.path.join(config.INPUT_DIR, 'herd/herd_results_C910_without_XP.log')
    },
    'C910_without_XX': {
        'cat': os.path.join(config.CAT_DIR, 'riscv-C910-without-XX.cat'),
        'log': os.path.join(config.INPUT_DIR, 'herd/herd_results_C910_without_XX.log')
    },
    'RVWMO':{
        'cat': os.path.join(config.CAT_DIR,'riscv-complete.cat'),
        'log': os.path.join(config.INPUT_DIR, 'herd/herd_results_rvwmo.log')
    },
}

filter_litmus_list = [
    os.path.join(config.INPUT_DIR,'chip_execution_logs/exceed_two_threads.txt'),
    os.path.join(config.INPUT_DIR,'chip_execution_logs/exceed_4_access.txt'),
    os.path.join(config.INPUT_DIR,'chip_execution_logs/exceptional.txt')
]
class TestC910Model:

    def test_get_C910_model(self):
        C910_cat = read_file(mm_suite['C910']['cat'])
        print(C910_cat)



    def test_get_C910_model_by_herd(self):
        with open('output.txt', 'w') as f:
            sys.stdout = f
            litmus_files = get_litmus_by_policy(GetLitmusPolicy.FilterByFile,{
                'file_list': filter_litmus_list})
            print(len(litmus_files))

            herd_run_all_valid_litmus(mm_suite['C910']['cat'], litmus_files, mm_suite['C910']['log'])

    def test_get_C910_without_AMOAMO_by_herd(self):
        litmus_files = get_litmus_by_policy(GetLitmusPolicy.FilterByFile, {
            'file_list': filter_litmus_list})
        print(len(litmus_files))

        print(litmus_files)
        herd_run_all_valid_litmus(mm_suite['C910_without_AMO']['cat'], litmus_files, mm_suite['C910_without_AMO']['log'])

    def test_get_C910_without_AMOW_by_herd(self):
        litmus_files = get_litmus_by_policy(GetLitmusPolicy.FilterByFile, {
            'file_list': filter_litmus_list})
        print(len(litmus_files))

        print(litmus_files)
        herd_run_all_valid_litmus(mm_suite['C910_without_AMOW']['cat'], litmus_files, mm_suite['C910_without_AMOW']['log'])

    def test_get_C910_without_PX_by_herd(self):
        litmus_files = get_litmus_by_policy(GetLitmusPolicy.FilterByFile, {
            'file_list': filter_litmus_list})
        print(len(litmus_files))

        print(litmus_files)
        herd_run_all_valid_litmus(mm_suite['C910_without_PX']['cat'], litmus_files, mm_suite['C910_without_PX']['log'])

    def test_get_C910_without_RAMO_by_herd(self):
        litmus_files = get_litmus_by_policy(GetLitmusPolicy.FilterByFile, {
            'file_list': filter_litmus_list})
        print(len(litmus_files))

        print(litmus_files)
        herd_run_all_valid_litmus(mm_suite['C910_without_RAMO']['cat'], litmus_files, mm_suite['C910_without_RAMO']['log'])

    def test_get_C910_without_WAMO_by_herd(self):
        litmus_files = get_litmus_by_policy(GetLitmusPolicy.FilterByFile, {
            'file_list': filter_litmus_list})
        print(len(litmus_files))

        print(litmus_files)
        herd_run_all_valid_litmus(mm_suite['C910_without_WAMO']['cat'], litmus_files, mm_suite['C910_without_WAMO']['log'])

    def test_get_C910_without_XP_by_herd(self):
        litmus_files = get_litmus_by_policy(GetLitmusPolicy.FilterByFile, {
            'file_list': filter_litmus_list})
        print(len(litmus_files))

        print(litmus_files)
        herd_run_all_valid_litmus(mm_suite['C910_without_XP']['cat'], litmus_files, mm_suite['C910_without_XP']['log'])

    def test_get_C910_without_XX_by_herd(self):
        litmus_files = get_litmus_by_policy(GetLitmusPolicy.FilterByFile, {
            'file_list': filter_litmus_list})
        print(len(litmus_files))

        print(litmus_files)
        herd_run_all_valid_litmus(mm_suite['C910_without_XX']['cat'], litmus_files, mm_suite['C910_without_XX']['log'])

    def test_all_herd(self):
        self.test_get_C910_without_AMOW_by_herd()
        self.test_get_C910_without_AMOAMO_by_herd()
        self.test_get_C910_without_PX_by_herd()
        self.test_get_C910_without_RAMO_by_herd()
        self.test_get_C910_without_WAMO_by_herd()
        self.test_get_C910_without_XP_by_herd()
        self.test_get_C910_without_XX_by_herd()

    def get_model_diff(self, log_file1, log_file2):

        dif_litmus_list = []
        model_1_herd_dict = {r.name: r.states for r in parse_herd_log(log_file1)}
        print(len(model_1_herd_dict))
        model_2_herd_dict = {r.name: r.states for r in parse_herd_log(log_file2)}

        for name in model_1_herd_dict:
            if name in model_2_herd_dict:
                if model_1_herd_dict[name] != model_2_herd_dict[name]:
                    dif_litmus_list.append(name)
        for name in model_2_herd_dict:
            if name in model_1_herd_dict:
                if model_2_herd_dict[name] != model_1_herd_dict[name]:
                    dif_litmus_list.append(name)
        dif_litmus_list = list(set(dif_litmus_list))
        print(len(dif_litmus_list))
        for litmus in dif_litmus_list:
            print(litmus)
        return dif_litmus_list

    def test_C910_diff_rvwmo(self):
        dif_litmus_list = self.get_model_diff(mm_suite['C910']['log'], mm_suite['RVWMO']['log'])
        print(len(dif_litmus_list))

    def test_C910_diff_PX(self):
        dif_litmus_list = self.get_model_diff(mm_suite['C910']['log'], mm_suite['C910_without_PX']['log'])
        print(len(dif_litmus_list))

    def test_C910_diff_XX(self):
        dif_litmus_list = self.get_model_diff(mm_suite['C910']['log'], mm_suite['C910_without_XX']['log'])
        print(len(dif_litmus_list))

    def test_C910_diff_XP(self):
        dif_litmus_list = self.get_model_diff(mm_suite['C910']['log'], mm_suite['C910_without_XP']['log'])
        print(len(dif_litmus_list))

    def test_C910_diff_AMOAMO(self):
        dif_litmus_list = self.get_model_diff(mm_suite['C910']['log'], mm_suite['C910_without_AMO']['log'])
        print(len(dif_litmus_list))


    def test_C910_diff_AMOW(self):
        dif_litmus_list = self.get_model_diff(mm_suite['C910']['log'], mm_suite['C910_without_AMOW']['log'])
        print(len(dif_litmus_list))

    def test_C910_diff_RAMO(self):
        dif_litmus_list = self.get_model_diff(mm_suite['C910']['log'], mm_suite['C910_without_RAMO']['log'])
        print(len(dif_litmus_list))

    def test_C910_diff_WAMO(self):
        dif_litmus_list = self.get_model_diff(mm_suite['C910']['log'], mm_suite['C910_without_WAMO']['log'])
        print(len(dif_litmus_list))

    def test_C910_Model(self):
        with open('output.txt','w') as f:
            original_stdout = sys.stdout
            sys.stdout = f

            config.init()
            config.set_var('reg_size', 64)

            litmus_test_suite = ['LB+addr+popx']
            litmus_test_suite = [search_file(name, f'{config.INPUT_DIR}/litmus', '.litmus') for name in litmus_test_suite]
            print(litmus_test_suite)
            target_mm = C910(plot_enabled=False)

            ppo_pool = get_C910_model()
            cat_file_name = os.path.join(config.CAT_DIR,'riscv-C910.cat')
            patches,_ = synth_by_test_pattern_online(target_mm, ppo_pool, litmus_test_suite,
                                                               cat_file_name)
            print('patches', patches)
            sys.stdout = original_stdout
            f.close()
