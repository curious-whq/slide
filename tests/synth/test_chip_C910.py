import os

from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

from src.slide import config
from src.slide.analysis.model.C910 import C910
from src.slide.analysis.model.C910_AMO import C910_AMO
from src.slide.comp.parse_result import parse_chip_log, parse_herd_log
from src.slide.litmus.load_litmus import get_litmus_by_policy, GetLitmusPolicy
from src.slide.validate.validate import validate, get_dif_detail

filter_litmus_list = [
    os.path.join(config.INPUT_DIR,'chip_execution_logs/exceed_two_threads.txt'),
    os.path.join(config.INPUT_DIR,'chip_execution_logs/exceed_4_access.txt'),
    os.path.join(config.INPUT_DIR,'chip_execution_logs/exceptional.txt'),
    os.path.join(config.INPUT_DIR,'chip_execution_logs/fence.i_ctrlfencei.txt'),
    os.path.join(config.INPUT_DIR, 'chip_execution_logs/exceed_two_threads_has_X.txt'),
    os.path.join(config.INPUT_DIR, 'chip_execution_logs/X_exceed_four.txt')
]

chip_log_list = {
    'C910':f'{config.INPUT_DIR}/chip_execution_logs/C910/chip_log.txt'
}
herd_log_list = {
    'RVWMO':f'{config.INPUT_DIR}/herd/herd_results_rvwmo.log',
    'C910':f'{config.INPUT_DIR}/herd/herd_results_C910.log',
    'C910_without_AMOAMO': f'{config.INPUT_DIR}/herd/herd_results_C910_without_AMOAMO.log',
    'C910_without_AMOW': f'{config.INPUT_DIR}/herd/herd_results_C910_without_AMOW.log',
    'C910_without_PX': f'{config.INPUT_DIR}/herd/herd_results_C910_without_PX.log',
    'C910_without_RAMO': f'{config.INPUT_DIR}/herd/herd_results_C910_without_RAMO.log',
    'C910_without_WAMO': f'{config.INPUT_DIR}/herd/herd_results_C910_without_WAMO.log',
    'C910_without_XP': f'{config.INPUT_DIR}/herd/herd_results_C910_without_XP.log',
    'C910_without_XX': f'{config.INPUT_DIR}/herd/herd_results_C910_without_XX.log',

}
sp_list = [
            'ISA-DEP-SUCCESS',
            'ISA-LB-DEP-DATA-SUCCESS',
            'ISA18',
            'ISA-S-DEP-DATA-SUCCESS',
            'ISA17',
            'ISA01',
            'Andy26',
            'ISA10',
            'ISA-MP-DEP-SUCCESS',
            'ISA11+BIS',
            'ISA-2+2W-SUCCESS',
            'LR-SC-diff-loc2',
            'Andy25',
            'S+po+ctrl'
            'LB+addr+po'
            'MP+po+addr',
            'R+poss',
            'RSW',
            'PPOCA',
            'ISA-LB-DEP-ADDR-SUCCESS',
            'ISA-LB-DEP-ADDR2-SUCCESS',
            'ISA-LB-DEP-ADDR3-SUCCESS',
            'ISA-MP-DEP-ADDR-LR-FAIL',
            'ISA-MP-DEP-ADDR-LR-SUCCESS',
            'ISA-S-DEP-ADDR-SUCCESS',
            'ISA16',
            'ISA18'
        ]
class TestC910Chip:

    def get_chip_diff_herd(self, chip_log_path, herd_log_path, detail=False, detail_file=None, mm=None):
        dif_litmus_list = []
        error_litmus_list = []
        filter_litmus_dict = {}
        for litmus_file in filter_litmus_list:
            with open(litmus_file, 'r') as f:
                filter_list = f.readlines()
                for filter_name in filter_list:
                    filter_name = filter_name.strip().replace('.litmus', '')
                    filter_litmus_dict[filter_name] = 1
        for sp_litmus in sp_list:
            filter_litmus_dict[sp_litmus] = 1
        print(len(filter_litmus_dict))
        chip_log = {r.name: r.states for r in parse_chip_log(chip_log_path)}

        herd_log = {r.name: r.states for r in parse_herd_log(herd_log_path)}
        for key in chip_log:
            if key in herd_log:
                if key in filter_litmus_dict:
                    continue
                if herd_log[key] != chip_log[key]:
                    if len(herd_log[key]) > len(chip_log[key]):
                        dif_litmus_list.append(key)
                    elif len(herd_log[key]) < len(chip_log[key]):
                        error_litmus_list.append(key)
        # print(len(dif_litmus_list))
        # for litmus in dif_litmus_list:
        #     print(litmus)
        # dif_litmus_list = [litmus for litmus in dif_litmus_list if 'fence.rw.w' not in litmus]
        # dif_litmus_list = [litmus for litmus in dif_litmus_list if 'fence.rw.r' not in litmus]
        # dif_litmus_list = [litmus for litmus in dif_litmus_list if 'fence.w.rw' not in litmus]
        # dif_litmus_list = [litmus for litmus in dif_litmus_list if 'fence.r.rw' not in litmus]
        # dif_litmus_list = [litmus for litmus in dif_litmus_list if 'fence.r.r' not in litmus]
        # dif_litmus_list = [litmus for litmus in dif_litmus_list if 'fence.r.w' not in litmus]
        # dif_litmus_list = [litmus for litmus in dif_litmus_list if 'fence.w.w' not in litmus]
        # dif_litmus_list = [litmus for litmus in dif_litmus_list if 'fence.w.r' not in litmus]
        # dif_litmus_list = [litmus for litmus in dif_litmus_list if 'fence.tso' not in litmus]
        # dif_litmus_list = [litmus for litmus in dif_litmus_list if not ('x' in litmus and 's' in litmus) ]
        # dif_litmus_list = [litmus for litmus in dif_litmus_list if 'NEW' not in litmus]
        if detail:
           get_dif_detail(mm, chip_log, herd_log, dif_litmus_list, detail_file)
        return dif_litmus_list, error_litmus_list

    def collect_all_chip_log(self, chip_log_path, collect_paths):
        with open(chip_log_path, 'a+') as wf:
            for collect_path in collect_paths:
                with open(collect_path, 'r') as rf:
                    while chunk := rf.read(1024):
                        wf.write(chunk)

    def test_collect_chip_log(self):
        collect_paths = [
            f'{config.INPUT_DIR}/chip_execution_logs/C910/chip_log_fix1.txt',
            f'{config.INPUT_DIR}/chip_execution_logs/C910/chip_log_fix2.txt',
            f'{config.INPUT_DIR}/chip_execution_logs/C910/chip_log_fix3.txt',
            f'{config.INPUT_DIR}/chip_execution_logs/C910/chip_log_fix4.txt',
            f'{config.INPUT_DIR}/chip_execution_logs/C910/chip_log_fix5.txt',
            f'{config.INPUT_DIR}/chip_execution_logs/C910/chip_log_fix7.txt',
            f'{config.INPUT_DIR}/chip_execution_logs/C910/chip_log_1000000.txt',
        ]
        self.collect_all_chip_log(chip_log_list['C910'], collect_paths)


    def test_C910_chip_diff_herd_rvwmo(self):
        file_path = os.path.join(config.OUTPUT_DIR,'dif_log/herd-rvwmo-chip-C910')
        dif_litmus_list, error_litmus_list = self.get_chip_diff_herd(chip_log_list['C910'], herd_log_list['RVWMO'],
                                                  detail=False, detail_file=file_path, mm=C910_AMO())
        print(len(dif_litmus_list))

        for litmus in dif_litmus_list:
            print(litmus)

        print(len(error_litmus_list))
        for litmus in error_litmus_list:
            print(litmus)


    def test_C910_chip_diff_herd(self):
        file_path = os.path.join(config.OUTPUT_DIR,'dif_log/herd-C910-chip-C910')
        dif_litmus_list, error_litmus_list = self.get_chip_diff_herd(chip_log_list['C910'], herd_log_list['C910'],
                                                  detail=False, detail_file=file_path, mm=C910_AMO())
        print(len(dif_litmus_list))

        for litmus in dif_litmus_list:
            print(litmus)

        print(len(error_litmus_list))
        for litmus in error_litmus_list:
            print(litmus)

    def test_C910_chip_diff_herd_AMOAMO(self):
        file_path = os.path.join(config.OUTPUT_DIR,'dif_log/herd-C910-chip-C910-AMOAMO')
        dif_litmus_list, error_litmus_list = self.get_chip_diff_herd(chip_log_list['C910'], herd_log_list['C910_without_AMOAMO'],
                                                  detail=False, detail_file=file_path, mm=C910_AMO())
        print(len(dif_litmus_list))

        for litmus in dif_litmus_list:
            print(litmus)

        print(len(error_litmus_list))
        for litmus in error_litmus_list:
            print(litmus)


    def test_C910_chip_diff_herd_AMOW(self):
        file_path = os.path.join(config.OUTPUT_DIR,'dif_log/herd-C910-chip-C910-AMOW')
        dif_litmus_list, error_litmus_list = self.get_chip_diff_herd(chip_log_list['C910'], herd_log_list['C910_without_AMOW'],
                                                  detail=False, detail_file=file_path, mm=C910_AMO())
        print(len(dif_litmus_list))

        for litmus in dif_litmus_list:
            print(litmus)

        print(len(error_litmus_list))
        for litmus in error_litmus_list:
            print(litmus)


    def test_C910_chip_diff_herd_PX(self):
        file_path = os.path.join(config.OUTPUT_DIR,'dif_log/herd-C910-chip-C910-PX')
        dif_litmus_list, error_litmus_list = self.get_chip_diff_herd(chip_log_list['C910'], herd_log_list['C910_without_PX'],
                                                  detail=False, detail_file=file_path, mm=C910_AMO())
        print(len(dif_litmus_list))

        for litmus in dif_litmus_list:
            print(litmus)

        print(len(error_litmus_list))
        for litmus in error_litmus_list:
            print(litmus)


    def test_C910_chip_diff_herd_RAMO(self):
        file_path = os.path.join(config.OUTPUT_DIR,'dif_log/herd-C910-chip-C910-RAMO')
        dif_litmus_list, error_litmus_list = self.get_chip_diff_herd(chip_log_list['C910'], herd_log_list['C910_without_RAMO'],
                                                  detail=False, detail_file=file_path, mm=C910_AMO())
        print(len(dif_litmus_list))

        for litmus in dif_litmus_list:
            print(litmus)

        print(len(error_litmus_list))
        for litmus in error_litmus_list:
            print(litmus)

    def test_C910_chip_diff_herd_WAMO(self):
        file_path = os.path.join(config.OUTPUT_DIR,'dif_log/herd-C910-chip-C910-WAMO')
        dif_litmus_list, error_litmus_list = self.get_chip_diff_herd(chip_log_list['C910'], herd_log_list['C910_without_WAMO'],
                                                  detail=False, detail_file=file_path, mm=C910_AMO())
        print(len(dif_litmus_list))

        for litmus in dif_litmus_list:
            print(litmus)

        print(len(error_litmus_list))
        for litmus in error_litmus_list:
            print(litmus)

    def test_C910_chip_diff_herd_XP(self):
        file_path = os.path.join(config.OUTPUT_DIR,'dif_log/herd-C910-chip-C910-XP')
        dif_litmus_list, error_litmus_list = self.get_chip_diff_herd(chip_log_list['C910'], herd_log_list['C910_without_XP'],
                                                  detail=False, detail_file=file_path, mm=C910_AMO())
        print(len(dif_litmus_list))

        for litmus in dif_litmus_list:
            print(litmus)

        print(len(error_litmus_list))
        for litmus in error_litmus_list:
            print(litmus)

    def test_C910_chip_diff_herd_XX(self):
        file_path = os.path.join(config.OUTPUT_DIR,'dif_log/herd-C910-chip-C910-XX')
        dif_litmus_list, error_litmus_list = self.get_chip_diff_herd(chip_log_list['C910'], herd_log_list['C910_without_XX'],
                                                  detail=False, detail_file=file_path, mm=C910_AMO())
        print(len(dif_litmus_list))

        for litmus in dif_litmus_list:
            print(litmus)

        print(len(error_litmus_list))
        for litmus in error_litmus_list:
            print(litmus)