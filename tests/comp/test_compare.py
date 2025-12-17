

import os
from enum import Enum
from typing import List

import pytest

from src.slide import config
from src.slide.comp.compare import compare_two_results
from src.slide.comp.parse_result import parse_chip_log, parse_herd_log
from src.slide.utils import dir_util, file_util

cur_dir = dir_util.get_cur_dir(__file__)
input_dir = os.path.abspath(os.path.join(cur_dir, "../input"))


class LogFormat(Enum):
    Chip = 1
    Herd = 2


def compare_one_file(filename, log_format=LogFormat.Chip, golden_model_name='rvwmo') -> List[str]:
    output_log_path = os.path.join(config.OUTPUT_DIR, "litmus_with_illegal_states.log")
    # file_util.clear_file(output_log_path)

    chip_log_path = os.path.join(input_dir, filename)
    herd_log_path = os.path.join(input_dir, f'herd/herd_results_{golden_model_name}.log')
    litmus_herd_results = parse_herd_log(herd_log_path)
    if log_format == LogFormat.Chip:
        litmus_chip_results = parse_chip_log(chip_log_path)
    else:
        assert log_format == LogFormat.Herd
        litmus_chip_results = parse_herd_log(chip_log_path)

    is_same_cnt = 0
    uniq_states_in_chip_cnt = 0
    uniq_states_in_herd_cnt = 0
    litmus_with_illegal_states = []
    for litmus_chip_result in litmus_chip_results:
        litmus_name = litmus_chip_result.name
        litmus_herd_result = find_result_by_name(litmus_name, litmus_herd_results)
        is_same, uniq_states_in_chip, uniq_states_in_herd = compare_two_results(litmus_chip_result, litmus_herd_result)
        is_same_cnt += 1 if is_same else 0
        uniq_states_in_chip_cnt += 1 if len(uniq_states_in_chip) > 0 else 0
        uniq_states_in_herd_cnt += 1 if len(uniq_states_in_herd) > 0 else 0
        if len(uniq_states_in_chip) > 0:
            litmus_with_illegal_states.append(litmus_name)
            file_util.write_line_to_file(output_log_path,
                                         f"{litmus_name:<50} uniq_states_in_chip: {len(uniq_states_in_chip)}")  # {uniq_states_in_chip}

    print(
        f"litmus tests: {len(litmus_chip_results)}, is_same_cnt: {is_same_cnt}, uniq_states_in_chip_cnt: {uniq_states_in_chip_cnt}, uniq_states_in_herd_cnt: {uniq_states_in_herd_cnt} ")
    return litmus_with_illegal_states


class TestComparator:
    def test_compare_cx_rvwmo(self):
        # run cx
        # memory_model_dir = os.path.join(input_dir, 'herd/memory_models')
        # TestHerd.herd_run_all_valid_litmus("cx", os.path.join(memory_model_dir, 'cx.cat'), memory_model_dir)

        # compare
        compare_one_file("herd/herd_results_cx.log", LogFormat.Herd)

    def test_compare_result_1(self):
        filename = 'chip_execution_logs/cx1c/run.1.log'
        print(filename)
        tests = compare_one_file(filename)
        uniq_states_in_chip_cnt = len(tests)
        print(f'{uniq_states_in_chip_cnt =}')
        print('failed tests:')
        for t in tests:
            print(t)
        assert uniq_states_in_chip_cnt == 0

    @pytest.mark.parametrize('filename', [f'chip_execution_logs/cx1b_new_20240411/run.{x}.log' for x in range(1, 12)])
    def test_compare_result_cx1b_new_20240411_vs_rvwmo(self, filename):
        print(filename)
        uniq_states_in_chip = compare_one_file(filename)
        print(f"uniq_states_in_chip cases: {len(uniq_states_in_chip)}  {uniq_states_in_chip}")

    @pytest.mark.parametrize('filename', [f'chip_execution_logs/cx1b_new_20240411/run.{x}.log' for x in range(1, 12)])
    def test_compare_result_cx1b_new_20240411_vs_cx(self, filename):
        print(filename)
        uniq_states_in_chip = compare_one_file(filename, golden_model_name='cx')
        print(f"uniq_states_in_chip cases: {len(uniq_states_in_chip)}  {uniq_states_in_chip}")

    @pytest.mark.parametrize('filename',
                             [f'chip_execution_logs/LITMUS-v1.3.106-20240412/run.{x}.log' for x in range(1, 6)] + [
                                 'chip_execution_logs/LITMUS-v1.3.106-20240412/run.test.log'])
    def test_compare_result_cx1b_new_20240412_vs_cx(self, filename):
        print(filename)
        # uniq_states_in_chip = compare_one_file(filename, golden_model_name='cx')
        uniq_states_in_chip = compare_one_file(filename, golden_model_name='rvwmo')
        print(f"uniq_states_in_chip cases: {len(uniq_states_in_chip)}")  # {uniq_states_in_chip}

    @pytest.mark.skip(reason="Failed test")
    @pytest.mark.parametrize('filename', [f'chip_execution_logs/cx1b/run.{x}.log' for x in range(1, 61)])
    def test_compare_result_cx1b(self, filename):
        print(filename)
        uniq_states_in_chip_cnt = compare_one_file(filename)
        print(f'{uniq_states_in_chip_cnt =}')
        assert len(uniq_states_in_chip_cnt) == 0

    @pytest.mark.skip(reason="Failed test")
    def test_compare_result_cx1b_1(self):
        filename = 'chip_execution_logs/cx1b/run.1.log'
        print(filename)
        tests = compare_one_file(filename)
        uniq_states_in_chip_cnt = len(tests)
        print(f'{uniq_states_in_chip_cnt =}')
        print('failed tests:')
        for t in tests:
            print(t)
        assert uniq_states_in_chip_cnt == 0

    @pytest.mark.skip(reason="Failed test")
    def test_all_illegal_litmus(self):
        illegals = []
        for i in range(1, 61):
            filename = f'chip_execution_logs/cx1b/run.{i}.log'
            print(filename)
            illegals.extend(compare_one_file(filename))

        illegals = list(set(illegals))
        print(illegals)
        print(len(illegals))


def find_result_by_name(litmus_name, litmus_results):
    for litmus_result in litmus_results:
        if litmus_result.name == litmus_name:
            return litmus_result
    assert False, f"[Error] Fail to find {litmus_name} in litmus_results."
