
"""Test for Litmus"""

import pytest

from src.slide.litmus import parse_litmus
from src.slide.utils.file_util import *

all_litmus_files = list_files('input/litmus', '.litmus')
manual_inputs = ['2+2W+fence.r.rw+fence.rw.rw']
classic_inputs = ['SB', 'MP+fence.rw.rw+ctrl']


# Auxiliary Function
def parse_litmus_by_name(name):
    """
    :param name: e.g., SB
    :return: Litmus
    """
    file = search_file(name, 'input/litmus', '.litmus')
    content = read_file(file)
    print(f'\nprocess {name}.litmus')
    return parse_litmus(content)


class TestLitmusParse:

    @pytest.mark.parametrize('filename', all_litmus_files)
    def test_parser_all(self, filename):
        content = read_file(filename)
        print(f'process {filename}')
        litmus = parse_litmus(content)
        print(str(litmus))

    @pytest.mark.parametrize('filename', all_litmus_files)
    def test_get_all_exceptional_aq_rl_cases(self, filename):
        content = read_file(filename)
        output_file = '../output/exceptional_aq_rl.txt'
        if 'lw.aq' in content or 'sw.rl' in content:
            name = filename.split("/")[-1]
            assert name.endswith('.litmus')
            name = name[:len(name) - len('.litmus')]
            write_str_to_file(output_file, f'{name}\n', True)

    def test_single_litmus(self):
        """
        Some tests may fail when executing test_parser_all(self, filename), so here we test one (failed) litmus file case at a time.
        ISA-S-DEP-DATA-SUCCESS.litmus;ISA10+BIS;ISA11+BIS;LR-SC-diff-loc1;
         ['SWAP-LR-SC', 'ISA-S-DEP-ADDR-SUCCESS', 'LB+addr+addrpx-poxp+VAR2', 'ISA-MP-DEP-ADDR-LR-SUCCESS',
                     'ISA03']
        'amoswap.w.aq.rl.litmus', 'lr.w.aq.rl.litmus', 'fence.tso.litmus', 'SC-FAIL.litmus',
                     'ISA12.litmus', 'CoWR0+fence.rw.rwspx.litmus', 'CoRW1+pospx.litmus', 'CoWW.litmus'
        """
        filenames = ['ISA11+BIS']
        for filename in filenames:
            litmus = parse_litmus_by_name(filename)
            print(str(litmus))

    def test_litmus_with_locations(self):
        """
        'ISA-Rel-Acq', 'ISA-2+2W-SUCCESS'
        """
        filenames = ['ISA-Rel-Acq']
        for filename in filenames:
            litmus = parse_litmus_by_name(filename)
            print(str(litmus))

    @pytest.mark.parametrize('name', manual_inputs)
    # case 1: same number of insts for each thread
    # case 2: different ...
    def test_pretty_printing(self, name):
        litmus = parse_litmus_by_name(name)
        print(str(litmus))

    @pytest.mark.parametrize('name', manual_inputs)
    def test_prog_manual(self, name):
        litmus = parse_litmus_by_name(name)
        print(str(litmus))

    @pytest.mark.parametrize('name', manual_inputs)
    def test_vars(self, name):
        litmus = parse_litmus_by_name(name)
        print(str(litmus))
        print(f"variables: {litmus.vars}")
        for var in litmus.vars:
            for i in range(litmus.n_threads):
                print(f"init addr of {var} in thread {i} is {litmus.get_init_var_addr(i, var)}")
                print(f"init value of {var} in thread {i} is {litmus.get_init_var_value(var)}")

    @pytest.mark.parametrize('name', manual_inputs)
    def test_regs(self, name):
        litmus = parse_litmus_by_name(name)
        print(f"registers: {litmus.regs}")
        print(f"input_regs: {litmus.input_regs}")
        print(f"addr_regs: {litmus.addr_regs}")
        print(f"out_regs: {litmus.out_regs}")
        print(f"trashed_regs: {litmus.trashed_regs}")
        # print(f"location_var_regs: {litmus.location_var_regs}")
        # for pid, reg in litmus.regs:
        #     print(f"init value for reg {reg} in thread {pid} is {litmus.get_init_reg_value(pid, reg)}")

    def test_SB(self):
        litmus = parse_litmus_by_name('non-mixed-size/RELAX/PodWR/SB')
        print(str(litmus))

        assert len(litmus.regs) == 8
        assert (0, 'x5') in litmus.regs
        assert (0, 'x6') in litmus.regs
        assert (0, 'x7') in litmus.regs
        assert (0, 'x8') in litmus.regs
        assert (1, 'x5') in litmus.regs
        assert (1, 'x6') in litmus.regs
        assert (1, 'x7') in litmus.regs
        assert (1, 'x8') in litmus.regs

        assert litmus.get_init_reg_value(0, 'x5') == 1
        assert litmus.get_init_reg_value(1, 'x5') == 1
        assert litmus.get_init_reg_value(2, 'x5') is None
        assert litmus.get_init_reg_value(0, 'x6') is None

        assert len(litmus.vars) == 2
        assert 'x' in litmus.vars
        assert 'y' in litmus.vars

        assert litmus.get_init_var_value('x') == 0
        assert litmus.get_init_var_value('y') == 0
        assert litmus.get_init_var_value('z') is None

        assert litmus.get_init_var_addr(0, 'x') == 'x6'
        assert litmus.get_init_var_addr(0, 'y') == 'x8'
        assert litmus.get_init_var_addr(1, 'x') == 'x8'
        assert litmus.get_init_var_addr(1, 'y') == 'x6'

        assert litmus.n_threads == 2

        assert len(litmus.out_regs) == 2
        assert (0, 'x7') in litmus.out_regs
        assert (1, 'x7') in litmus.out_regs

        assert len(litmus.input_regs) == 2
        assert len(litmus.addr_regs) == 4
        assert len(litmus.trashed_regs) == 0

    def test_Swap_Acqs(self):
        litmus = parse_litmus_by_name('2+2Swap+Acqs')
        print(str(litmus))

        assert len(litmus.out_regs) == 4
        assert len(litmus.vars) == 2
        assert len(litmus.trashed_regs) == 4
        assert len(litmus.addr_regs) == 4
        assert len(litmus.input_regs) == 0
