
"""Test for RISC-V Program Parser"""
from src.slide.analysis import CFG, path_to_ssa, cfg_to_ssa
from src.slide.prog import *


class TestProg:
    def test1(self):
        text = 'li x1, 5' \
                'addi x2, x1, 1' \
                'add x2, x2, x2'
        program = parse_program(text)
        print(program.insts)

    def test2(self):
        text = 'li x1, 5' \
               'addi x2, x1, 1' \
               'add x2, x2, x2'
        program = parse_program(text)
        cfg = CFG(program.insts)
        paths = cfg.find_all_paths()
        for p in paths:
            ssa = path_to_ssa(p)
            print(ssa)

    def test3(self):
        # TODO:
        text = 'li x1, 5' \
               'addi x2, x1, 1' \
               'add x2, x2, x2'
        program = parse_program(text)
        cfg = CFG(program.insts)
        i2s = cfg_to_ssa(cfg)
        for i in program.insts:
            print(f'{i} -> {i2s[i]}')