
"""Test for Litmus"""
import os.path
from datetime import datetime
import sys

sys.path.append("../src")
from src.slide.synth.diy7_generator import Cycle, Diy7Generator
from src.slide.synth.memory_relation import *
from src.slide.synth.ppo_def import SinglePPO, PPOFlag
from src.slide import config

from src.slide.utils.file_util import *

class TestDiy7Generator:


    def test_diy7_ppo3_Sc(self):
        with open('rwvmo.txt', 'w') as f:
            sys.stdout = f
            diy7Generator = Diy7Generator()
            print('test ppo3:Lr;Rmw;Sc;rfi;R')
            ppo = SinglePPO([Lr(),Rmw(),Sc(),Rfi(),R()])
            before_cat_file = f'{config.INPUT_DIR}/CAT/riscv-remove-ppo-3.cat'
            after_cat_file = f'{config.INPUT_DIR}/CAT/riscv-complete.cat'
            diy7Generator.set_ppo(ppo, before_cat_file = before_cat_file, after_cat_file = after_cat_file)
            diy7Generator.init_cycle_list()
            cycle, content, name= diy7Generator.generate_litmus_test_legal()
            print(cycle)
            print(content)
            print(name)

    def test_diy7_ppo2_rsw(self):
        with open('rwvmo.txt', 'w') as f:
            sys.stdout = f
            diy7Generator = Diy7Generator()
            print('test ppo3:R;rsw;R')
            ppo = SinglePPO([R(),Rsw(),R()], ppo_flag=PPOFlag.Relaxed)
            before_cat_file = f'{config.INPUT_DIR}/CAT/change/riscv-0.cat'
            after_cat_file = f'{config.INPUT_DIR}/CAT/change/riscv-1.cat'
            diy7Generator.set_ppo(ppo, before_cat_file = before_cat_file, after_cat_file = after_cat_file)
            diy7Generator.init_cycle_list()
            cycle, content, name= diy7Generator.generate_litmus_test_legal()
            print(cycle)
            print(content)
            print(name)




