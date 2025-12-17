import sys


sys.path.append("../src")

from src.slide.synth.memory_relation import *
from src.slide.synth.ppo_def import PPOFlag, PPOInitFlag, PPOValidFlag, SinglePPO, get_ppo_item_by_str
from src.slide.synth.ppo_dict import get_rvwmo

class TestPPODef:

    def test_contain_AMO(self):
        ppo_1 = SinglePPO([R(),PoLoc(),R()])
        ppo_2 = SinglePPO([AMO(),Rfi(),R()])
        print(ppo_1.is_contain(ppo_2))
        print(ppo_2.is_contain(ppo_1))
    
    def test_ppo3(self):
        ppo_1 = SinglePPO([AMO(),PoLoc(),R()])
        ppo_2 = SinglePPO([AMO(),Rfi(),R()])
        print(ppo_1.is_contain(ppo_2))
        print(ppo_2.is_contain(ppo_1))

    def test_rvwmo(self):
        rvwmo = get_rvwmo()
        array = rvwmo.get_cat_form([PPOValidFlag.Valid], [PPOInitFlag.Init, PPOInitFlag.Verified], ppo_list = [])
        for item in array:
            print(item)

    def test_contain(self):
        rvwmo = get_rvwmo()
        array = ['[W];po;[R]',
                '[W];po;[W]',
                '[W];rfi;[R]',
                '[AMO];po;[R]',
                '[AMO];po;[W]',
                '[W];po-loc;[R]',
                '[AMO];po-loc;[R]',
                '[W];po;[R];po;[W]',
                '[W];rfi;[R];po;[W]',
                '[AMO];po;[W];po;[R]',
                '[AMO];po;[W];po;[W]',
                '[AMO];po;[R];po;[W]',
                '[W];po;[R];addr;[W]',
                '[AMO];po;[W];rfi;[R]',
                '[AMO];coi;[W];po;[R]',
                '[AMO];coi;[W];po;[W]',
                '[W];rfi;[R];addr;[W]',
                '[AMO];coi;[W];rfi;[R]',
                '[AMO];po;[R];addr;[W]',
                '[W];po-loc;[R];po;[W]',
                '[AMO];po;[W];po-loc;[R]',
                '[AMO];po-loc;[W];po;[R]',
                '[AMO];po-loc;[W];po;[W]',
                '[AMO];po-loc;[R];po;[W]',
                '[W];po-loc;[R];addr;[W]',
                '[AMO];po-loc;[W];rfi;[R]',
                '[AMO];coi;[W];po-loc;[R]',
                '[AMO];po-loc;[R];addr;[W]',
                '[AMO];po;[W];po;[R];po;[W]',
                '[AMO];po-loc;[W];po-loc;[R]',
                '[AMO];po;[W];rfi;[R];po;[W]',
                '[AMO];coi;[W];po;[R];po;[W]',
                '[AMO];po;[W];po;[R];addr;[W]',
                '[AMO];coi;[W];rfi;[R];po;[W]',
                '[AMO];po;[W];rfi;[R];addr;[W]',
                '[AMO];coi;[W];po;[R];addr;[W]',
                '[AMO];po;[W];po-loc;[R];po;[W]',
                '[AMO];po-loc;[W];po;[R];po;[W]',
                '[AMO];coi;[W];rfi;[R];addr;[W]',
                '[AMO];po-loc;[W];rfi;[R];po;[W]',
                '[AMO];coi;[W];po-loc;[R];po;[W]',
                '[AMO];po;[W];po-loc;[R];addr;[W]',
                '[AMO];po-loc;[W];po;[R];addr;[W]',
                '[AMO];po-loc;[W];rfi;[R];addr;[W]',
                '[AMO];coi;[W];po-loc;[R];addr;[W]',
                '[AMO];po-loc;[W];po-loc;[R];po;[W]',
                '[AMO];po-loc;[W];po-loc;[R];addr;[W]',]
        # array = ['[R];po;[R]',
        #         '[W];po;[R]',
        #         '[W];rfi;[R]',
        #         '[R];po-loc;[R]',
        #         '[W];po-loc;[R]',
        #         '[R];po;[W];po;[R]',
        #         '[R];po;[R];po;[R]',
        #         '[W];po;[R];po;[R]',
        #         '[R];po;[W];rfi;[R]',
        #         '[R];fri;[W];po;[R]',
        #         '[W];rfi;[R];po;[R]',
        #         '[R];fri;[W];rfi;[R]',
        #         '[R];po;[R];addr;[R]',
        #         '[W];po;[R];addr;[R]',
        #         '[W];rfi;[R];addr;[R]',
        #         '[R];po;[W];po-loc;[R]',
        #         '[R];po-loc;[W];po;[R]',
        #         '[R];po-loc;[R];po;[R]',
        #         '[W];po-loc;[R];po;[R]',
        #         '[R];po-loc;[W];rfi;[R]',
        #         '[R];fri;[W];po-loc;[R]',
        #         '[R];po-loc;[R];addr;[R]',
        #         '[W];po-loc;[R];addr;[R]',
        #         '[R];po;[W];po;[R];po;[R]',
        #         '[R];po-loc;[W];po-loc;[R]',
        #         '[R];po;[W];rfi;[R];po;[R]',
        #         '[R];fri;[W];po;[R];po;[R]',
        #         '[R];po;[W];po;[R];addr;[R]',
        #         '[R];fri;[W];rfi;[R];po;[R]',
        #         '[R];po;[W];rfi;[R];addr;[R]',
        #         '[R];fri;[W];po;[R];addr;[R]',
        #         '[R];po;[W];po-loc;[R];po;[R]',
        #         '[R];po-loc;[W];po;[R];po;[R]',
        #         '[R];fri;[W];rfi;[R];addr;[R]',
        #         '[R];po-loc;[W];rfi;[R];po;[R]',
        #         '[R];fri;[W];po-loc;[R];po;[R]',
        #         '[R];po;[W];po-loc;[R];addr;[R]',
        #         '[R];po-loc;[W];po;[R];addr;[R]',
        #         '[R];po-loc;[W];rfi;[R];addr;[R]',
        #         '[R];fri;[W];po-loc;[R];addr;[R]',
        #         '[R];po-loc;[W];po-loc;[R];po;[R]',
        #         '[R];po-loc;[W];po-loc;[R];addr;[R]',]
        can_relax_ppo_list = []
        for ppo_str in array:
            ppo = get_ppo_item_by_str(ppo_str, PPOFlag.Relaxed)
            rvwmo.add_ppo(ppo)
            contain_flag, can_relax_flag = rvwmo.check_contain_ppo(ppo)
            if(can_relax_flag):
                can_relax_ppo_list.append(ppo)
        
        for ppo in can_relax_ppo_list:
            print('can relax',ppo)