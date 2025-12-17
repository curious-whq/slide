
#
"""Test for Litmus"""
from typing import List
#pytest -s analysis/test_mm.py::TestMemoryModel::test_MP_fence_w_w_fri_rfi_addr
import pytest
import sys
sys.path.append("../../src")
from src.slide import config
from src.slide.analysis.model import *
from src.slide.comp.parse_result import parse_herd_log
from src.slide.litmus import parse_litmus
from src.slide.utils.file_util import *

all_litmus_files = list_files(f'{config.INPUT_DIR}/litmus', '.litmus')
output_dir = f'{config.OUTPUT_DIR}'

exceptional_aqrl = read_file(f'{config.INPUT_DIR}/herd/exceptional_aq_rl.txt').split('\n')

TARGET_MEMORY_MODEL = 'rvwmo'
model_class = {
    'rvwmo': RVWMO,  # 2023-12-24: 19s
    'rvtso': RVTSO,  # 2023-12-24: 11s
    'sc': SC,  # 2023-12-24: 7s
}


# Auxiliary Function
def parse_litmus_by_name(name):
    """
    :param name: e.g., SB
    :return: Litmus
    """
    file = search_file(name, f'{config.INPUT_DIR}/litmus', '.litmus')
    content = read_file(file)
    print(f'\nprocess {name}.litmus')
    return parse_litmus(content)


def run_litmus(litmus_name, plot_enabled: bool = False) -> List:
    print('-' * 50 + f'\nRun {litmus_name} with {TARGET_MEMORY_MODEL}')
    litmus = parse_litmus_by_name(litmus_name)
    mm = model_class[TARGET_MEMORY_MODEL](plot_enabled=plot_enabled)
    mm.run(litmus)
    print('state',mm.states)
    return mm.states


log_path = f'{config.INPUT_DIR}/herd/herd_results_{TARGET_MEMORY_MODEL}.log'
herd_logs = {r.name: r.states for r in parse_herd_log(log_path) if r.name not in exceptional_aqrl}


def compare_states(states1, states2):
    assert len(states1) == len(states2)
    for i in range(len(states1)):
        assert states1[i] == states2[i]


class TestMemoryModel:
    # 2023-12-19: 2m31s
    # 2023-12-19: 1m18s
    # 2023-12-19: 1m7s
    # 2023-12-20: 34s
    # 2024-01-04: 19s (python 3.10 -> 3.11)
    def setup_method(self):
        print("\nSetup...\n")
        config.init()
        config.set_var('reg_size', 64)

    def teardown_method(self):
        print("\nTearDown...\n")
        config.reset()

    @classmethod
    def setup_class(cls):
        # remove all files in output dir
        remove_files(output_dir)

    @pytest.mark.skip(reason="Skip test for all")
    # @pytest.mark.parametrize('name, state_cnt', herd_logs.items())
    def test_one(self, name, state_cnt):
        # 2023-12-24: TIMEOUT=60s N=16
        #   rvtso-3m46s | timeout: ISA03
        #   sc-2m20s | timeout: ISA03
        #   rvwmo-6m53s | timeout: ISA03+SB02, WWC+posxxs, ISA03, SB+pos-po-ctrlfenceiss,
        #                           SB+pos-po-ctrlfenceis_pos-po-ctrlfencei, SB+pos-po-ctrlfenceis
        if name in exceptional_aqrl:
            return
        if name in ['ISA-LB-DEP-ADDR-SUCCESS',
                    'ISA-LB-DEP-ADDR2-SUCCESS',
                    'ISA-LB-DEP-ADDR3-SUCCESS',
                    'ISA-MP-DEP-ADDR-LR-FAIL',
                    'ISA-MP-DEP-ADDR-LR-SUCCESS',
                    'ISA-S-DEP-ADDR-SUCCESS',
                    'ISA16',
                    'ISA18']:
            # pointer to pointer
            return
        assert run_litmus(name) == state_cnt, f'test fail: {name}'

        print('PASS\n' + '-' * 50)

    # @pytest.mark.skip(reason="It takes too long")
    def test_3_LB_ctrls(self):
        # 2023.12.9 22:37: 44s
        # 2023.12.9 23:40: 8s
        assert run_litmus('3.LB+ctrls') == herd_logs['3.LB+ctrls']

    def test_ISA17(self):
        assert run_litmus('ISA17') == herd_logs['ISA17']

    def test_ISA_DEP_ADDR(self):
        assert run_litmus('ISA-DEP-ADDR') == herd_logs['ISA-DEP-ADDR']

    def test_2_2W_po_poarar_NEW(self):
        assert run_litmus('2+2W+po+poarar+NEW') == herd_logs['2+2W+po+poarar+NEW']
 
    def test_test_0_5(self):
        run_litmus('new_test_0_5')

    def test_RSW(self):
        run_litmus('RSW')

    def test_PPO13(self):
        run_litmus('LB+data+addr-fri-rfi-addr')

    def test_bad(self):
        with open('rwvmo.txt', 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            # run_litmus('ForwardAMO')
            run_litmus('ForwardSc')
    
    def test_MP_fence_w_w_fri_rfi_addr(self):
        run_litmus('MP+fence.w.w+fri-rfi-addr')

    def test_CoRW2(self):
        assert run_litmus('CoRW2') == herd_logs['CoRW2']

    def test_2_2W_fence_r_rw_fence_rw_rw(self):
        assert run_litmus('2+2W+fence.r.rw+fence.rw.rw') == herd_logs['2+2W+fence.r.rw+fence.rw.rw']

    def test_2_2W_fence_r_rw_fence_rw_w(self):
        assert run_litmus('2+2W+fence.r.rw+fence.rw.w') == herd_logs['2+2W+fence.r.rw+fence.rw.w']

    def test_ISA12(self):
        assert run_litmus('ISA12') == herd_logs['ISA12']

    def test_CoRW1_fence_rw_rwsxp(self):
        assert run_litmus('CoRW1+fence.rw.rwsxp') == herd_logs['CoRW1+fence.rw.rwsxp']

    def test_CoRW1_pospx(self):
        assert run_litmus('CoRW1+pospx') == herd_logs['CoRW1+pospx']

    def test_SC_FAIL(self):
        assert run_litmus('SC-FAIL') == herd_logs['SC-FAIL']

    def test_SB(self):
        assert run_litmus('SB') == herd_logs['SB']

    def test_SB_plot(self):
        assert run_litmus('SB', plot_enabled=True) == herd_logs['SB']

    def test_2_2W_fence_rw_rws_pos(self):
        assert run_litmus('2+2W+fence.rw.rws+pos') == herd_logs['2+2W+fence.rw.rws+pos']

    def test_CoRW1_fence_rw_rws(self):
        assert run_litmus('CoRW1+fence.rw.rws') == herd_logs['CoRW1+fence.rw.rws']

    def test_CoRW2_fence_rw_rws_X(self):
        assert run_litmus('CoRW2+fence.rw.rws+X') == herd_logs['CoRW2+fence.rw.rws+X']

    def test_SB_popaqs_NEW(self):
        # 2023.12.9 22:16: 40s
        # 2023.12.9 22:26: 7s
        # 2023.12.9 22:32: 6s
        # 2023.12.10 18:06: 1.7s
        assert run_litmus('SB+popaqs+NEW') == herd_logs['SB+popaqs+NEW']

    def test_CoRW1(self):
        assert run_litmus('CoRW1') == herd_logs['CoRW1']

    def test_CoWR0(self):
        assert run_litmus('CoWR0') == herd_logs['CoWR0']

    def testSB_fence_w_wprlxs(self):
        assert run_litmus('SB+fence.w.wprlxs') == herd_logs['SB+fence.w.wprlxs']

    def test_LR_SC_diff_loc1(self):
        assert run_litmus('LR-SC-diff-loc1') == herd_logs['LR-SC-diff-loc1']

    def test_SWAP_LR_SC(self):
        assert run_litmus('SWAP-LR-SC') == herd_logs['SWAP-LR-SC']

    def test_Andy25(self):
        assert run_litmus('Andy25') == herd_logs['Andy25']

    def test_CoRR_fence_rw_rwsxx(self):
        assert run_litmus('CoRR+fence.rw.rwsxx') == herd_logs['CoRR+fence.rw.rwsxx']

    def test_ISA_DEP_WR_ADDR(self):
        assert run_litmus('ISA-DEP-WR-ADDR') == herd_logs['ISA-DEP-WR-ADDR']

    def test_ISA_DEP_WW_ADDR(self):
        assert run_litmus('ISA-DEP-WW-ADDR') == herd_logs['ISA-DEP-WW-ADDR']

    def test_ISA_DEP_WW_CTRL(self):
        assert run_litmus('ISA-DEP-WW-CTRL') == herd_logs['ISA-DEP-WW-CTRL']

    @pytest.mark.skip(reason="pointer to pointer")
    def test_ISA16(self):
        assert run_litmus('ISA16') == herd_logs['ISA16']

    @pytest.mark.skip(reason="pointer to pointer")
    def test_ISA18(self):
        assert run_litmus('ISA18') == herd_logs['ISA18']

    # @pytest.mark.skip(reason="ppo-r13 with failed sc")
    def test_LB_addr_addrpx_poxp_VAR(self):
        assert run_litmus('LB+addr+addrpx-poxp+VAR') == herd_logs['LB+addr+addrpx-poxp+VAR']

    def test_LB_addr_addrpx_poxp_VAR2(self):
        assert run_litmus('LB+addr+addrpx-poxp+VAR2') == herd_logs['LB+addr+addrpx-poxp+VAR2']

    @pytest.mark.skip(reason="pointer to pointer")
    def test_ISA_LB_DEP_ADDR_SUCCESS(self):
        assert run_litmus('ISA-LB-DEP-ADDR-SUCCESS') == herd_logs['ISA-LB-DEP-ADDR-SUCCESS']

    @pytest.mark.skip(reason="pointer to pointer")
    def test_ISA_LB_DEP_ADDR3_SUCCESS(self):
        assert run_litmus('ISA-LB-DEP-ADDR3-SUCCESS') == herd_logs['ISA-LB-DEP-ADDR3-SUCCESS']

    def test_LB_addr_fri_rfi_addr(self):
        assert run_litmus('LB+addr+fri-rfi-addr') == herd_logs['LB+addr+fri-rfi-addr']

    def test_LB_addr_rfi_data_addr_rfi_ctrl(self):
        # 2023.12.14 18:50: 45s
        # 2023.12.14 19:22: 30s (optimize rf)
        # 2023.12.14 19:53: 17s (optimize co)
        # 2023.12.19 17:36: 8s (partial order)
        # 2023.12.20 08:59: 6s (global location value)
        assert run_litmus('LB+addr-rfi-data+addr-rfi-ctrl') == herd_logs['LB+addr-rfi-data+addr-rfi-ctrl']

    def test_RDW(self):
        assert run_litmus('RDW') == herd_logs['RDW']

    def test_PPOAA(self):
        assert run_litmus('PPOAA') == herd_logs['PPOAA']

    # @pytest.mark.skip(reason="need check")
    def test_ISA11_BIS(self):
        assert run_litmus('ISA11+BIS') == herd_logs['ISA11+BIS']

    def test_SB_rfi_addr_rfi_rfi_data_rfi(self):
        # 2023.12.21 17:53: 8s
        assert run_litmus('SB+rfi-addr-rfi+rfi-data-rfi') == herd_logs['SB+rfi-addr-rfi+rfi-data-rfi']

    def test_ISA_MP_DEP_SUCCESS_SWAP(self):
        assert run_litmus('ISA-MP-DEP-SUCCESS-SWAP') == herd_logs['ISA-MP-DEP-SUCCESS-SWAP']

    def test_RR_RR_rmw_fence_tso_rmw_fence_tsopx(self):
        assert run_litmus('RR+RR+rmw-fence.tso+rmw-fence.tsopx') == herd_logs['RR+RR+rmw-fence.tso+rmw-fence.tsopx']

    # @pytest.mark.skip(reason="wrong spinlock")
    def test_ISA03_SIMPLE(self):
        assert run_litmus('ISA03+SIMPLE') == herd_logs['ISA03+SIMPLE']

    def test_PPODA(self):
        assert run_litmus('PPODA') == herd_logs['PPODA']

    def test_ISA_MP_DEP_SUCCESS_SWAP_SIMPLE(self):
        assert run_litmus('ISA-MP-DEP-SUCCESS-SWAP-SIMPLE') == herd_logs['ISA-MP-DEP-SUCCESS-SWAP-SIMPLE']

    def test_ISA14_NEW(self):
        assert run_litmus('ISA14+NEW') == herd_logs['ISA14+NEW']

    def test_ISA03_SB01(self):
        assert run_litmus('ISA03+SB01') == herd_logs['ISA03+SB01']

    @pytest.mark.skip(reason="it takes too long")
    def test_SB_pos_po_ctrlfenceis(self):
        # 2023-12-20: 3m5s
        # 2024-01-04: 1m42s
        assert run_litmus('SB+pos-po-ctrlfenceis') == herd_logs['SB+pos-po-ctrlfenceis']

    @pytest.mark.skip(reason="it takes too long")
    def test_ISA03_SB02(self):
        # 2023-12-20: 2m53s
        # 2024-01-04: 1m28s
        assert run_litmus('ISA03+SB02') == herd_logs['ISA03+SB02']

    @pytest.mark.skip(reason="it takes too long")
    def test_ISA03(self):
        assert run_litmus('ISA03') == herd_logs['ISA03']

    def test_fence_tso(self):
        assert run_litmus('fence.tso') == herd_logs['fence.tso']

    def test_CoRR2_cleaninit(self):
        assert run_litmus('CoRR2-cleaninit') == herd_logs['CoRR2-cleaninit']

    def test_SB_rfi_data_rfi_rfi_ctrl_rfi(self):
        assert run_litmus('SB+rfi-data-rfi+rfi-ctrl-rfi') == herd_logs['SB+rfi-data-rfi+rfi-ctrl-rfi']