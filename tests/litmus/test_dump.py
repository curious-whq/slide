
"""Test for Litmus"""

class TestLitmusDump:

    def test_dump_params(self):
        from src.slide.litmus.dump import dump_params
        print(dump_params(1, 2, 3, 4, 5, 6))

    def test_dump_template(self):
        from src.slide.litmus.dump import dump_template
        print(dump_template('0_license.c'))
        print(dump_template('1_header.c'))
        print(dump_template('2_topology.c'))

    def test_dump_ctx_t(self):
        from src.slide.litmus.dump import dump_ctx_t
        print(dump_ctx_t(['x', 'y'], [(0, 'x7'), (1, 'x7')]))

    def test_dump_outcome_collection(self):
        from src.slide.litmus.dump import dump_outcome_collection
        print(dump_outcome_collection(['x', 'y'], [(0, 'x7'), (1, 'x7')]))

    def test_dump_prefetch(self):
        from src.slide.litmus.dump import dump_prefetch
        print(dump_prefetch('2+2Swap+Acqs', ['x', 'y']))

    def test_dump_context(self):
        pass

    def test_dump_zyva(self):
        from src.slide.litmus.dump import dump_zyva
        print(dump_zyva('2+2Swap+Acqs', 2, ['x', 'y'], [(0, 'x10'), (0, 'x11'), (1, 'x10'), (1, 'x11')]))

    def test_dump_postlude(self):
        from src.slide.litmus.dump import dump_postlude
        print(
            dump_postlude('2+2Swap+Acqs', 'exists (x=2 /\\\ y=2 /\\\ 0:x10=1 /\\\ 0:x11=0 /\\\ 1:x10=1 /\\\ 1:x11=0)'))

    def test_dump_func_name(self):
        from src.slide.litmus.dump import dump_func_name
        assert dump_func_name(
            '2+2W+[rf-addr-fr]+fence.rw.w') == 'X2_2B_2W_2B__5B_rf_2D_addr_2D_fr_5D__2B_fence_2E_rw_2E_w'
        assert dump_func_name('2+2Swap+Acqs') == 'X2_2B_2Swap_2B_Acqs'
        assert dump_func_name(
            'MP+[ws-rf]-ctrlfencei+fence.rw.rw') == 'MP_2B__5B_ws_2D_rf_5D__2D_ctrlfencei_2B_fence_2E_rw_2E_rw'
