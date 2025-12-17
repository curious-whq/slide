
"""Test for Litmus"""

from src.slide import config
from src.slide.analysis import *
from src.slide.litmus import parse_litmus
from src.slide.utils.file_util import *

all_litmus_files = list_files('input/litmus', '.litmus')
manual_inputs = ['MP+fence.rw.rw+ctrl_multi_labels', 'C-Will01-Bad', '2+2W+fence.rw.rws+fence.rw.rwspx',
                 'C-Will02+HEAD-remove-locs']
classic_inputs = ['SB', 'MP+fence.rw.rw+ctrl']
output_dir = f'{os.getcwd()}/output/'


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


class TestPPO:
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

    def test_SB(self):
        sb = parse_litmus_by_name('SB')
        for i in range(sb.n_threads):
            prog = sb.progs[i]
            print(f'P{i}:\n{prog}')
            cfg = CFG(prog.insts)
            cfg.plot(f'{os.getcwd()}/output/cfg_SB_prog_{i}')
            paths = cfg.find_all_paths()
            for j, path in enumerate(paths):
                cfg.plot_path(path, f'{os.getcwd()}/output/cfg_SB_prog_{i}_path_{j}')

                ssas = path_to_ssa(path)
                CFG(ssas).plot(f'{os.getcwd()}/output/cfg_SB_prog_{i}_path_{j}_ssa')
                sim = Simulator(verbose=1)
                ctx = sim.run(ssas)
                config.set_var('ctx', ctx)
                events = ctx.events
                print(events)

    def test_SB_Fence_ppo(self):
        litmus_name = 'SB+fence'
        litmus = parse_litmus_by_name(litmus_name)
        for pp in litmus.gen_parallel_paths():
            for e in pp.gen_executions():
                print(e)
            # print('\n'.join(pp.gen_executions()))

    def test_2(self):
        print(f"hello2")
