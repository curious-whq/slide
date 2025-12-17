import itertools
import os
import sys
import time
from copy import deepcopy


from src.slide import config
from src.slide.analysis import RVWMO
from src.slide.comp.parse_result import parse_herd_log
from src.slide.litmus import parse_litmus
from src.slide.litmus.load_litmus import get_litmus_by_policy, GetLitmusPolicy, sort_litmus_by_weight
from src.slide.synth.diy7_generator import Diy7Generator
from src.slide.synth.model.ppo_pool_C910 import get_C910_model
from src.slide.synth.model.ppo_pool_rvwmo import get_rvwmo_model
from src.slide.synth.model.ppo_pool_rvwmo_init import get_rvwmo_init_model
from src.slide.synth.model.ppo_pool_rvwmo_synth import get_rvwmo_model_synth
from src.slide.synth.pattern_synth import filter_not_suitable_ppo, synth_by_test_pattern_online
from src.slide.synth.ppo_def import SinglePPO, get_ppo_item_by_str, PPOInitFlag, PPOValidFlag, PPOFlag
# from src.slide.synth.ppo_dict import get_rvwmo

from src.slide.synth.memory_relation import *
from src.slide.utils import cmd_util
from src.slide.utils.file_util import read_file, search_file
from src.slide.utils.herd_util import create_cat_file, herd_run_all_valid_litmus

exp_4_output_result_file_path = os.path.join(config.TEST_DIR,'results/exp4_cycle_rvwmo_result.txt')
exp_4_output_log_path = os.path.join(config.TEST_DIR,'results/exp4_synth_rvwmo.log')


exp_4_synth_output_log_path = os.path.join(config.TEST_DIR,'results/exp4_synth_rvwmo_post.log')
exp_4_synth_output_result_file_path = os.path.join(config.TEST_DIR,'results/exp4_synth_rvwmo_post_result.txt')
exp_4_synth_test_ppo_path = os.path.join(config.TEST_DIR,'results/exp4_synth_rvwmo_post_tested_ppo.txt')


exp4_synth_litmus_dir = os.path.join(config.OUTPUT_DIR, 'complete_litmus_rvwmo')

def delete_history():
    print(f'deleting exp4 rvwmo litmus suite: {exp4_synth_litmus_dir}')
    for filename in os.listdir(exp4_synth_litmus_dir):
        file_path = os.path.join(exp4_synth_litmus_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    if os.path.exists(exp_4_output_result_file_path):
        print(f'deleting exp_4_output_result_file_path: {exp_4_output_result_file_path}')
        os.remove(exp_4_output_result_file_path)
    if os.path.exists(exp_4_output_log_path):
        print(f'deleting exp_4_output_log_path: {exp_4_output_log_path}')
        os.remove(exp_4_output_log_path)
    if os.path.exists(exp_4_synth_output_log_path):
        print(f'deleting exp_4_synth_output_log_path: {exp_4_synth_output_log_path}')
        os.remove(exp_4_synth_output_log_path)
    if os.path.exists(exp_4_synth_output_result_file_path):
        print(f'deleting exp_4_synth_output_result_file_path: {exp_4_synth_output_result_file_path}')
        os.remove(exp_4_synth_output_result_file_path)
    if os.path.exists(exp_4_synth_test_ppo_path):
        print(f'deleting exp_4_synth_test_ppo_path: {exp_4_synth_test_ppo_path}')
        os.remove(exp_4_synth_test_ppo_path)

class CycleTest():

    unary_relations = [R(), W(), Sc(), Lr(), AMO(), AMO(annotation=Annotation.RL),
                       AMO(annotation=Annotation.AQ), AMO(annotation=Annotation.AQRL)]
    binary_relations = [Po(),AddrD(),DataD(),CtrlD(),Rsw(),
                        FenceD(annotation=FenceAnnotation.RW_RW),FenceD(annotation=FenceAnnotation.W_RW),
                        FenceD(annotation=FenceAnnotation.R_RW), FenceD(annotation=FenceAnnotation.RW_R),
                        FenceD(annotation=FenceAnnotation.RW_W), FenceD(annotation=FenceAnnotation.W_W),
                        FenceD(annotation=FenceAnnotation.R_R), FenceD(annotation=FenceAnnotation.R_W),
                        FenceD(annotation=FenceAnnotation.W_R), FenceD(annotation=FenceAnnotation.TSO),
                        Coi(),Fri(),Rfi(),PoLoc(),Rmw()
                        ]

    def get_ppo_by_length(self, length = 5):
        assert length % 2 == 1 and length > 1, 'the ppo must is 5'
        queue = []
        queue.extend([[deepcopy(relation)] for relation in self.unary_relations ])
        while len(queue)!=0:
            item = queue.pop(0)
            if len(item)==length:
                yield item
                continue
            if len(item) %2 ==0:
                queue.extend([deepcopy(item)+[deepcopy(relation)] for relation in self.unary_relations ])
            else:
                queue.extend([deepcopy(item)+[deepcopy(relation)] for relation in self.binary_relations ])


    def synth_all_ppo(self, length = 3, unroll_length = 2):
        # ppo_list = [[W(),Po(),Lr()]]
        ppo_list = []
        sum = 0
        # for ppo in ppo_list:
        for ppo in self.get_ppo_by_length(length = length):
            rvwmo = get_rvwmo_model_synth()
            sum += 1
            # check sc
            sc_ppo = [i for i,rel in enumerate(ppo) if type(rel) == Sc]
            flag = True
            for index in sc_ppo:
                if index == 0 or type(ppo[index-2]) != Lr:
                    flag = False
                    break

            # check fence and X all exist
            if len([rel for rel in ppo if type(rel) == FenceD])!=0 and len([rel for rel in ppo if type(rel) in [Lr,Sc,AMO]])!=0:
                flag = False
            if not flag:
                continue
            ppo = SinglePPO(ppo)
            print(sum)
            if not ppo.check_ppo():
                continue
            if not filter_not_suitable_ppo(ppo):
                print(f'not suitable ppo for {ppo}')
                continue
            print(f'pass check_ppo {ppo}')
            rvwmo.unroll_by_length(length = unroll_length)
            rvwmo.add_ppo(ppo)
            flag, _ = rvwmo.check_contain_ppo_for_post(ppo)
            if not flag:
                ppo_list.append(ppo)
                print(f'add this ppo {ppo}')
        print('ppo_list')
        print(f'{sum} / {len(ppo_list)}')
        for ppo in ppo_list:
            print(ppo)
        return ppo_list



    def get_not_contain_ppo_in_dataset(self, ppo_list, contain_ppo_file):
        contain_ppo_list = []
        not_be_contain_ppo_list = []
        # with open(ppo_file, 'r') as f:
        #     ppo_list = f.readlines()
        with open(contain_ppo_file, 'r') as f:
            contain_ppo_list = f.readlines()
        # ppo_list = [get_ppo_item_by_str(item.strip()) for item in ppo_list]
        contain_ppo_list = [get_ppo_item_by_str(item.strip()) for item in contain_ppo_list]
        for ppo in ppo_list:
            if ppo in contain_ppo_list:
                continue
            not_be_contain_ppo_list.append(ppo)
        for ppo in not_be_contain_ppo_list:
            print(ppo)
        return not_be_contain_ppo_list


    def synth_litmus(self, ppo_list):
        generator = Diy7Generator(time_limit=50)
        synth_ppo_list = []
        not_synth_ppo_list = []
        for ppo in ppo_list:
            print(f'synth ppo {str(ppo)}')
            model = get_rvwmo_model_synth()
            before_ppo_array = model.get_cat_form(valid_flag_list=[PPOValidFlag.Valid],
                                                     init_flag_list=[PPOInitFlag.Init, PPOInitFlag.Verified],
                                                     ppo_list=[])
            before_cat_str_array = [str(item) for item in before_ppo_array]
            model.add_ppo(ppo)
            after_ppo_array = model.get_cat_form(valid_flag_list=[PPOValidFlag.Valid],
                                                    init_flag_list=[PPOInitFlag.Init, PPOInitFlag.Verified],
                                                    ppo_list=[ppo])
            after_cat_str_array = [str(item) for item in after_ppo_array]
            old_cat_file_path = create_cat_file(0, before_cat_str_array)
            new_cat_file_path = create_cat_file(1, after_cat_str_array)
            generator.set_ppo(ppo, old_cat_file_path, new_cat_file_path)
            generator.init_cycle_list()
            cycle, litmus_file_content, litmus_name = generator.generate_litmus_test_legal()

            if cycle:
                synth_ppo_list.append(ppo)
                litmus_path = os.path.join(exp4_synth_litmus_dir, f'{litmus_name}.litmus')
                with open(litmus_path, 'w') as wf:
                    for line in litmus_file_content:
                        wf.write(line)
            else:
                not_synth_ppo_list.append(ppo)

            with open(exp_4_output_result_file_path, 'w') as wf:
                wf.write('----------------------------------------------------\n')
                wf.write('these ppo find litmus test')
                for ppo in synth_ppo_list:
                    wf.write(f'{str(ppo)}\n')
                wf.write('----------------------------------------------------\n')
                wf.write('----------------------------------------------------\n')
                wf.write('these ppo not find litmus test')
                for ppo in not_synth_ppo_list:
                    wf.write(f'{str(ppo)}\n')
                wf.write('----------------------------------------------------\n')


    def ppo_length_3(self):
        return test.synth_all_ppo(3, unroll_length=1)
        # with open(output_file, 'w') as wf:
        #     for ppo in ppo_list:
        #         wf.write(str(ppo))
        #         wf.write('\n')

    def ppo_length_5(self):
        return test.synth_all_ppo(5, unroll_length=2)
        # with open(output_file, 'w') as wf:
        #     for ppo in ppo_list:
        #         wf.write(str(ppo))
        #         wf.write('\n')



    def run_litmus(self):
        all_paths = []
        for root, dirs, files in os.walk(exp4_synth_litmus_dir):
            for file in files:
                all_paths.append(os.path.join(root, file))
        memory_model_dir = os.path.join(config.INPUT_DIR, 'CAT')
        rvwmo_cat_file_path = os.path.join(memory_model_dir, 'riscv-complete.cat')
        rvwmo_log = os.path.join(f'{config.INPUT_DIR}', 'herd/herd_results_exp4_rvwmo.log')
        model = get_rvwmo_model_synth()
        ppo_array = model.get_cat_form(valid_flag_list=[PPOValidFlag.Valid],
                                              init_flag_list=[PPOInitFlag.Init, PPOInitFlag.Verified],
                                              ppo_list=[])
        cat_str_array = [str(item) for item in ppo_array]
        synth_cat_file_path = create_cat_file(0, cat_str_array)
        synth_log = os.path.join(f'{config.INPUT_DIR}', 'herd/herd_results_exp4_rvwmo_synth.log')
        for path in all_paths:
            print(path)
        herd_run_all_valid_litmus(rvwmo_cat_file_path, all_paths, rvwmo_log)
        herd_run_all_valid_litmus(synth_cat_file_path, all_paths, synth_log)

    def compare_litmus(self):
        rvwmo_log = os.path.join(f'{config.INPUT_DIR}', 'herd/herd_results_exp4_rvwmo.log')
        synth_log = os.path.join(f'{config.INPUT_DIR}', 'herd/herd_results_exp4_rvwmo_synth.log')
        rvwmo_dict = {r.name: r.states for r in parse_herd_log(rvwmo_log)}
        synth_dict = {r.name: r.states for r in parse_herd_log(synth_log)}
        dif_list = []
        for key in rvwmo_dict.keys():
            if set(rvwmo_dict[key]) != set(synth_dict[key]):
                dif_list.append(key)
        return dif_list

    def post_synth(self):
        start = time.time()
        litmus_test_suite = self.compare_litmus()
        litmus_test_suite = sort_litmus_by_weight(litmus_test_suite, litmus_dir = [exp4_synth_litmus_dir])
        print(len(litmus_test_suite))
        for litmus in litmus_test_suite:
            print(litmus)
        litmus_test_suite = [search_file(litmus, exp4_synth_litmus_dir, '.litmus') for litmus in
                             litmus_test_suite]
        # return
        # synth
        with open(exp_4_synth_output_log_path, 'w') as f:
            ori = sys.stdout
            sys.stdout = f
            config.init()
            config.set_var('reg_size', 64)

            rvwmo = RVWMO()
            cur_mm = get_rvwmo_model_synth()
            rvwmo_cat = os.path.join(config.CAT_DIR, 'riscv-complete.cat')
            patch, ppo_list = synth_by_test_pattern_online(rvwmo, cur_mm, litmus_test_suite, rvwmo_cat)
            time_cost = time.time() - start
            sys.stdout = ori
            f.close()

        with open(exp_4_synth_output_result_file_path, 'a+') as f:
            ori = sys.stdout
            sys.stdout = f
            if patch is not None:
                for _, ppo, _, _ in patch:
                    print(ppo)
            print('time: ', f'{time_cost}s')
            sys.stdout = ori
            f.close()

        with open(exp_4_synth_test_ppo_path, 'w') as f:
            ori = sys.stdout
            sys.stdout = f
            for ppo in ppo_list:
                print(ppo)
            sys.stdout = ori
            f.close()


if __name__ == '__main__':
    bar = "=" * 60
    header = f"Experiment 4: synth RVWMO post"
    centered_header = header.center(60)
    print(f"\n{bar}\n{centered_header}\n{bar}\n")
    test = CycleTest()
    delete_history()
    with open(exp_4_output_log_path, 'w') as f:
        od = sys.stdout
        sys.stdout = f
        start_ppo_time = time.time()
        ppo_list = test.ppo_length_3()
        ppo_list.extend(test.ppo_length_5())
        end_ppo_time = time.time()
        with open(exp_4_synth_output_result_file_path,'w') as af:
            pass
        with open(exp_4_synth_output_result_file_path,'a+') as af:
            af.write('search ppo time: ')
            af.write(str(end_ppo_time - start_ppo_time))
            af.write('s')
            af.write('\n')

        print('to synth these ppo')
        start_synth_time = time.time()
        test.synth_litmus(ppo_list)
        end_synth_time = time.time()
        with open(exp_4_synth_output_result_file_path, 'a+') as af:
            af.write('synth litmus test time: ')
            af.write(str(end_synth_time - start_synth_time))
            af.write('s')
            af.write('\n')

        test.run_litmus()
        test.post_synth()
        sys.stdout = od




