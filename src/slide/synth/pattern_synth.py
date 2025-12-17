# MIT License
#
# Copyright (c) 2024 DehengYang (dehengyang@qq.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
from os import replace
from src.slide import config
from src.slide.analysis.model.rvwmo import *
from src.slide.synth import transform
from src.slide.utils import cmd_util, file_util, dir_util
from src.slide.utils.str_util import *
from src.slide.validate.validate import validate, validate_use_herd, validate_use_chip_log, validate_for_chip
from src.slide.synth.gnode import MEM_TYPES
from src.slide.utils.ppo.ppo_parser import parse_to_gnode_tree
from src.slide.utils.herd_util import *
from src.slide.synth.ppo_dict import *
from src.slide.synth.ppo_def import *
from src.slide.synth.diy7_generator import *
import re

ppo_list = []
diy_litmus_cycle_dict = {}
cache_file = config.DIY7_CACHE_FILE_PATH

disallow_ppo_list = [
]

chip_pre_ppo_dict = {
    "[W];po;[W]": '2+2W',
    "[W];po;[R]": 'SB',
    "[R];po;[W]": 'LB'
}

def pre_test(chip_log):
    # filter some ppo
    for key in chip_pre_ppo_dict.keys():
        litmus = chip_pre_ppo_dict[key]
        if litmus in chip_log:
            if len(set(chip_log[litmus])) == 4:
                disallow_ppo_list.append(key)


def init_diy_cache():
    global diy_litmus_cycle_dict
    ppo_position_list = []
    with open(cache_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip() == 'new litmus test':
                ppo_position_list.append(i)
        # print(ppo_position_list)
        for index, i in enumerate(ppo_position_list):
            j = len(lines)
            if index != len(ppo_position_list) - 1:
                j = ppo_position_list[index + 1]
            ppo = lines[i + 1].strip()
            litmus_name = lines[i + 2].strip()
            litmus_content = ''.join(lines[i + 3:j])
            diy_litmus_cycle_dict[ppo] = (litmus_name, litmus_content)


def filter_not_suitable_ppo(ppo: SinglePPO):
    # if return False then don't consider this ppo
    filter_flag = {
        'fence_flag': True,
        'rmw_flag': True,
    }
    # if a ppo have fence and size>3, then filter
    size = ppo.size
    if size > 3:
        if any([isinstance(obj, FenceD) for obj in ppo.ppo]):
            filter_flag['fence_flag'] = False

    # if a ppo have lr, sc, then they must be [Lr],rmw,[Sc]
    for i, relation in enumerate(ppo.ppo):
        if isinstance(relation, Sc):
            if i >= 2 and (not isinstance(ppo.ppo[i - 1], Rmw) or not isinstance(ppo.ppo[i - 2], Lr)):
                filter_flag['rmw_flag'] = False
            if i + 1 < len(ppo.ppo) and (isinstance(ppo.ppo[i+1], CtrlD) or isinstance(ppo.ppo[i+1], CtrlS)): # remove [Sc];ctrl
                filter_flag['rmw_flag'] = False

    filter_list = list(filter_flag.values())


    if False in filter_list:
        return False
    return True


global_relation_list = ['co', 'int', 'ext', 'rsw', 'rfi', 'rfe', 'fri', 'fre', 'coi', 'coe', 'gmo', 'co', 'rf', 'fr']


# rvwmo_cat_file_path = f'{config.INPUT_DIR}/CAT/riscv.cat'

# ppo_pool =  get_rvwmo()


#
# def get_new_ppo_by_remove_validate_ppo(validate_ppo:SinglePPO):
#     ppos = ppo_pool.only_get_ppo_not_must_and_update_by_ppo(validate_ppo,True)
#     return [str(ppo) for ppo in ppos]


def get_candidate_mm_by_func_list(func_list):
    for _, _, python_func_string, _ in func_list:
        exec(python_func_string, globals())
    global_candiate_funcs_array = [globals()[f'{index}'] for index, _, _, _ in func_list]
    local_candiate_funcs_array = []
    for index, ppo_gnode_string, _, _ in func_list:
        if len(list(filter(lambda r: r in ppo_gnode_string, global_relation_list))) == 0:
            local_candiate_funcs_array.append(globals()[f'{index}'])

            # print('global func',global_candiate_funcs_array)
    # print('local func',local_candiate_funcs_array)

    # 3.2 prepare candidate_mm
    candidate_mm = RVWMO()
    candidate_mm.ppo_g = PPO(global_candiate_funcs_array)
    candidate_mm.ppo_l = PPO(local_candiate_funcs_array)
    return candidate_mm


def get_candidate_mm_by_ppo(ppo_pool, ppo_item=None):
    # 3.1 get func list
    ppo_list = []
    if ppo_item != None:
        ppo_list.append(ppo_item)
    func_list = ppo_pool.get_func(start_index=0, contain_init_func_flag=True, ppo_list=ppo_list)
    return get_candidate_mm_by_func_list(func_list), func_list


def synth_by_test_pattern_online(target_mm=None, ppo_pool=None, litmus_test_suite=None, target_cat_file_path=None,
                                 mode='herd', chip_log_path=None):
    # for chip or herd
    assert mode == 'herd' or mode == 'chip'
    if mode == 'herd':
        assert target_mm is not None and target_cat_file_path is not None, f'for herd synth must get target_mm and target_cat_file_path'
    else:
        assert chip_log_path is not None and os.path.exists(chip_log_path), 'for chip synth must get chip_log_path'
    chip_log = []
    if mode == 'chip':
        chip_log = {r.name: r.states for r in parse_chip_log(chip_log_path)}
        pre_test(chip_log)

    # 0. init(prepare dir,init data struct)
    test_dir = os.path.join(config.OUTPUT_DIR, 'new_tests')
    dir_util.mk_dir_from_dir_path(test_dir)
    file_util.rm_files_with_suffix_in_dir(test_dir, '.litmus')

    # 0.1 init counter
    ppo_index = 0  # global func index
    iterate_index = 0  # iterations
    max_iterate_count = 1  # max iterate count

    # 0.2 init func and dataset
    validate_test_suite = [litmus for litmus in litmus_test_suite]  # Test set for iteration

    # 0.3 init cache
    init_diy_cache()

    # 1. iterate
    while (True):
        iterate_index += 1
        if iterate_index > max_iterate_count:
            print('The maximum number of iterations exceeded')
            break
        print(f'now is the {iterate_index} iteration')
        # 1.1 synth ppo
        for i, litmus_test in enumerate(validate_test_suite):
            print(f'synth ppo by litmus test {i}')
            # 1.1.1 get candidate_ppos
            # valid_ppos = list(filter(lambda x:x[3],candidate_valid_ppos))
            cur_mm, func_list = get_candidate_mm_by_ppo(ppo_pool)
            # First, use herd to test whether it is necessary to generate ppo.
            candidate_mm_array = ppo_pool.get_cat_form(valid_flag_list=[PPOValidFlag.Valid],
                                                       init_flag_list=[PPOInitFlag.Init, PPOInitFlag.Verified],
                                                       ppo_list=[])
            candidate_mm_cat_str_array = [str(item) for item in candidate_mm_array]
            print('now herd cat')
            for item in candidate_mm_cat_str_array:
                print(item)
            candidate_mm_cat_file_path = create_cat_file(1, candidate_mm_cat_str_array)
            if mode == 'herd':
                _, validated = validate_use_herd(target_cat_file_path, candidate_mm_cat_file_path, [litmus_test])
            else:
                _, validated = validate_use_chip_log(chip_log, candidate_mm_cat_file_path, [litmus_test])
                pass

            if validated:
                continue
            # synth ppo
            candidate_ppos = synth_by_test_pattern(target_mm, ppo_pool, func_list, litmus_test_suite=[litmus_test],
                                                   ppo_index=ppo_index, target_cat_file_path=target_cat_file_path,
                                                   mode=mode, chip_log=chip_log)
            print(f'finished {litmus_test}')
        # 2 validate all litmus test
        print(f'iterate {iterate_index} final validate:')

        # 2.1 validate

        candidate_mm_array = ppo_pool.get_cat_form(valid_flag_list=[PPOValidFlag.Valid],
                                                   init_flag_list=[PPOInitFlag.Init, PPOInitFlag.Verified], ppo_list=[])
        candidate_mm_cat_str_array = [str(item) for item in candidate_mm_array]
        print('check cat array final start')
        for item in candidate_mm_cat_str_array:
            print(item)
        print('check cat arrat final end')
        candidate_mm_cat_file_path = create_cat_file(1, candidate_mm_cat_str_array)
        print(candidate_mm_cat_file_path)
        if mode == 'herd':
            failed_litmus_tests, validated = validate_use_herd(target_cat_file_path, candidate_mm_cat_file_path,
                                                               litmus_test_suite,
                                                               early_exit=False)
        else:
            failed_litmus_tests, validated = validate_use_chip_log(chip_log, candidate_mm_cat_file_path,
                                                                   litmus_test_suite)
        print('final validate', validated)
        print('failed_litmus_tests', failed_litmus_tests)
        if validated:
            break

    func_list = ppo_pool.get_func(start_index=ppo_index)
    print(func_list)
    print('ppo_list')
    for item in ppo_list:
        print(item)
    return list(filter(lambda x: x[2], func_list)), ppo_list


def synth_by_test_pattern(target_mm, ppo_pool, func_list, litmus_test_suite, ppo_index, target_cat_file_path,
                          mode='herd', chip_log=None):
    # 0.init
    print(f'synth ppo by litmus test {litmus_test_suite}')
    # 0.1 create test dir to save new litmus tests
    test_dir = os.path.join(config.OUTPUT_DIR, 'new_tests')
    dir_util.mk_dir_from_dir_path(test_dir)
    file_util.rm_files_with_suffix_in_dir(test_dir, '.litmus')

    # 0.2 validate
    cur_mm, func_list = get_candidate_mm_by_ppo(ppo_pool)

    if mode == 'herd':
        failed_litmus_tests, any_ppo_all, validated = validate(target_mm, cur_mm, func_list, litmus_test_suite,
                                                               verify_flag=False)
    else:
        failed_litmus_tests, any_ppo_all, validated = validate_for_chip(cur_mm, func_list, chip_log, litmus_test_suite,
                                                                        verify_flag=False)
    print(f'init failed: {len(failed_litmus_tests)}')

    # 0.3 init data struct
    tried_candidate_ppos = {}  # {candidate_ppo: (constraint, relax_flag)}
    litmus_test_dict = {}  # {candidate_ppo: litmus_test_paths}
    cnt = 0
    validated_ppo_mm_pairs = []
    all_new_test_paths = []
    ppo_index_tmp = ppo_index
    now_candiate_ppo = None  # (ppo_id,ppo_string,ppo_node)
    diy_args = []  # store create litmus test args
    pass_test_litmus_test_paths = []  # all litmus test which pass test

    print('start synth')
    # 1. synth ppo prepare

    for _, _, _, any_ppo, relax_flag in any_ppo_all:
        print(f"any_ppo size: {len(any_ppo)}")
        for constraint in any_ppo:
            print(f"candidate_ppos size: {len(constraint.candidate_ppos)}")
            for i, candidate_ppo in enumerate(constraint.candidate_ppos):
                # 1.1 prepare synth
                if candidate_ppo in tried_candidate_ppos:
                    continue
                if not filter_not_suitable_ppo(get_ppo_item_by_str(candidate_ppo)):
                    continue
                if candidate_ppo in disallow_ppo_list:
                    continue
                tried_candidate_ppos[candidate_ppo] = (constraint, relax_flag, constraint.diy_cycles[i])

    tried_candidate_ppos = dict(sorted(tried_candidate_ppos.items(), key=lambda item: len(item[0])))
    for candidate_ppo in tried_candidate_ppos:
        if candidate_ppo not in ppo_list:
            ppo_list.append(candidate_ppo)
        _, relax_flag, _ = tried_candidate_ppos[candidate_ppo]
        print(f'candidate_ppo: {candidate_ppo}, relax is {relax_flag}')

    # 2. synth
    for candidate_ppo in tried_candidate_ppos:
        constraint, relax_flag, diy_cycle_list = tried_candidate_ppos[candidate_ppo]
        print(f"ID: {cnt} cur candidate_ppo: {candidate_ppo},relax is {relax_flag}")
        cnt += 1
        # 2.1 prepare ppo function
        flag = PPOFlag.Strengthen if not relax_flag else PPOFlag.Relaxed
        print(candidate_ppo)
        ppo_item = get_ppo_item_by_str(candidate_ppo, flag=flag)

        # 2.2.1 check and get cat form ppo
        before_ppo_array = ppo_pool.get_cat_form(valid_flag_list=[PPOValidFlag.Valid],
                                                 init_flag_list=[PPOInitFlag.Init, PPOInitFlag.Verified], ppo_list=[])
        before_cat_str_array = [str(item) for item in before_ppo_array]
        ppo_pool.add_ppo(ppo_item)
        contain_flag, can_relax_flag = ppo_pool.check_contain_ppo(ppo_item)
        print('ppo', ppo_item, 'contain_flag', contain_flag, 'can_relax_flag', can_relax_flag)
        if contain_flag:  # if ppo be contained then skip
            continue
        if not can_relax_flag and relax_flag:  # if relax ppo not be contained then skip
            continue
        valid_flag = ppo_pool.get_ppo_valid_flag(ppo_item)
        print("add ppo", ppo_item, valid_flag)
        if valid_flag == PPOValidFlag.Invalid:
            continue
        after_ppo_array = ppo_pool.get_cat_form(valid_flag_list=[PPOValidFlag.Valid],
                                                init_flag_list=[PPOInitFlag.Init, PPOInitFlag.Verified],
                                                ppo_list=[ppo_item])
        after_cat_str_array = [str(item) for item in after_ppo_array]
        print('before_cat_str_array', before_cat_str_array)
        print('after cat_str_array', after_cat_str_array)

        # 2.2 create new litmus test
        if mode == "chip":
            diy_cycle_list = []
        new_test_paths = create_new_tests(test_dir, before_cat_str_array, after_cat_str_array, ppo_item, diy_cycle_list,
                                          cnt)
        if len(new_test_paths) == 0:
            print("[WARNING] fail to generate litmus test for this ppo.")  # FIX
            continue
        all_new_test_paths.extend(new_test_paths)
        litmus_test_dict[candidate_ppo] = new_test_paths

    # 3. run litmus test
    for candidate_ppo in tried_candidate_ppos:
        if candidate_ppo not in litmus_test_dict:
            continue
        new_test_paths = litmus_test_dict[candidate_ppo]
        constraint, relax_flag, diy_cycle_list = tried_candidate_ppos[candidate_ppo]
        print('check ppo', candidate_ppo)
        print('new_test_paths', new_test_paths)
        ppo_item = get_ppo_item_by_str(candidate_ppo, flag=relax_flag)

        # 3.3 validate
        candidate_mm_array = ppo_pool.get_cat_form(valid_flag_list=[PPOValidFlag.Valid],
                                                   init_flag_list=[PPOInitFlag.Init, PPOInitFlag.Verified],
                                                   ppo_list=[ppo_item])
        candidate_mm_cat_str_array = [str(item) for item in candidate_mm_array]
        candidate_mm_cat_file_path = create_cat_file(1, candidate_mm_cat_str_array)
        if mode == 'herd':
            cur_failed_litmus_tests, cur_validated = validate_use_herd(target_cat_file_path, candidate_mm_cat_file_path,
                                                                       new_test_paths,
                                                                       early_exit=True)
        else:
            cur_failed_litmus_tests, cur_validated = validate_use_chip_log(chip_log, candidate_mm_cat_file_path,
                                                                           new_test_paths)
        # 3.4 postprocess
        print('check ppo', candidate_ppo, cur_validated)
        if cur_validated:
            validated_ppo_mm_pairs.append((candidate_ppo, relax_flag))
            # diy_args.append((test_dir,ppo_use_for_create_litmus_test_before,ppo_use_for_create_litmus_test_after,use_string,cnt,ppo_index))
            pass_test_litmus_test_paths.extend(new_test_paths)
        else:
            ppo_pool.invalid_ppo(ppo_item)

    # 4. final test
    candidate_ppos = []
    print("===== run all new tests on the validated candidate ppos =====")
    print('check these ppo')
    for i, (candidate_ppo, relax_flag) in enumerate(validated_ppo_mm_pairs):
        print(f'ID{i}:candidate_ppo', candidate_ppo)
    print('start check')

    # 4.1 check counter ppo and create new litmus test to check
    print('check counter ppo')
    ppo_tmp_list = []
    for i, (candidate_ppo, relax_flag) in enumerate(validated_ppo_mm_pairs):
        flag = PPOFlag.Relaxed if relax_flag else PPOFlag.Strengthen
        ppo_item = get_ppo_item_by_str(candidate_ppo, flag)
        ppo_tmp_list.append(ppo_item)

    counter_litmus_list = []
    for ppo in ppo_tmp_list:
        counter_ppo_list = ppo.get_counter_ppo_list()
        print(f'ppo:{ppo} => counter_ppo:')
        for counter_ppo, litmus_ppo in counter_ppo_list:
            print(f'\t{counter_ppo} - {litmus_ppo}')
        print('-----------------')
        for counter_ppo, litmus_ppo in counter_ppo_list:
            if counter_ppo in ppo_tmp_list:
                add_ppo_array = ppo_pool.get_cat_form(valid_flag_list=[PPOValidFlag.Valid],
                                                      init_flag_list=[PPOInitFlag.Init, PPOInitFlag.Verified],
                                                      ppo_list=[ppo])
                add_cat_str_array = [str(item) for item in add_ppo_array]
                counter_ppo_array = ppo_pool.get_cat_form(valid_flag_list=[PPOValidFlag.Valid],
                                                          init_flag_list=[PPOInitFlag.Init, PPOInitFlag.Verified],
                                                          ppo_list=[litmus_ppo], is_virtual_flag=True)
                counter_ppo_cat_str_array = [str(item) for item in counter_ppo_array]
                cnt += 1
                counter_litmus_list.extend(
                    create_new_tests(test_dir, add_cat_str_array, counter_ppo_cat_str_array, litmus_ppo, [], cnt))
                print('get counter ppo litmus test', counter_ppo)

    for i, (candidate_ppo, relax_flag) in enumerate(validated_ppo_mm_pairs):
        print(f'ID{i}:candidate_ppo', candidate_ppo)
        flag = PPOFlag.Relaxed if relax_flag else PPOFlag.Strengthen
        ppo_item = get_ppo_item_by_str(candidate_ppo, flag)
        print('before add cat array')
        old_mm_array = ppo_pool.get_cat_form(valid_flag_list=[PPOValidFlag.Valid],
                                             init_flag_list=[PPOInitFlag.Init, PPOInitFlag.Verified], ppo_list=[])
        old_mm_cat_str_array = [str(item) for item in old_mm_array]
        for item in old_mm_cat_str_array:
            print(item)
        print('end add cat array')
        candidate_mm_array = ppo_pool.get_cat_form(valid_flag_list=[PPOValidFlag.Valid],
                                                   init_flag_list=[PPOInitFlag.Init, PPOInitFlag.Verified],
                                                   ppo_list=[ppo_item])
        candidate_mm_cat_str_array = [str(item) for item in candidate_mm_array]
        print('check cat array start')
        for item in candidate_mm_cat_str_array:
            print(item)
        print('check cat array end')
        candidate_mm_cat_file_path = create_cat_file(1, candidate_mm_cat_str_array)
        print('check litmus test suite')
        for litmus_path in litmus_test_suite + pass_test_litmus_test_paths:
            print(litmus_path)
        if mode == 'herd':
            cur_failed_litmus_tests, cur_validated = validate_use_herd(target_cat_file_path, candidate_mm_cat_file_path,
                                                                       litmus_test_suite + pass_test_litmus_test_paths + counter_litmus_list,
                                                                       early_exit=True)
        else:
            cur_failed_litmus_tests, cur_validated = validate_use_chip_log(chip_log, candidate_mm_cat_file_path,
                                                                           litmus_test_suite + pass_test_litmus_test_paths + counter_litmus_list)
        if cur_validated:
            candidate_ppos.append((candidate_ppo, ppo_item, relax_flag))
        print(
            f"candidate_ppo: {candidate_ppo}, cur_failed_litmus_tests: {cur_failed_litmus_tests}, passed? {cur_validated}\n")
    for _, ppo_item, _ in candidate_ppos:
        print(f'{str(ppo_item)} pass test')
        ppo_pool.verified_ppo(ppo_item)

    print("==== candidate ppo ====")
    for candidate_ppo, ppo_item, relax_flag in candidate_ppos:
        print(candidate_ppo, str(ppo_item), relax_flag)
    return candidate_ppos


def create_new_tests_by_generator(test_dir, before_cat_fragment, after_cat_fragment, ppo: SinglePPO, cnt, i):
    mutateGenerator = Diy7Generator()

    old_cat_file_path = create_cat_file(0, before_cat_fragment)
    new_cat_file_path = create_cat_file(1, after_cat_fragment)

    mutateGenerator.set_ppo(ppo, old_cat_file_path, new_cat_file_path)
    mutateGenerator.init_cycle_list()
    cycle, litmus_file_content, litmus_name = mutateGenerator.generate_litmus_test_legal()
    print('cycle, litmus_file_content', cycle, litmus_file_content, litmus_name)
    if cycle != None:
        litmus_suite = []
        diy_str = str(cycle)
        new_test_path = os.path.join(test_dir, f"{litmus_name}.litmus")
        new_test_name = f"{litmus_name}"
        with open(new_test_path, 'w') as f:
            f.write(litmus_file_content)
        litmus_suite.append(new_test_path)
        return litmus_suite, [(litmus_name, litmus_file_content)]
    return [], []
    # assert False, 'need to fix the generator'


def call_diy7_create_litmus_test(path, name, diy_cycle):
    cmd = f"{config.HERD_EVAL} diyone7 -arch RISC-V -obs local -name {path} {diy_cycle}"
    print(f"diyone7 cmd: {cmd}")
    cmd_util.run_cmd(cmd)
    if os.path.exists(path):
        with open(path, 'r') as f:
            content = f.readlines()
        content[0] = f'RISCV {name}'
        content = '\n'.join(content)
        with open(path, 'w') as wf:
            wf.write(content)
    return os.path.exists(path)


def create_new_tests(test_dir, before_cat_fragment, after_cat_fragment, ppo: SinglePPO, diy_cycle_list, cnt):
    '''
    cnt: The current number of ppo
    '''

    new_test_path_list = []

    # 0.0 get index
    regex = re.compile(pattern=rf'new_test_(\d+)_{cnt}.litmus')
    litmus_index = -1
    for file_name in os.listdir(test_dir):
        print(file_name)
        match = regex.match(file_name)
        if match:
            print('match,', match.group(1))
            litmus_index = max(litmus_index, int(match.group(1)))
    litmus_index += 1
    print('ppo', ppo, 'index', litmus_index)

    # 0. Check the diy cycle provided in advance
    for i, diy_cycle in enumerate(diy_cycle_list):
        litmus_name = str(ppo).replace('[', '').replace(']', '').replace(';', '_').replace('(', '').replace(')', '')
        litmus_name += '_'
        litmus_name += diy_cycle.replace(' ', '_').replace('.', '_')

        new_test_path = os.path.join(test_dir, f"{litmus_name}.litmus")
        new_test_name = f"{litmus_name}"
        if call_diy7_create_litmus_test(new_test_path, new_test_name, diy_cycle):
            new_test_path_list.append(new_test_path)
            litmus_index += 1

    # 1. check cache file
    print(ppo)
    if str(ppo) in diy_litmus_cycle_dict:
        new_test_name = f"{diy_litmus_cycle_dict[str(ppo)][0]}"
        new_test_path = os.path.join(test_dir, f"{new_test_name}.litmus")
        with open(new_test_path, 'w') as litmus_f:
            litmus_f.write(diy_litmus_cycle_dict[str(ppo)][1])
            new_test_path_list.append(new_test_path)
            litmus_index += 1

    # 2. herd was used to test whether the above ppo exhibited differential behavior
    old_cat_file_path = create_cat_file(0, before_cat_fragment)
    new_cat_file_path = create_cat_file(1, after_cat_fragment)
    print('test use herd')
    print('old_cat')
    for item in before_cat_fragment:
        print(item)
    print('new_cat')
    for item in after_cat_fragment:
        print(item)
    print('ppo', ppo)
    print('new_test_path_list', new_test_path_list)
    difference_flag = use_herd_test_ppo_by_litmus(new_test_path_list, old_cat_file_path, new_cat_file_path)
    if difference_flag and len(new_test_path_list) >= 1:
        return new_test_path_list

    litmus_suite, litmus_content_suite = create_new_tests_by_generator(test_dir, before_cat_fragment,
                                                                       after_cat_fragment, ppo, cnt, litmus_index)
    print('litmus suite', litmus_suite)
    # for litmus_name, litmus_content in litmus_content_suite:
    #     with open(cache_file, 'a') as cache_f:
    #         cache_f.write('new litmus test\n')
    #         cache_f.write(str(ppo))
    #         cache_f.write('\n')
    #         cache_f.write(litmus_name)
    #         cache_f.write('\n')
    #         cache_f.write(litmus_content)
    #         cache_f.write('\n')
    return litmus_suite # can be []


