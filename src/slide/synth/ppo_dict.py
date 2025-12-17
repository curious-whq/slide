import enum

from typing import List
from enum import Enum
from src.slide.synth.memory_relation import *
from time import time

from src.slide.synth.ppo_def import *
from src.slide.synth.transform import transform
from src.slide.utils.ppo.ppo_parser import parse_to_gnode_tree



class PPOPool:
    def __init__(self):
        self.ppo_pool = [] # (SinglePPO, PPOValidFlag, PPOInitFlag)
        self.ppo_to_index_map = {} # ppo in index
        self.ppo_contain_list = [] #(ppo index, contained ppo index)
        self.ppo_contain_dict = {} #(ppo index, contained ppo index) = 1
        pass


    def add_ppo_in_pool(self, ppo: SinglePPO, ppo_valid_flag = PPOValidFlag.Valid, ppo_init_flag = PPOInitFlag.Added):
        self.ppo_pool.append((ppo, ppo_valid_flag, ppo_init_flag))
        self.ppo_to_index_map[ppo] = len(self.ppo_pool) - 1

    def get_ppo_valid_flag(self, ppo: SinglePPO) -> PPOValidFlag:
        return self.ppo_pool[self.ppo_to_index_map[ppo]][1]

    def get_ppo_init_flag(self, ppo: SinglePPO) -> PPOInitFlag:
        return self.ppo_pool[self.ppo_to_index_map[ppo]][2]

    def get_ppo_flag(self, ppo: SinglePPO) ->PPOFlag:
        return self.ppo_pool[self.ppo_to_index_map[ppo]][0].flag

    def get_all_be_contain_list(self, ppo: SinglePPO, valid_flag_list: List[PPOValidFlag] , init_flag_list: List[PPOInitFlag]):
        index_list = list(map(lambda pair : pair[0], filter(lambda pair : pair[1] == self.ppo_to_index_map[ppo], self.ppo_contain_list)))
        if valid_flag_list != None:
            index_list = [index for index in index_list if self.ppo_pool[index][1] in valid_flag_list]
        if init_flag_list != None:
            index_list = [index for index in index_list if self.ppo_pool[index][2] in init_flag_list]
        return index_list

    def get_all_contain_list(self, ppo: SinglePPO, valid_flag_list: List[PPOValidFlag] , init_flag_list: List[PPOInitFlag]):
        index_list = list(map(lambda pair : pair[1], filter(lambda pair : pair[0] == self.ppo_to_index_map[ppo], self.ppo_contain_list)))
        if valid_flag_list != None:
            index_list = [index for index in index_list if self.ppo_pool[index][1] in valid_flag_list]
        if init_flag_list != None:
            index_list = [index for index in index_list if self.ppo_pool[index][2] in init_flag_list]
        return index_list

    def get_strict_be_contain_list(self, ppo: SinglePPO, valid_flag_list: List[PPOValidFlag] , init_flag_list: List[PPOInitFlag]):
        strict_be_contain_list = []
        be_contain_list = self.get_all_be_contain_list(ppo, valid_flag_list, init_flag_list)
        print(be_contain_list)
        # strict indicates that x will not contain y, and y will contain ppo
        for be_contain_index in be_contain_list:
            be_contain_ppo,_,_ = self.ppo_pool[be_contain_index]
            contain_list = self.get_all_contain_list(be_contain_ppo, valid_flag_list, init_flag_list)
            print(be_contain_ppo)
            print(contain_list, be_contain_index)
            if len(set(be_contain_list).intersection(set(contain_list))) ==0 :
                strict_be_contain_list.append(be_contain_index)

        return strict_be_contain_list

    def get_strict_contain_list(self, ppo: SinglePPO, valid_flag_list: List[PPOValidFlag] , init_flag_list: List[PPOInitFlag]):
        strict_contain_list = []
        contain_list = self.get_all_contain_list(ppo, valid_flag_list, init_flag_list)
        for contain_index in contain_list:
            contain_ppo,_,_ = self.ppo_pool[contain_index]
            be_contain_list = self.get_all_be_contain_list(contain_ppo, valid_flag_list, init_flag_list)
            if len(set(be_contain_list).intersection(set(contain_list))) == 0 :
                strict_contain_list.append(contain_index)
        return strict_contain_list


    def add_ppo(self, ppo: SinglePPO, ppo_valid_flag = PPOValidFlag.Valid, ppo_init_flag = PPOInitFlag.Added):
        # check Is it worth checking its effectiveness, by query ppo is Valid decide to verify this ppo.

        # 1).find out if the ppo already exists
        if ppo in self.ppo_to_index_map:
            #TODO :  Consider a more robust case
            single_ppo, single_ppo_valid_flag, single_ppo_init_flag = self.ppo_pool[self.ppo_to_index_map[ppo]]

            if single_ppo.flag != ppo.flag :
                if single_ppo_init_flag == PPOInitFlag.Init or single_ppo_init_flag == PPOInitFlag.Verified:
                    return

                if single_ppo_valid_flag == PPOValidFlag.Valid: # If the ppo is Valid, its effect is hidden and set to removed
                    self.ppo_pool[self.ppo_to_index_map[ppo]]=(single_ppo, PPOValidFlag.Invalid, PPOInitFlag.Removed)

                if single_ppo_valid_flag == PPOValidFlag.Invalid: # If this ppo is Invalid, it is equivalent to adding a new ppo
                    self.ppo_pool[self.ppo_to_index_map[ppo]]=(ppo, ppo_valid_flag, PPOInitFlag.Added)
            return
        # 2).add this ppo to ppo_pool
        self.add_ppo_in_pool(ppo, ppo_valid_flag, ppo_init_flag)
        ppo_index = self.ppo_to_index_map[ppo]

        # 3).Add a contain relationship for the newly added ppo
        # process contain relation TODO:preprocess
        for single_ppo, _, _ in self.ppo_pool:
            single_ppo_index = self.ppo_to_index_map[single_ppo]
            if single_ppo_index == ppo_index:
                continue
            if single_ppo.is_contain(ppo):
                self.ppo_contain_list.append((single_ppo_index, ppo_index))
                self.ppo_contain_dict[(single_ppo_index, ppo_index)]=1
            if ppo.is_contain(single_ppo):
                self.ppo_contain_list.append((ppo_index, single_ppo_index))
                self.ppo_contain_dict[(ppo_index, single_ppo_index)]=1


    def check_contain_ppo(self, ppo: SinglePPO):
        # return (contain_flag, relax_flag) contain_flag show be contain by flag equal ppo, relax_flag show be contain by flag unequal ppo
        ppo_index = self.ppo_to_index_map[ppo]
        single_ppo, single_valid_flag ,single_init_flag = self.ppo_pool[ppo_index]
        if single_init_flag == PPOInitFlag.Init:
            return True, False
        contain_list = self.get_strict_be_contain_list(ppo, valid_flag_list=[PPOValidFlag.Valid], init_flag_list=[PPOInitFlag.Init, PPOInitFlag.Verified])
        if contain_list != []:
            print('these ppo strict contain ppo ',ppo, 'contain_list is ', [self.ppo_pool[index][0] for index in contain_list])
        can_relax_flag = False
        for index in contain_list:
            contain_ppo, contain_ppo_valid_flag, contain_ppo_init_flag = self.ppo_pool[index]
            # print(contain_ppo)
            # if contain_ppo_valid_flag == PPOValidFlag.Invalid:
            #     continue
            # if contain_ppo_init_flag == PPOInitFlag.Added:
            #     continue
            print(contain_ppo.flag, ppo.flag)
            if contain_ppo.flag != ppo.flag and contain_ppo_init_flag != PPOInitFlag.Init:
                can_relax_flag = True
                # if (ppo_index, index) not in self.ppo_contain_dict: 
                continue
            self.ppo_pool[ppo_index] = (single_ppo, PPOValidFlag.Invalid, single_init_flag)
            return True, can_relax_flag
        return False, can_relax_flag

    def check_contain_ppo_for_post(self, ppo: SinglePPO):
        # return (contain_flag, relax_flag) contain_flag show be contain by flag equal ppo, relax_flag show be contain by flag unequal ppo
        ppo_index = self.ppo_to_index_map[ppo]
        single_ppo, single_valid_flag ,single_init_flag = self.ppo_pool[ppo_index]
        if single_init_flag in [PPOInitFlag.Init, PPOInitFlag.Verified]:
            return True, False
        contain_list = self.get_strict_be_contain_list(ppo, valid_flag_list=[PPOValidFlag.Valid], init_flag_list=[PPOInitFlag.Init, PPOInitFlag.Verified])
        if contain_list != []:
            print('these ppo strict contain ppo ',ppo, 'contain_list is ', [self.ppo_pool[index][0] for index in contain_list])
        can_relax_flag = False
        for index in contain_list:
            contain_ppo, contain_ppo_valid_flag, contain_ppo_init_flag = self.ppo_pool[index]
            # print(contain_ppo)
            # if contain_ppo_valid_flag == PPOValidFlag.Invalid:
            #     continue
            # if contain_ppo_init_flag == PPOInitFlag.Added:
            #     continue
            print(contain_ppo.flag, ppo.flag)
            if contain_ppo.flag != ppo.flag:
                can_relax_flag = True
                # if (ppo_index, index) not in self.ppo_contain_dict:
                continue
            self.ppo_pool[ppo_index] = (single_ppo, PPOValidFlag.Invalid, single_init_flag)
            return True, can_relax_flag
        return False, can_relax_flag

    def verified_ppo(self, ppo: SinglePPO): # pass validate, change state to verified
        index = self.ppo_to_index_map[ppo]
        single_ppo, valid_flag, init_flag = self.ppo_pool[index]

        if init_flag == PPOInitFlag.Init: # init ppo cannot be changed
            return
        if init_flag == PPOInitFlag.Removed: # for strong to relax
            single_ppo.rev_flag()
            self.ppo_pool[index] = (single_ppo, PPOValidFlag.Invalid, PPOInitFlag.Verified)
            return

        self.ppo_pool[index] = (single_ppo, PPOValidFlag.Valid, PPOInitFlag.Verified)

    def invalid_ppo(self, ppo: SinglePPO): # validate false then invoke invalid_ppo
        index = self.ppo_to_index_map[ppo]
        single_ppo, valid_flag, init_flag = self.ppo_pool[index]

        if init_flag == PPOInitFlag.Init: # init ppo cannot be changed
            return
        if init_flag == PPOInitFlag.Removed: # for strong to relax
            single_ppo.rev_flag()
            self.ppo_pool[index] = (single_ppo, PPOValidFlag.Valid, PPOInitFlag.Verified)
            return

        self.ppo_pool[index] = (single_ppo, PPOValidFlag.Invalid, PPOInitFlag.Verified)

    def get_cat_form(self, valid_flag_list: List[PPOValidFlag] , init_flag_list: List[PPOInitFlag], ppo_list:List[SinglePPO], is_virtual_flag = False):
        # use mix_ppo
        # for ppo, valid_flag, init_flag in self.ppo_pool:
        #     print(ppo, valid_flag, init_flag)
        cat_form_list = []
        filter_ppo_list = self.ppo_pool
        filter_ppo_list = list(filter(lambda ppo : ppo[1] in valid_flag_list, filter_ppo_list))
        filter_ppo_list = list(filter(lambda ppo : ppo[2] in init_flag_list, filter_ppo_list))
        for ppo in ppo_list:
            if is_virtual_flag : 
                filter_ppo_list.append((ppo, PPOValidFlag.Valid, PPOInitFlag.Added))
            else:
                filter_ppo_list.append(self.ppo_pool[self.ppo_to_index_map[ppo]])

        strengthen_ppo_list = list(filter(lambda x:x[0].flag==PPOFlag.Strengthen, filter_ppo_list))
        relaxed_ppo_list = list(filter(lambda x:x[0].flag==PPOFlag.Relaxed, filter_ppo_list))

        for strengthen_ppo_item,_,init_flag in strengthen_ppo_list:
            relaxed_list = []
            ppo_index = self.ppo_to_index_map.get(strengthen_ppo_item, -1)
            for relaxed_ppo,_,_ in relaxed_ppo_list:
                
                relaxed_ppo_index = self.ppo_to_index_map.get(relaxed_ppo, -1)
                if ppo_index == -1 or relaxed_ppo_index == -1:
                    if strengthen_ppo_item.is_contain(relaxed_ppo):
                        _, _, relax_ppo_init_flag = self.ppo_pool[relaxed_ppo_index]
                        if relax_ppo_init_flag == PPOInitFlag.Init and init_flag != PPOInitFlag.Init:
                            continue
                        relaxed_list.append(relaxed_ppo)
                elif (ppo_index, relaxed_ppo_index) in self.ppo_contain_dict:
                    relaxed_list.append(relaxed_ppo)
            cat_form_list.append(MixPPO(strengthen_ppo_item, relaxed_list, init_flag))

        return cat_form_list
    

    def get_func(self, start_index, contain_init_func_flag = False, ppo_list = []): #[(index,ppo_gnode,func_string,init_flag)]
        func_list = []
        init_flag_list = [PPOInitFlag.Verified]
        if contain_init_func_flag :
            init_flag_list.append(PPOInitFlag.Init)
        cat_form_list = self.get_cat_form(valid_flag_list=[PPOValidFlag.Valid], init_flag_list=init_flag_list,ppo_list=ppo_list)
        
        for mix_ppo in cat_form_list:
            if mix_ppo.init_flag == PPOInitFlag.Init and not contain_init_func_flag:
                continue
            ppo_gnode_string = mix_ppo.get_gnode_form()
            ppo_gnode = parse_to_gnode_tree(ppo_gnode_string)
            python_func_string = transform(ppo_gnode, ppo_index= start_index)
            # print(python_func_string)
            func_list.append((f'ppo_candidate_func{start_index}', ppo_gnode_string, python_func_string, mix_ppo.init_flag))
            start_index += 1

        return func_list
    def unroll_by_length(self, length = 2):
        # For covering all possible PPOs of a specified length. only for strengthen ppo
        # Fix: consider more .
        unroll_ppo_list = []
        for ppo_i, _, init_flag_i in self.ppo_pool:
            for ppo_j, _, init_flag_j in self.ppo_pool:
                if not (init_flag_i in [PPOInitFlag.Init, PPOInitFlag.Verified] and init_flag_j in [PPOInitFlag.Init, PPOInitFlag.Verified]):
                    continue
                if ppo_i.flag != ppo_j.flag:
                    continue
                if type(ppo_i.ppo[-1]) == AMO and type(ppo_j.ppo[0]) == W:
                    pass
                elif type(ppo_i.ppo[-1]) == W and type(ppo_j.ppo[0]) == AMO:
                    new_ppo = SinglePPO(ppo_i.ppo[:-1] + ppo_j.ppo[:], ppo_i.flag)
                    unroll_ppo_list.append(new_ppo)
                    print(new_ppo)
                    continue
                elif type(ppo_i.ppo[-1]) != type(ppo_j.ppo[0]) :
                    continue
                new_ppo = SinglePPO(ppo_i.ppo+ppo_j.ppo[1:],ppo_i.flag)
                unroll_ppo_list.append(new_ppo)
        for ppo in unroll_ppo_list:
            self.add_ppo(ppo, PPOValidFlag.Valid, PPOInitFlag.Init)






def create_cat_file_fragment(temp_file,ppo_list):
    var_str = '\n'
    for i, ppo in enumerate(ppo_list):
        var_str += 'let 'if i==0 else 'and '
        var_str += f'r{i+1} = {ppo} \n'

    ppo_str = '\nlet ppo = \n'
    for i, ppo in enumerate(ppo_list):
        ppo_str += f' ' if i==0 else '| '
        ppo_str += f'r{i+1}\n'

    print(var_str)
    print(ppo_str)

    with open(temp_file, 'a') as f:
        f.write(var_str)
        f.write(ppo_str)

def helper_add_function(mm, ppo):
    mm.add_ppo(ppo)
    flag = mm.check_contain_ppo(ppo)
    state = mm.get_ppo_valid_flag(ppo)
    return flag, state

