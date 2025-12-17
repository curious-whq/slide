
import sys
sys.path.append('../src')
from src.slide.synth import transform
from src.slide.ppo.ppo_parser import parse_to_gnode_tree
from src.slide.synth.rel_expr import *
from src.slide.utils.str_util import *
from src.slide.synth.gnode import MEM_TYPES
#pytest -s synth/test_transform.py::TestTransform::test_transform_ppo1to13
import os

class TestTransform:

            


    def test_func(self):
        ppo_gnode_string = 'AMO;po;AMO'
        ppo_gnode = parse_to_gnode_tree(ppo_gnode_string)
        python_func_string = transform.transform(ppo_gnode, ppo_index = 0)
        print(python_func_string)

        ppo_gnode_string = 'R;po;AMO'
        ppo_gnode = parse_to_gnode_tree(ppo_gnode_string)
        python_func_string = transform.transform(ppo_gnode, ppo_index = 0)
        print(python_func_string)

        ppo_gnode_string = 'W;po;AMO'
        ppo_gnode = parse_to_gnode_tree(ppo_gnode_string)
        python_func_string = transform.transform(ppo_gnode, ppo_index = 0)
        print(python_func_string)

        ppo_gnode_string = 'AMO;po;R'
        ppo_gnode = parse_to_gnode_tree(ppo_gnode_string)
        python_func_string = transform.transform(ppo_gnode, ppo_index = 0)
        print(python_func_string)

        ppo_gnode_string = 'AMO;po;W'
        ppo_gnode = parse_to_gnode_tree(ppo_gnode_string)
        python_func_string = transform.transform(ppo_gnode, ppo_index = 0)
        print(python_func_string)

