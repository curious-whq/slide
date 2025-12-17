
"""Test for File Utils"""

import sys
sys.path.append('../src')
from src.slide.utils.file_util import *
from src.slide.utils.str_util import *

class TestFile:

    def test_list_files(self):
        print(list_files('input/litmus', '.litmus'))

    def test_read_files(self):
        print(read_files('input/litmus', '.litmus'))
    
    def test_prefix(self):
        print(remove_prefix('hello,world','world'))
    
    def test_suffix(self):
        print(remove_suffix('hello,world','hello'))