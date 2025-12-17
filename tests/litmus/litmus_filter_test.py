from src.slide.litmus.litmus_filter import LitmusFilter
from src.slide.litmus.load_litmus import get_litmus_by_policy, GetLitmusPolicy




class TestLitmusFilter:

    def test(self):
        pass
    def filter_X_thread_pass_two(self, litmus_test):
        pass





if __name__ == '__main__':
    test = TestLitmusFilter()

    litmus_files = get_litmus_by_policy(GetLitmusPolicy.All, None)
    print(1)
    print(litmus_files)
    litmus_filter = LitmusFilter(litmus_files)
