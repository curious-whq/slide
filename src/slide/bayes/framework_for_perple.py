from src.slide.bayes.framework import LitmusRunner, get_score
from src.slide.bayes.litmus_params import LitmusParams
from src.slide.bayes.util import get_files, read_log_to_summary

litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive_perple"
stat_log = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record.log"
dir_path = "/home/whq/Desktop/code_list/perple_test/bayes_log"
log_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_stat.csv"
if __name__ == "__main__":
    litmus_list = get_files(litmus_path)
    for litmus in litmus_list:
        print(litmus)

    # config_list = [LitmusParams() for _ in litmus_list]
    # vector = [-1,5,-1,-1,-1,-1,-1,-1,-1,-1,1]
    # for item in config_list:
    #     item.from_vector(vector)
    #     item.apply_standard_form()
    #
    # runner = LitmusRunner(litmus_list, config_list, stat_log)
    # runner.run()

    read_log_to_summary(dir_path, log_path)
