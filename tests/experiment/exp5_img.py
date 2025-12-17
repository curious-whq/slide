import copy
import itertools
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from src.slide import config


model = {
    'RW' : ['[R];po;[W]'],
    'AMO_X' : ['[AMO];po;[M]', '[M];po;[AMO]', '[X];po;[M]', '[M];po;[X]'],
    'fence' : ['[M];fencerel(Fence.r.r);[M]', '[M];fencerel(Fence.r.w);[M]', '[M];fencerel(Fence.r.rw);[M]',
               '[M];fencerel(Fence.w.r);[M]', '[M];fencerel(Fence.w.w);[M]', '[M];fencerel(Fence.w.rw);[M]',
               '[M];fencerel(Fence.rw.r);[M]', '[M];fencerel(Fence.rw.w);[M]', '[M];fencerel(Fence.rw.rw);[M]',
               '[M];fencerel(Fence.tso);[M]'],
    'SC' : ['[M];po;[M]'],
    'TSO' : ['[R];po;[M]', '[W];po;[W]'],
    'strong_ppo12' : ['[M];(addr|data);[W];po;[R]'],
    'strong_ppo13' : ['[M];addr;[M];po;[R]']
}
variants = ['RW', 'AMO_X', 'fence', 'strong_ppo12', 'strong_ppo13']

exp_5_output_log_dir = os.path.join(config.TEST_DIR,'results/exp5_result')
dif_log_dir = os.path.join(config.HERD_LOG_DIR_PATH, f'variant/dif')
def pre():
    assert len(os.listdir(exp_5_output_log_dir)) != 0, "must pass exp5 synth rvwmo variant post"
exp5_png_path = os.path.join(config.TEST_DIR, 'results/exp5_result.png')


new_model = {}
SC_model = []
TSO_model = []
def get_model_data():
    global new_model, SC_model, TSO_model

    keys = list(model.keys())
    for r in range(1, len(keys) + 1):
        for key_combo in itertools.combinations(keys, r):
            if 'TSO' in key_combo and len(key_combo) > 1:
                continue
            if 'SC' in key_combo and len(key_combo) > 1:
                continue
            new_key = '_'.join(key_combo)
            new_model[new_key] = []
            for key in variants:
                if key in key_combo:
                    new_model[new_key].append('✓')
                else:
                    new_model[new_key].append('')

    # for root, dirs, files in os.walk(dif_log_dir):
    #     for file in files:
    #         file_path = os.path.join(root, file)
    #         if not file.startswith('dif_'):
    #             continue
    #         model_name = os.path.splitext(file)[0][4:-6]
    #         print(model_name)
    #         with open(file_path, 'r') as f:
    #             if model_name in new_model:
    #                 new_model[model_name].append(len(f.readlines()))



    for root, dirs, files in os.walk(exp_5_output_log_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.startswith('result_'):
                continue
            model_name = os.path.splitext(file)[0][7:]
            print(model_name)
            lines = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if model_name not in new_model:
                    continue
                new_model[model_name].append(len(lines)-1)
                match = re.search(r"[-+]?\d*\.\d+|\d+", lines[-1])
                if match:
                    num = float(match.group())
                    rounded = round(num, 2)
                    new_model[model_name].append(f'{rounded}s')
    for k in new_model:
        print(k)
        print(new_model[k])

    SC_model = copy.deepcopy(new_model['SC'])
    TSO_model = copy.deepcopy(new_model['TSO'])
    new_model.pop('SC')
    new_model.pop('TSO')


def draw_img():

    data = list(new_model.values())

    columns = variants + ['Synth PPO Nums', 'Time']


    df = pd.DataFrame(data, columns=columns)

    rows = len(new_model)
    cols = 7  #

    fig_height = max(5, rows * 0.3)  #
    fig_width = max(8, cols * 1.3)


    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')  #


    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     loc='center',
                     cellLoc='center',
                     colLoc='center')

    for key, cell in table.get_celld().items():
        row, col = key
        cell.set_linewidth(1)

        if row == 0:
            cell.visible_edges = 'open'
            cell.set_text_props(weight='bold')


        elif row == rows:
            cell.visible_edges = 'open'


        else:
            cell.visible_edges = 'TB'

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()

    plt.savefig(exp5_png_path, bbox_inches='tight', dpi=300)
    # plt.savefig("variant_summary_table.png", bbox_inches='tight')
    

if __name__ == '__main__':
    pre()
    get_model_data()
    draw_img()
