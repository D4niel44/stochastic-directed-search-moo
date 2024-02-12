import joblib
import argparse
import pandas as pd
import os
import yaml
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt

from src.timeseries.moo.experiments.config import moea_map
from src.timeseries.utils.util import write_text_file 
from src.timeseries.moo.experiments.util import get_reference_point, load_results, parse_reference_point_arg
from src.timeseries.utils.moo import sort_arr_1st_col
from src.timeseries.moo.core.harness import plot_2D_pf
from src.timeseries.moo.experiments.util import load_config
from src.timeseries.moo.experiments.util import get_best_f_moea

def get_f_from_gd_res(gd_res):
    f_res = []
    for res in gd_res:
        f_res.append([res['val_quantile_coverage_risk'][-1], res['val_quantile_estimation_risk'][-1]])
    return np.array(f_res)
 
def load_moea_group(path, moeas):
    config = load_config(path)
    n_repeat = config['number_runs']
    problem_size = config['problem_size']
    _, res_dict = load_results(path, moeas, n_repeat, problem_size)
    return res_dict
 

def preprocess_gd(path):
    gd_res = joblib.load(os.path.join(path, 'results.z'))
    return get_f_from_gd_res(gd_res)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare the Pareto front from multiple algorithms.')
    parser.add_argument('path', help='path to the config of the experiment')
    args = parser.parse_args()

    comparisons = None
    with open(os.path.join(args.path, "config.yaml"), 'r') as stream:
        comparisons = yaml.safe_load(stream)

    for title, config in chain.from_iterable(c.items() for c in comparisons['pareto_fronts']):
        fronts = []
        names = []

        for gd_algo in config['gd']:
            for name, path in gd_algo.items():
                names.append(name)
                fronts.append(preprocess_gd(path))

        for i, moea_group in enumerate(config['moeas']):
            res_dict = load_moea_group(moea_group['path'], [moea_map[n] for n in moea_group['names']])

            if 'reference_point' in moea_group:
                ref_point = parse_reference_point_arg(moea_group['reference_point'])
            else:
                ref_point = get_reference_point(res_dict)

            write_text_file(os.path.join(path, f'reference_point_group{i}'), str(ref_point))

            for name in moea_group['names']:
                fronts.append(get_best_f_moea(res_dict[moea_map[name]], ref_point))
                if 'name_suffix' in moea_group and moea_group['name_suffix']:
                    names.append(name + '_' + moea_group["name_suffix"])            
                else:
                    names.append(name)

        sorted_fronts = [sort_arr_1st_col(f) for f in fronts]

        plot_2D_pf(Fs=sorted_fronts,
                fx_inis=[],
                names=names,
                f_markersize=6,
                colors_ixs=None,
                save=True,
                label_scale=1.7,
                size=(1000, 700),
                save_pdf=False,
                save_png=False,
                title=f'Pareto front - {title}',
                img_path=os.path.join(args.path, title, 'pf'))
