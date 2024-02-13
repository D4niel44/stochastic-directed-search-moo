import joblib
import argparse
import os
import yaml
import math

import numpy as np

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


def get_best_idx_for_weight(F, weight):
    return np.argmin([weight * f[0] + (1 - weight) * f[1] for f in F])
 
def preprocess_gd(path):
    gd_res = joblib.load(os.path.join(path, 'results.z'))

    for i, res in enumerate(gd_res):
        res['exp_path'] = path
        res['exp_run_idx'] = i
    return gd_res
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare the Pareto front from multiple algorithms.')
    parser.add_argument('path', help='path to the config of the experiment')
    args = parser.parse_args()

    config = None
    with open(os.path.join(args.path, "config.yaml"), 'r') as stream:
        config = yaml.safe_load(stream)

    sols = []    
    for gd_exp in config['paths']:
        for name, path in gd_exp.items():
            gd_res = preprocess_gd(path)
            sols = sols + gd_res
    
    F = get_f_from_gd_res(sols)

    improved_sols = []
    for weight in config['combinations']:
        best_idx = get_best_idx_for_weight(F, weight)
        improved_sols.append(sols[best_idx])
        print(weight, best_idx, F[best_idx], sols[best_idx]['exp_path'], sols[best_idx]['exp_run_idx'])

    joblib.dump(
        improved_sols,
        os.path.join(args.path, f'results.z'),
        compress=3)
