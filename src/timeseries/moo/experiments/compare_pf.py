import joblib
import argparse
import pandas as pd
import os
import yaml

from src.timeseries.utils.moo import get_hypervolume

from src.timeseries.moo.core.harness import plot_2D_pf

import numpy as np
import matplotlib.pyplot as plt

from src.timeseries.moo.experiments.config import moea_map
from src.timeseries.utils.util import write_text_file 
from src.timeseries.moo.experiments.utils import get_reference_point, load_results

def get_best_f_moea(moea_results, ref_point, n_gen):
    best_idx = 0
    best_hv = 0
    for i, r in enumerate(moea_results):
        hv = get_hypervolume(r[n_gen - 1]['F'], ref_point)
        if hv > best_hv:
            best_idx = i
            hv = best_hv
    return moea_results[best_idx][n_gen - 1]['F']

def get_f_from_gd_res(gd_res):
    f_res = []
    for res in gd_res:
        f_res.append([res['val_quantile_coverage_risk'][-1], res['val_quantile_estimation_risk'][-1]])
    return np.array(f_res)
 
def preprocess_moea(name, path):
    config = load_config(path)
    n_gen = config['generations']
    n_repeat = config['number_runs']
    problem_size = config['problem_size']
    moea = moea_map[name]
    _, res_dict = load_results(path, [moea], n_repeat, problem_size)
    ref_point = get_reference_point(res_dict, [moea], n_repeat, n_gen)
    return get_best_f_moea(res_dict[moea], ref_point, n_gen)

def preprocess_gd(path):
    gd_res = joblib.load(os.path.join(path, 'results.z'))
    return get_f_from_gd_res(gd_res)
    
def load_config(path):
    with open(os.path.join(path, "config.yaml"), 'r') as stream:
        config = yaml.safe_load(stream)
    return config
    
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare the Pareto front from multiple algorithms.')
    parser.add_argument('path', help='path to the config of the experiment')
    args = parser.parse_args()

    config = None
    with open(os.path.join(args.path, "config.yaml"), 'r') as stream:
        config = yaml.safe_load(stream)
    print(config)

    fronts = []
    names = []

    for gd_algo in config['gd']:
        for name, path in gd_algo.items():
            names.append(name)
            fronts.append(preprocess_gd(path))

    for moea in config['moeas']:
        for name, path in moea.items():
            names.append(name)            
            fronts.append(preprocess_moea(name, path))

    plot_2D_pf(Fs=fronts,
               fx_inis=[],
               names=names,
               f_markersize=6,
               colors_ixs=None,
               save=True,
               label_scale=1.7,
               size=(1000, 700),
               save_pdf=False,
               save_png=False,
               img_path=os.path.join(path, 'pf'))
