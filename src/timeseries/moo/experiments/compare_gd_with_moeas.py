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
from src.timeseries.utils.util import write_text_file, latex_table
from src.timeseries.utils.critical_difference import draw_cd_diagram
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

def graph_pf_gd_vs_moea(moea_results, gd_f_results, ref_point, n_gen, output_path):
    smsemoa_f_results =  get_best_f_moea(moea_results[moea_map['SMSEMOA']], ref_point, n_gen)
    nsgaii_f_results =  get_best_f_moea(moea_results[moea_map['NSGA2']], ref_point, n_gen)
    moead_f_results =  get_best_f_moea(moea_results[moea_map['MOEAD']], ref_point, n_gen)
    nsgaiii_f_results =  get_best_f_moea(moea_results[moea_map['NSGA3']], ref_point, n_gen)
    plot_2D_pf(Fs=[gd_f_results, smsemoa_f_results, nsgaii_f_results, nsgaiii_f_results, moead_f_results],
               fx_inis=[],
               names=['GD', 'SMSEMOA', 'NSGA2', 'NSGA3', 'MOEAD'],
               f_markersize=6,
               colors_ixs=[0, 2, 1, 3, 4],
               save=True,
               label_scale=1.7,
               size=(1000, 700),
               save_pdf=False,
               save_png=False,
               img_path=os.path.join(output_path, 'pf'))


def get_f_from_gd_res(gd_res):
    f_res = []
    for res in gd_res:
        f_res.append([res['val_quantile_coverage_risk'][-1], res['val_quantile_estimation_risk'][-1]])
    return np.array(f_res)
 
# %%
if __name__ == '__main__':
    ## --------------- CFG ---------------
    path_moea = './output/exp_large_p100_g200_r20/'
    config = None
    with open(os.path.join(path_moea, "config.yaml"), 'r') as stream:
        config = yaml.safe_load(stream)

    problem_size = config['problem_size']
    moeas = [moea_map[m] for m in ['NSGA2', 'NSGA3', 'MOEAD', 'SMSEMOA']]
    n_repeat = config['number_runs']
    pop_size = config['population_size']
    n_gen = config['generations']
    skip_train_metrics = config['skip_train_metrics'] if 'skip_train_metrics' in config else True
    output_path = os.path.join('./output/comparison', 'smsemoa_vs_gd')
    ## ------------------------------------
    path_gd = './output/exp_large_n20_normFalse'
    config_gd = None
    with open(os.path.join(path_gd, "config.yaml"), 'r') as stream:
        config_gd = yaml.safe_load(stream)

    gd_res = joblib.load(os.path.join(path_gd, 'results.z'))

    times_dict, res_dict = load_results(path_moea, moeas, n_repeat, problem_size)
    ref_point = get_reference_point(res_dict, moeas, n_repeat, n_gen)

    os.makedirs(output_path, exist_ok=True)

    write_text_file(os.path.join(output_path, 'reference_point'), str(ref_point))
    graph_pf_gd_vs_moea(res_dict, get_f_from_gd_res(gd_res), ref_point, n_gen, output_path)
