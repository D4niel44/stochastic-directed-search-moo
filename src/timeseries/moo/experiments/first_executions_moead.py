import joblib
import argparse
import pandas as pd
import os
import yaml
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3 
from pymoo.algorithms.moo.moead import MOEAD 
from pymoo.algorithms.moo.sms import SMSEMOA 

from src.timeseries.utils.moo import get_hypervolume

import numpy as np
import matplotlib.pyplot as plt

from timeseries.moo.experiments.config import moea_map
from src.timeseries.utils.util import write_text_file, latex_table
from src.timeseries.utils.critical_difference import draw_cd_diagram

def load_results(path, moeas, n_repeat):
    res_dict = {}
    times_dict = {}
    sol_dict = {}
    for moea in moeas:
        results = [joblib.load(f'{path}{moea.__name__}_{problem_size}_results_{seed}.z') for seed in range(n_repeat)]
        sol_dict[moea] = [r[0] for r in results]
        times_dict[moea] = [r[1] for r in results]
        res_dict[moea] = [r[2] for r in results]
    return sol_dict, times_dict, res_dict

def graph_pareto_front(res_dict, path, moeas, n_repeat, ref_point):
    for seed in range(n_repeat):
        for i, moea in enumerate(moeas):
            for gen in range(10):
                fig = plt.figure()
                sol = res_dict[moea]
                F = sol[seed][gen]['F']
                plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
                plt.title(f'Pareto front {moea.__name__} - Gen {gen} - HV {get_hypervolume(F, ref_point)}')

                plt.savefig(f'{path}{moea.__name__}_seed_{seed}_gen_{gen}_pareto_front.png')
                plt.close(fig)

def get_reference_point(sol_dict, moeas, n_repeat):
    max_f1 = 0
    max_f2 = 0
    for seed in range(n_repeat):
        for i, moea in enumerate(moeas):
            sol = sol_dict[moea]
            F = sol[seed].get("F")
            max_f1 = max(max_f1, max(F[:, 0]))
            max_f2 = max(max_f2, max(F[:, 1]))
    return max_f1 * 1.1, max_f2 * 1.1


# %%
if __name__ == '__main__':
    ## --------------- CFG ---------------
    parser = argparse.ArgumentParser(description='Analyze results of experiment.')
    parser.add_argument('--end', type=int, dest='end_seed', default=None, help='end running the experiment before the given seed (exclusive)')
    parser.add_argument('path', help='path to store the experiment folder')

    args = parser.parse_args()

    config = None
    with open(os.path.join(args.path, "config.yaml"), 'r') as stream:
        config = yaml.safe_load(stream)

    project = 'snp'
    problem_size = config['problem_size']
    moeas = [moea_map['MOEAD'], moea_map['NSGA2']]
    n_repeat = args.end_seed if args.end_seed is not None else config['number_runs']
    pop_size = config['population_size']
    n_gen = config['generations']
    path = args.path
    ## ------------------------------------

    sol_dict, times_dict, res_dict = load_results(path, moeas, n_repeat)
    ref_point = [72.07351531982422, 12.603998470306397]

    graph_pareto_front(res_dict, path, moeas, n_repeat, ref_point)
