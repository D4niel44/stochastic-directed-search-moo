import joblib
import sys
import argparse
import os
from itertools import chain
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.timeseries.utils.critical_difference import draw_cd_diagram
from src.timeseries.utils.moo import get_hypervolume
from src.timeseries.moo.experiments.config import moea_map
from src.timeseries.utils.util import write_text_file
from src.timeseries.moo.experiments.util import load_moea_results_to_exp, load_ws_results_to_exp
from src.timeseries.utils.util import latex_table, write_text_file
from src.timeseries.moo.experiments.experiment import CompositeExperimentBuilder

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'font.size' : 12,
    'pgf.rcfonts': False,
})

NAMES_TO_TITLE = {
        "NSGA2": "NSGA-II",
        "NSGA3": "NSGA-III",
        "MOEAD": "MOEA/D",
        "SMSEMOA": "SMS-EMOA",
        "WS": "Suma ponderada",
}

def get_metrics_from_ws_res(ws_res):
    f_res = []
    times_res = []
#    for res in ws_res:
#        qcr = res['val_quantile_coverage_risk']
#        qer = res['val_quantile_estimation_risk']
#        f_res.append([[qcr[i], qer[i]] for i in range(len(qcr))])
#        times_res.append(res['time'])

    for res in ws_res:
        qcr = res['val_quantile_coverage_risk']
        qer = res['val_quantile_estimation_risk']
        f_res.append([qcr[-1], qer[-1]])
        times_res.append(res['time'])
    return [np.array(f_res)], np.array(times_res)


def compute_hv(algo, res, ref, recalculate_hv=False):
    if algo == 'WS':
        return get_hypervolume(res, ref)
    else:
        return get_hypervolume(res[-1]['F'], ref) if recalculate_hv else res[-1]['hv']
 
def plot_cd_diagram(experiments, n_runs, path, file_prefix=''):
    run_res = []

    # Weighted Sum
    i = 0
    for size, exp in experiments.name_iter('WS'):
        ref_point = experiments.compute_reference_point(size)
        for res in exp:
            # Copy WS result for each run.
            for _ in range(n_runs):
                run_res.append({
                    'classifier_name': exp.get_name(),
                    'dataset_name': i,
                    'hv': res.compute_hv(ref_point),
                })
                i += 1

    # MOEAs
    for name in ['NSGA2', 'NSGA3', 'MOEAD', 'SMSEMOA']:
        i = 0
        for size, exp in experiments.name_iter(name):
            ref_point = experiments.compute_reference_point(size)
            for res in exp:
                run_res.append({
                    'classifier_name': NAMES_TO_TITLE[exp.get_name()],
                    'dataset_name': i,
                    'hv': res.compute_hv(ref_point),
                })
                i += 1

    runs_df = pd.DataFrame.from_records(run_res)
    runs_df.to_csv(f'{path}seed_results.csv')
    draw_cd_diagram(df_perf=runs_df, key_name='hv', labels=True, path=path, filename=f'{file_prefix}cd_diagram.png')
    draw_cd_diagram(df_perf=runs_df, key_name='hv', labels=True, path=path, filename=f'{file_prefix}cd_diagram.pgf')

def mean_std_table(experiments, path, file_prefix=''):
    gens_res = []

    for size in experiments.get_problem_sizes():
        t_row = {
            'metric': 'time',
            'size': size,
        }
        hv_row = {
            'metric': 'hv',
            'size': size,
        }
        for name, exp in experiments.size_iter(size):
            t_mean, t_std = exp.get_mean_std_time()
            t_row[name] = '{:.2f} ({:.2f})'.format(t_mean, t_std)

            hv_mean, hv_std = exp.get_mean_std_hv(experiments.compute_reference_point(size))
            hv_row[NAMES_TO_TITLE[name]] = '{:.4f} ({:.2E})'.format(hv_mean, hv_std)

        gens_res.append(t_row)
        gens_res.append(hv_row)

    # TODO: Relative hv

    gens_res_df = pd.DataFrame.from_records(gens_res).sort_values(['metric', 'size'], ascending=False)
    print(gens_res_df)
    title = f'Table results'
    write_text_file(os.path.join(path, f'{file_prefix}results_table'),
                    latex_table(title, gens_res_df.to_latex(escape=False, index=False)))

def get_median_run(results, ref, recalculate_hv=True):
    sorted_results = sorted(results, key=lambda r: get_hypervolume(r[-1]['F'], ref) if recalculate_hv else r[-1]['hv'])
    return sorted_results[len(sorted_results)//2]


def plot_median_evo(experiments, size, path, file_prefix=''):
    ref_point = experiments.compute_reference_point(size)

    fig = plt.figure()
    fig.supxlabel("generaciones")
    fig.supylabel("hv")

    for name, exp in experiments.size_iter(size):
        res = exp.get_median_result(ref_point)
        if not res.is_multi_gen_exp():
            # We want to skip Weighted sum, since plotting HV per gen doesn't make sense
            continue
        x_axis = [i for i in range(1, res.get_generations()+1)]
        y_axis = res.compute_hv_per_generation(ref_point)

        plt.plot(x_axis, y_axis, label = NAMES_TO_TITLE[exp.get_name()])


    #plt.title('Median evolution')
    plt.legend()
    plt.savefig(os.path.join(path, f'{file_prefix}_{size}_median_evo_graph.pgf'))
    plt.savefig(os.path.join(path, f'{file_prefix}_{size}_median_evo_graph.png'))
    plt.close(fig)

def plot_pareto_front(experiments, size, path, file_prefix=''):
    fig = plt.figure(constrained_layout=True)
    axd = fig.subplot_mosaic(
    """
    AABBCC
    .DDEE.
    """
    )

    names_to_axes = {
        "NSGA2": "A",
        "NSGA3": "B",
        "MOEAD": "C",
        "SMSEMOA": "D",
        "WS": "E",
    }

    

    ref_point = experiments.compute_reference_point(size)

    # Create a sub dir for pareto fronts of this size
    subdir_name = f'{size}_pareto_front'
    Path(os.path.join(path, subdir_name)).mkdir(exist_ok=True)

    for name, exp in experiments.size_iter(size):
        res = exp.get_median_result(ref_point)
        F = res.get_evaluation()

        axd[names_to_axes[name]].scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
        axd[names_to_axes[name]].set_title(NAMES_TO_TITLE[name])

    fig.savefig(os.path.join(path, subdir_name, f'{file_prefix}pareto_front.png'))
    fig.savefig(os.path.join(path, subdir_name, f'{file_prefix}pareto_front.pgf'))
    plt.close(fig)

# %%
if __name__ == '__main__':
    ## --------------- CFG ---------------
    parser = argparse.ArgumentParser(description='Analyze results of experiment.')
    parser.add_argument('--end', type=int, dest='end_seed', default=None, help='end running the experiment before the given seed (exclusive)')
    #parser.add_argument('--ref_point', dest='ref_point', type=parse_reference_point_arg, default=None, help='provide a custom reference point for HV')
    parser.add_argument('exp_path', help='path of the experiment config')
    parser.add_argument('--file_prefix', default='', help='prefix to add to file names')
    parser.add_argument('--subfolder_name', default='results', help='name of the subfolder that will contain the output of this script (defaults to results)')

    args = parser.parse_args()

    exp_config = None
    with open(os.path.join(args.exp_path, "config.yaml"), 'r') as stream:
        exp_config = yaml.safe_load(stream)

    exp_builder = CompositeExperimentBuilder()
    exp_builder.set_number_objectives(2)

    # Gradient Descent
    for size, ws_path in chain.from_iterable(c.items() for c in exp_config['gd']):
        ws_exp = load_ws_results_to_exp(ws_path, size)
        exp_builder.add_experiment(ws_exp)

    # MOEAs
    for size, path in chain.from_iterable(c.items() for c in exp_config['moeas']):
        config = None
        with open(os.path.join(path, "config.yaml"), 'r') as stream:
            config = yaml.safe_load(stream)

        n_gen = config['generations']
        exp_builder.set_number_generations(size, n_gen)

        problem_size = config['problem_size']
        moeas = [moea_map[m] for m in config['moeas']]
        n_repeat = args.end_seed if args.end_seed is not None else config['number_runs']
        exp_for_size = load_moea_results_to_exp(path, moeas, n_repeat, problem_size)

        for exp in exp_for_size:
            exp_builder.add_experiment(exp)

    experiments = exp_builder.build()
    output_path = os.path.join(args.exp_path, args.subfolder_name)
    file_prefix = args.file_prefix

    plot_cd_diagram(experiments, int(n_repeat), output_path, file_prefix)
    mean_std_table(experiments, output_path, file_prefix)

    for size in experiments.get_problem_sizes():
        plot_median_evo(experiments, size, output_path)
        plot_pareto_front(experiments, size, output_path)

