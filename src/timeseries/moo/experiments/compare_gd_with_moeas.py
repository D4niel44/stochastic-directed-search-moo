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

from src.timeseries.moo.core.harness import get_model_and_params, get_ts_problem, get_continuation_method, \
    plot_2D_pf

import numpy as np
import matplotlib.pyplot as plt

from src.timeseries.moo.experiments.config import moea_map
from src.timeseries.utils.util import write_text_file, latex_table
from src.timeseries.utils.critical_difference import draw_cd_diagram

def get_median_run(results):
    sorted_results = sorted(results, key=lambda r: r[-1]['hv'])
    return sorted_results[len(sorted_results)//2]

def load_results(path, moeas, n_repeat, skip_train_metrics=True):
    res_dict = {}
    times_dict = {}
    last_idx = -2 if skip_train_metrics else -3
    for moea in moeas:
        results = [joblib.load(f'{path}{moea.__name__}_{problem_size}_results_{seed}.z')[last_idx:] for seed in range(n_repeat)]
        times_dict[moea] = [r[0] for r in results]
        res_dict[moea] = [r[1] for r in results]
    return times_dict, res_dict

def graph_evo(res_dict, path, moeas, n_repeat, ref, recalculate_hv=True, file_prefix=''):
    for seed in range(n_repeat):
        fig = plt.figure()
        fig.supxlabel("Generaciones")
        fig.supylabel("HV")

        for moea in moeas:
            res = res_dict[moea][seed]

            plt.plot([i for i in range(1, n_gen+1)], [get_hypervolume(res[i]['F'], ref) if recalculate_hv else res[i]['hv'] for i in range(n_gen)], label = moea.__name__)

        plt.title(f'Evolución semilla {seed}')
        plt.legend()
        plt.savefig(os.path.join(path, f'{file_prefix}seed_{seed}_evo_graph.png'))
        plt.close(fig)

def graph_median_evo(res_dict, path, moeas, ref, recalculate_hv=True, file_prefix=''):
    fig = plt.figure()
    fig.supxlabel("Generaciones")
    fig.supylabel("HV")
    for moea in moeas:
        moea_results = res_dict[moea]
        median = get_median_run(moea_results)
        plt.plot([i for i in range(1, n_gen+1)], [get_hypervolume(median[i]['F'], ref) if recalculate_hv else median[i]['hv'] for i in range(n_gen)], label = moea.__name__)

    plt.title('Evolución Mediana')
    plt.legend()
    plt.savefig(os.path.join(path, f'{file_prefix}median_exec_graph.png'))
    plt.close(fig)

def graph_pareto_front(res_dict, path, moeas, n_repeat, file_prefix=''):
    for seed in range(n_repeat):
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.suptitle(f'Pareto front - Semilla {seed}')
        axs = axs.flat
        for i, moea in enumerate(moeas):
            sol = res_dict[moea]
            F = sol[seed][n_gen - 1]['F']
            axs[i].scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
            axs[i].set_title(f"{moea.__name__}")

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs:
            ax.label_outer()

        plt.savefig(os.path.join(path, f'{file_prefix}seed_{seed}_pareto_front.png'))
        plt.close(fig)

def get_reference_point(res_dict, moeas, n_repeat):
    max_f1 = 0
    max_f2 = 0
    for seed in range(n_repeat):
        for i, moea in enumerate(moeas):
            sol = res_dict[moea]
            F = sol[seed][n_gen - 1]['F']
            max_f1 = max(max_f1, max(F[:, 0]))
            max_f2 = max(max_f2, max(F[:, 1]))
    return max_f1 * 1.1, max_f2 * 1.1

def graph_cd_diagram(res_dict, path, moeas, ref, recalculate_hv=False, file_prefix=''):
    run_res = []
    for moea in moeas:
        moea_results = res_dict[moea]

        for i, res in enumerate(moea_results):
            run_res.append({
                'classifier_name': moea.__name__,
                'dataset_name': i,
                'hv': get_hypervolume(res[n_gen - 1]['F'], ref) if recalculate_hv else res[n_gen - 1]['hv'],
            })

    runs_df = pd.DataFrame.from_records(run_res)
    print(runs_df)
    runs_df.to_csv(f'{path}seed_results.csv')
    draw_cd_diagram(df_perf=runs_df, key_name='hv', labels=True, path=path, filename=f'{file_prefix}cd_diagram.png')

def mean_std_table(times_dict, res_dict, path, moeas, ref, recalculate_hv=False, file_prefix=''):
    gens_res = []
    for moea in moeas:
        times, moea_results = times_dict[moea], res_dict[moea]

        c = n_gen - 1
        ts = [t[c] for t in times]

        distances = [r[c]['distances'] for r in moea_results]
        distances = [item for listoflists in distances for item in listoflists]
        hvs = [get_hypervolume(r[c]['F'], ref) for r in moea_results] if recalculate_hv else [r[c]['hv'] for r in moea_results]

        gens_res.append({'method': moea.__name__,
                         'time': '{:.2f} ({:.2f})'.format(np.mean(ts), np.std(ts)),
                         'hv': '{:.4f} ({:.2E})'.format(np.mean(hvs), np.std(hvs)),
                         'distance': '{:.2E} ({:.2E})'.format(np.mean(distances), np.std(distances)),
                         'f_evals': '{:,.2f} ({:.2f})'.format((c+1)*pop_size, 0)})

    gens_res_df = pd.DataFrame.from_records(gens_res)
    print(gens_res_df)
    title = f'{moea.__name__} results'
    write_text_file(os.path.join(path, f'{file_prefix}{problem_size}_results_table'),
                    latex_table(title, gens_res_df.to_latex(escape=False, index=False)))

def parse_reference_point_arg(str):
    return [float(n) for n in str.split(',')]

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
               save_pdf=True,
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

    times_dict, res_dict = load_results(path_moea, moeas, n_repeat, skip_train_metrics)
    ref_point = get_reference_point(res_dict, moeas, n_repeat)

    os.makedirs(output_path, exist_ok=True)


    write_text_file(os.path.join(output_path, 'reference_point'), str(ref_point))
    graph_pf_gd_vs_moea(res_dict, get_f_from_gd_res(gd_res), ref_point, n_gen, output_path)
