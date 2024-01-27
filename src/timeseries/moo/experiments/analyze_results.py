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
# %%
if __name__ == '__main__':
    ## --------------- CFG ---------------
    parser = argparse.ArgumentParser(description='Analyze results of experiment.')
    parser.add_argument('--end', type=int, dest='end_seed', default=None, help='end running the experiment before the given seed (exclusive)')
    parser.add_argument('--use_default_hv', action='store_true', dest='default_hv', help='use the default HV instead of recalculating it')
    parser.add_argument('--ref_point', dest='ref_point', type=parse_reference_point_arg, default=None, help='provide a custom reference point for HV')
    parser.add_argument('path', help='path to store the experiment folder')
    parser.add_argument('--file_prefix', default='', help='prefix to add to file names')
    parser.add_argument('--subfolder_name', default='results', help='name of the subfolder that will contain the output of this script (defaults to results)')

    args = parser.parse_args()

    config = None
    with open(os.path.join(args.path, "config.yaml"), 'r') as stream:
        config = yaml.safe_load(stream)

    project = 'snp'
    problem_size = config['problem_size']
    moeas = [moea_map[m] for m in config['moeas']]
    n_repeat = args.end_seed if args.end_seed is not None else config['number_runs']
    pop_size = config['population_size']
    n_gen = config['generations']
    skip_train_metrics = config['skip_train_metrics'] if 'skip_train_metrics' in config else True
    path = args.path
    output_path = os.path.join(path, args.subfolder_name)
    file_prefix = args.file_prefix
    ## ------------------------------------

    times_dict, res_dict = load_results(path, moeas, n_repeat, skip_train_metrics)

    ref_point = args.ref_point if args.ref_point is not None else get_reference_point(res_dict, moeas, n_repeat)
    recalculate_hv = not args.default_hv

    os.makedirs(output_path, exist_ok=True) 

    write_text_file(os.path.join(output_path, 'reference_point'), str(ref_point))
    graph_evo(res_dict, output_path, moeas, n_repeat, ref_point, recalculate_hv, file_prefix)
    graph_median_evo(res_dict, output_path, moeas, ref_point, recalculate_hv, file_prefix)
    graph_pareto_front(res_dict, output_path, moeas, n_repeat, file_prefix)
    graph_cd_diagram(res_dict, output_path, moeas, ref_point, recalculate_hv, file_prefix)
    mean_std_table(times_dict, res_dict, output_path, moeas, ref_point, recalculate_hv, file_prefix)
