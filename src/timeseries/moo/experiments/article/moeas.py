import argparse
import os
from itertools import chain
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.timeseries.utils.critical_difference import draw_cd_diagram
from src.timeseries.utils.moo import get_hypervolume
from src.timeseries.moo.experiments.config import moea_map
from src.timeseries.utils.util import write_text_file
from src.timeseries.moo.experiments.util import get_reference_point, load_results, parse_reference_point_arg
from src.timeseries.utils.util import latex_table, write_text_file

def plot_cd_diagram(size_dict, path, recalculate_hv=False, file_prefix=''):
    run_res = []

    prefix_i = 0
    for size, dict in size_dict.items():
        res_dict = dict['res_dict']
        ref = dict['ref_point']
        n_gen = dict['n_gen']
        for moea in moeas:
            moea_results = res_dict[moea]

            for i, res in enumerate(moea_results):
                run_res.append({
                    'classifier_name': moea.__name__,
                    'dataset_name': prefix_i + i,
                    'hv': get_hypervolume(res[n_gen - 1]['F'], ref) if recalculate_hv else res[n_gen - 1]['hv'],
                })
        prefix_i += dict['n_repeat']

    runs_df = pd.DataFrame.from_records(run_res)
    print(runs_df)
    runs_df.to_csv(f'{path}seed_results.csv')
    draw_cd_diagram(df_perf=runs_df, key_name='hv', labels=True, path=path, filename=f'{file_prefix}cd_diagram.png')

def mean_std_table(size_dict, path, recalculate_hv=False, file_prefix=''):
    gens_res = []

    
    for size, dict in size_dict.items():
        res_dict = dict['res_dict']
        times_dict = dict['times_dict']
        ref = dict['ref_point']
        n_gen = dict['n_gen']
        hv_map = {}
        c = n_gen - 1

        for moea in moeas:
            hv_map[moea] = np.mean([get_hypervolume(r[c]['F'], ref) for r in res_dict[moea]] if recalculate_hv else [r[c]['hv'] for r in res_dict[moea]])

        max_hv = max(hv_map.values())

        row = {
            'metric': 'time',
            'size': size
        }
        for moea in moeas:
            times, moea_results = times_dict[moea], res_dict[moea]
            ts = [t[c] for t in times]
            row[moea.__name__] = '{:.2f} ({:.2f})'.format(np.mean(ts), np.std(ts))
        gens_res.append(row)

        row = {
            'metric': 'hv',
            'size': size,
        }
        for moea in moeas:
            moea_results = res_dict[moea]
            hvs = [get_hypervolume(r[c]['F'], ref) for r in moea_results] if recalculate_hv else [r[c]['hv'] for r in moea_results]
            row[moea.__name__] = '{:.4f} ({:.2E})'.format(np.mean(hvs), np.std(hvs))
        gens_res.append(row)

        row = {
            'metric': 'relative hv',
            'size': size,
        }
        for moea in moeas:
            moea_results = res_dict[moea]
            hvs = [get_hypervolume(r[c]['F'], ref) for r in moea_results] if recalculate_hv else [r[c]['hv'] for r in moea_results]
            row[moea.__name__] = '{:.4f}'.format(1 - ((max_hv - hv_map[moea]) / max_hv))
        gens_res.append(row)

    gens_res_df = pd.DataFrame.from_records(gens_res).sort_values(['metric', 'size'], ascending=False)
    print(gens_res_df)
    title = f'{moea.__name__} results'
    write_text_file(os.path.join(path, f'{file_prefix}{problem_size}_results_table'),
                    latex_table(title, gens_res_df.to_latex(escape=False, index=False)))

def get_median_run(results, ref, recalculate_hv=True):
    sorted_results = sorted(results, key=lambda r: get_hypervolume(r[-1]['F'], ref) if recalculate_hv else r[-1]['hv'])
    return sorted_results[len(sorted_results)//2]


def plot_median_evo(res_dict, path, moeas, ref, n_gen, recalculate_hv=True, file_prefix=''):
    fig = plt.figure()
    fig.supxlabel("Generations")
    fig.supylabel("HV")
    for moea in moeas:
        moea_results = res_dict[moea]
        median = get_median_run(moea_results, ref)
        plt.plot([i for i in range(1, n_gen+1)], [get_hypervolume(median[i]['F'], ref) if recalculate_hv else median[i]['hv'] for i in range(n_gen)], label = moea.__name__)

    plt.title('Median evolution')
    plt.legend()
    plt.savefig(os.path.join(path, f'{file_prefix}median_exec_graph.png'))
    plt.close(fig)

def plot_pareto_front(res_dict, path, moeas, n_repeat, n_gen, file_prefix=''):
    for seed in range(n_repeat):
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.suptitle(f'Pareto front')
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

def detailed_pf(res_dict, path, moeas, n_repeat, ref_point, gens, prefix=''):
    for seed in range(n_repeat):
        for i, moea in enumerate(moeas):
            for gen in gens:
                fig = plt.figure()
                sol = res_dict[moea]
                F = sol[seed][gen]['F']
                plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
                plt.title(f'Pareto front {moea.__name__} - Gen {gen} - HV {get_hypervolume(F, ref_point)}')

                plt.savefig(os.path.join(path, f'{prefix}{moea.__name__}_seed_{seed}_gen_{gen}_pareto_front.png'))
                plt.close(fig)

# %%
if __name__ == '__main__':
    ## --------------- CFG ---------------
    parser = argparse.ArgumentParser(description='Analyze results of experiment.')
    parser.add_argument('--end', type=int, dest='end_seed', default=None, help='end running the experiment before the given seed (exclusive)')
    parser.add_argument('--use_default_hv', action='store_true', dest='default_hv', help='use the default HV instead of recalculating it')
    parser.add_argument('--ref_point', dest='ref_point', type=parse_reference_point_arg, default=None, help='provide a custom reference point for HV')
    parser.add_argument('exp_path', help='path of the experiment config')
    parser.add_argument('--file_prefix', default='', help='prefix to add to file names')
    parser.add_argument('--subfolder_name', default='results', help='name of the subfolder that will contain the output of this script (defaults to results)')

    args = parser.parse_args()

    exp_config = None
    with open(os.path.join(args.exp_path, "config.yaml"), 'r') as stream:
        exp_config = yaml.safe_load(stream)

    size_dict = {}
    for size, path in chain.from_iterable(c.items() for c in exp_config['moeas']):

        config = None
        with open(os.path.join(path, "config.yaml"), 'r') as stream:
            config = yaml.safe_load(stream)

        project = 'snp'
        problem_size = config['problem_size']
        moeas = [moea_map[m] for m in config['moeas']]
        n_repeat = args.end_seed if args.end_seed is not None else config['number_runs']
        pop_size = config['population_size']
        n_gen = config['generations']
        skip_train_metrics = config['skip_train_metrics'] if 'skip_train_metrics' in config else True
        output_path = os.path.join(args.exp_path, args.subfolder_name)
        ## ------------------------------------

        times_dict, res_dict = load_results(path, moeas, n_repeat, problem_size)
        ref_point = args.ref_point if args.ref_point is not None else get_reference_point(res_dict)
        print(ref_point)

        size_dict[size] = {
            'times_dict': times_dict,
            'res_dict': res_dict,
            'ref_point': ref_point,
            'pop_size': pop_size,
            'n_gen': n_gen,
            'n_repeat': n_repeat
        }

    recalculate_hv = not args.default_hv
    file_prefix = args.file_prefix
    os.makedirs(output_path, exist_ok=True) 

    #write_text_file(os.path.join(output_path, 'reference_point'), str(ref_point))
    plot_cd_diagram(size_dict, output_path, recalculate_hv, file_prefix)
    mean_std_table(size_dict, output_path, recalculate_hv, file_prefix)
    
    for size, m in size_dict.items():
        res_dict = m['res_dict']
        plot_median_evo(res_dict, output_path, moeas, m['ref_point'], m['n_gen'], recalculate_hv, size + '_')
        plot_pareto_front(res_dict, output_path, moeas, m['n_repeat'], m['n_gen'], size + '_')
        detailed_pf(res_dict, output_path, [moea_map['MOEAD']], m['n_repeat'], m['ref_point'], [1, 2, 5, 10, 20], size + '_')
