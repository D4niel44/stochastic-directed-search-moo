import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.timeseries.utils.critical_difference import draw_cd_diagram
from src.timeseries.utils.moo import get_hypervolume
from src.timeseries.utils.util import latex_table, write_text_file

def plot_evo(res_dict, path, moeas, n_repeat, ref, n_gen, recalculate_hv=True, file_prefix=''):
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


def get_median_run(results):
    sorted_results = sorted(results, key=lambda r: r[-1]['hv'])
    return sorted_results[len(sorted_results)//2]


def plot_median_evo(res_dict, path, moeas, ref, n_gen, recalculate_hv=True, file_prefix=''):
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


def plot_pareto_front(res_dict, path, moeas, n_repeat, n_gen, file_prefix=''):
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


def plot_cd_diagram(res_dict, path, moeas, ref, n_gen, recalculate_hv=False, file_prefix=''):
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


def mean_std_table(times_dict, res_dict, path, moeas, ref, problem_size, pop_size, n_gen, recalculate_hv=False, file_prefix=''):
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
