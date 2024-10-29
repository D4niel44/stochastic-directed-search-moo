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
from src.timeseries.moo.experiments.experiment import CompositeExperimentBuilder, CompositeExperiment

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

class ResultVisualization():

    def __init__(self, experiments: CompositeExperiment, path, n_runs, names_to_title, file_prefix=''):
        self.experiments = experiments
        self.path = path
        self.file_prefix = file_prefix
        self.n_runs = n_runs
        self.names_to_title = names_to_title


    def plot_cd_diagram(self):
        run_res = []

        # Weighted Sum
        i = 0
        for size, exp in self.experiments.name_iter('WS'):
            ref_point = self.experiments.compute_reference_point(size)
            for res in exp:
                # Copy WS result for each run.
                for _ in range(self.n_runs):
                    run_res.append({
                        'classifier_name': exp.get_name(),
                        'dataset_name': i,
                        'hv': res.compute_hv(ref_point),
                    })
                    i += 1

        # MOEAs
        for name in ['NSGA2', 'NSGA3', 'MOEAD', 'SMSEMOA']:
            i = 0
            for size, exp in self.experiments.name_iter(name):
                ref_point = self.experiments.compute_reference_point(size)
                for res in exp:
                    run_res.append({
                        'classifier_name': NAMES_TO_TITLE[exp.get_name()],
                        'dataset_name': i,
                        'hv': res.compute_hv(ref_point),
                    })
                    i += 1

        runs_df = pd.DataFrame.from_records(run_res)
        runs_df.to_csv(os.path.join(self.path, 'seed_results.csv'))
        draw_cd_diagram(df_perf=runs_df, key_name='hv', labels=True, path=self.path, filename=f'{self.file_prefix}cd_diagram.png')
        draw_cd_diagram(df_perf=runs_df, key_name='hv', labels=True, path=self.path, filename=f'{self.file_prefix}cd_diagram.pgf')

    def mean_std_table(self):
        gens_res = []

        for size in self.experiments.get_problem_sizes():
            t_row = {
                'metric': 'time',
                'size': size,
            }
            hv_row = {
                'metric': 'hv',
                'size': size,
            }
            for name, exp in self.experiments.size_iter(size):
                t_mean, t_std = exp.get_mean_std_time()
                t_row[name] = '{:.2f} ({:.2f})'.format(t_mean, t_std)

                hv_mean, hv_std = exp.get_mean_std_hv(self.experiments.compute_reference_point(size))
                hv_row[NAMES_TO_TITLE[name]] = '{:.4f} ({:.2E})'.format(hv_mean, hv_std)

            gens_res.append(t_row)
            gens_res.append(hv_row)

        # TODO: Relative hv

        gens_res_df = pd.DataFrame.from_records(gens_res).sort_values(['metric', 'size'], ascending=False)
        print(gens_res_df)
        title = f'Table results'
        write_text_file(os.path.join(self.path, f'{self.file_prefix}results_table'),
                        latex_table(title, gens_res_df.to_latex(escape=False, index=False)))

    def plot_median_evo(self, size):
        ref_point = self.experiments.compute_reference_point(size)

        fig = plt.figure()
        fig.supxlabel("generaciones")
        fig.supylabel("hv")

        for name, exp in self.experiments.size_iter(size):
            if not exp.is_multi_gen_exp():
                # We want to skip Weighted sum, since plotting HV per gen doesn't make sense
                continue
            x_axis = [i for i in range(1, exp.get_generations()+1)]
            res = exp.get_median_result(ref_point)
            y_axis = res.compute_hv_per_generation(ref_point)

            plt.plot(x_axis, y_axis, label = NAMES_TO_TITLE[name])


        #plt.title('Median evolution')
        plt.legend()
        plt.savefig(os.path.join(path, f'{self.file_prefix}_{size}_median_evo_graph.pgf'))
        plt.savefig(os.path.join(path, f'{self.file_prefix}_{size}_median_evo_graph.png'))
        plt.close(fig)

    def plot_pareto_front(self, size):
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


        ref_point = self.experiments.compute_reference_point(size)

        # Create a sub dir for pareto fronts of this size
        subdir_name = f'{size}_pareto_front'
        Path(os.path.join(self.path, subdir_name)).mkdir(exist_ok=True)

        for name, exp in self.experiments.size_iter(size):
            res = exp.get_median_result(ref_point)
            F = res.get_evaluation()

            axd[names_to_axes[name]].scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
            axd[names_to_axes[name]].set_title(NAMES_TO_TITLE[name])

        fig.savefig(os.path.join(self.path, subdir_name, f'{self.file_prefix}pareto_front.png'))
        fig.savefig(os.path.join(self.path, subdir_name, f'{self.file_prefix}pareto_front.pgf'))
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

    vis = ResultVisualization(experiments, output_path, int(n_repeat), NAMES_TO_TITLE, file_prefix)

    vis.plot_cd_diagram()
    vis.mean_std_table()

    for size in experiments.get_problem_sizes():
        vis.plot_median_evo(size)
        vis.plot_pareto_front(size)

