import os
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from src.timeseries.moo.experiments.experiment import CompositeExperiment, Tag, name_grouping, tag_grouping
from src.timeseries.utils.critical_difference import draw_cd_diagram
from src.timeseries.utils.util import latex_table, write_text_file

class ResultVisualization():

    def __init__(self, experiments: CompositeExperiment, path, n_runs, names_to_title, file_prefix=''):
        self.experiments = experiments
        self.path = path
        self.file_prefix = file_prefix
        self.n_runs = n_runs
        self.names_to_title = names_to_title

        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'font.size' : 12,
            'pgf.rcfonts': False,
        })



    def plot_cd_diagram(self, group_key_func=name_grouping(), group_to_title=None, filter_func=lambda _: True, prefix=''):
        if not group_to_title:
            group_to_title = self.names_to_title

        run_res = []

        for group_key, experiments in self.experiments.group_by(group_key_func):
            i = 0
            for exp in filter(filter_func, experiments):
                size = exp.get_problem_size()
                title = group_to_title[group_key]
                if not exp.is_multi_gen_exp():
                    ref_point = self.experiments.compute_reference_point(size)
                    for res in exp:
                        # Copy WS result for each run.
                        for _ in range(self.n_runs):
                            run_res.append({
                                'classifier_name': title,
                                'dataset_name': i,
                                'hv': res.compute_hv(ref_point),
                            })
                            i += 1
                else:
                    ref_point = self.experiments.compute_reference_point(size)
                    for res in exp:
                        run_res.append({
                            'classifier_name': title,
                            'dataset_name': i,
                            'hv': res.compute_hv(ref_point),
                        })
                        i += 1

        runs_df = pd.DataFrame.from_records(run_res)
        runs_df.to_csv(os.path.join(self.path, 'seed_results.csv'))
        draw_cd_diagram(df_perf=runs_df, key_name='hv', labels=True, path=self.path, filename=f'{self.file_prefix}{prefix}cd_diagram.png')
        draw_cd_diagram(df_perf=runs_df, key_name='hv', labels=True, path=self.path, filename=f'{self.file_prefix}{prefix}cd_diagram.pgf')

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
                t_row[self.names_to_title[name]] = '{:.2f} ({:.2f})'.format(t_mean, t_std)

                hv_mean, hv_std = exp.get_mean_std_hv(self.experiments.compute_reference_point(size))
                hv_row[self.names_to_title[name]] = '{:.4f} ({:.2E})'.format(hv_mean, hv_std)

            gens_res.append(t_row)
            gens_res.append(hv_row)

        # TODO: Relative hv

        gens_res_df = pd.DataFrame.from_records(gens_res).sort_values(['metric', 'size'], ascending=False)
        print(gens_res_df)
        title = f'Table results'
        write_text_file(os.path.join(self.path, f'{self.file_prefix}results_table'),
                        latex_table(title, gens_res_df.to_latex(escape=False, index=False)))

    def mean_std_table_mb(self):
        gens_res = []

        for size in self.experiments.get_problem_sizes():
           for batch_size, exps in self.experiments.group_by(tag_grouping(Tag.BATCH_SIZE)):
                t_row = {
                    'metric': 'time',
                    'size': size,
                    'batch_size': batch_size
                }
                hv_row = {
                    'metric': 'hv',
                    'size': size,
                    'batch_size': batch_size
                }

                for exp in exps:
                    name = exp.get_tag(Tag.ALGORITHM)

                    # Time
                    t_mean, t_std = exp.get_mean_std_time()
                    t_row[self.names_to_title[name]] = '{:.2f} ({:.2f})'.format(t_mean, t_std)

                    # Hv
                    hv_mean, hv_std = exp.get_mean_std_hv(self.experiments.compute_reference_point(size))
                    hv_row[self.names_to_title[name]] = '{:.4f} ({:.2E})'.format(hv_mean, hv_std)

                gens_res.append(t_row)
                gens_res.append(hv_row)

        # TODO: Relative hv

        gens_res_df = pd.DataFrame.from_records(gens_res).sort_values(['metric', 'size', 'batch_size'], ascending=False)
        print(gens_res_df)
        title = f'Table results'
        write_text_file(os.path.join(self.path, f'{self.file_prefix}results_table'),
                        latex_table(title, gens_res_df.to_latex(escape=False, index=False)))


    def plot_median_evo(self):
        fig = plt.figure(constrained_layout=True, figsize=(6, 8), dpi=400)
        axd = fig.subplot_mosaic(
            """
            L
            A
            A
            A
            B
            B
            B
            C
            C
            C
            """
        )

        size_to_axes = {
            "small": "A",
            "medium": "B",
            "large": "C",
        }
        size_to_title = {
            "small": "Problema pequeño",
            "medium": "Problema mediano",
            "large": "Problema grande",
        }

        fig.supxlabel("generaciones")
        fig.supylabel("hv")

        for name, size, exp in self.experiments.exp_iter():
            if not exp.is_multi_gen_exp():
                # We want to skip Weighted sum, since plotting HV per gen doesn't make sense
                continue
            ref_point = self.experiments.compute_reference_point(size)
            print(size, ref_point)
            x_axis = [i for i in range(1, exp.get_generations()+1)]
            res = exp.get_median_result(ref_point)
            y_axis = res.compute_hv_per_generation(ref_point)

            ax = axd[size_to_axes[size]]
            ax.set_title(size_to_title[size])
            ax.plot(x_axis, y_axis, label = self.names_to_title[name])

        axd['L'].axis('off')
        handles, labels = axd['A'].get_legend_handles_labels()
        axd['L'].legend(handles, labels, loc="upper center", ncol=4, mode="expand")


        plt.savefig(os.path.join(self.path, f'{self.file_prefix}median_evo_graph.pgf'))
        plt.savefig(os.path.join(self.path, f'{self.file_prefix}median_evo_graph.png'))
        plt.close(fig)

    def plot_median_evo_mb(self, filter_func, prefix=''):
        fig = plt.figure(constrained_layout=True, figsize=(6, 8), dpi=400)
        axd = fig.subplot_mosaic(
            """
            L
            A
            A
            A
            B
            B
            B
            C
            C
            C
            D
            D
            D
            """
        )

        algo_to_axes = {
            "NSGA2": "A",
            "NSGA3": "B",
            "MOEAD": "C",
            "SMSEMOA": "D",
        }

        algo_to_title = {
            "NSGA2": "NSGA-II",
            "NSGA3": "NSGA-III",
            "MOEAD": "MOEA/D",
            "SMSEMOA": "SMS-EMOA",
        }

        fig.supxlabel("generaciones")
        fig.supylabel("hv")

        for algo, exps in self.experiments.group_by(tag_grouping(Tag.ALGORITHM)):
            for exp in filter(filter_func, exps):
                if not exp.is_multi_gen_exp():
                    # We want to skip Weighted sum, since plotting HV per gen doesn't make sense
                    continue
                size = exp.get_problem_size()
                ref_point = self.experiments.compute_reference_point(size)
                x_axis = [i for i in range(1, exp.get_generations()+1)]
                res = exp.get_median_result(ref_point)
                y_axis = res.compute_hv_per_generation(ref_point)

                ax = axd[algo_to_axes[exp.get_tag(Tag.ALGORITHM)]]
                ax.set_title(algo_to_title[algo])
                ax.plot(x_axis, y_axis, label = exp.get_tag(Tag.BATCH_SIZE))
                ax.legend(ncol=3)

        axd['L'].axis('off')
        handles, labels = axd['A'].get_legend_handles_labels()
        axd['L'].legend(handles, labels, loc="upper center", ncol=4, mode="expand")


        plt.savefig(os.path.join(self.path, f'{self.file_prefix}{prefix}median_evo_graph.pgf'))
        plt.savefig(os.path.join(self.path, f'{self.file_prefix}{prefix}median_evo_graph.png'))
        plt.close(fig)


    def plot_pareto_front(self, size, layout, names_to_axes, figsize):
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        axd = fig.subplot_mosaic(layout, sharex=True, sharey=True)

        ref_point = self.experiments.compute_reference_point(size)

        for ax in axd.values():
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.yaxis.set_tick_params(labelleft=True)

        for name, exp in self.experiments.size_iter(size):
            res = exp.get_median_result(ref_point)
            F = res.get_evaluation()

            axd[names_to_axes[name]].scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
            axd[names_to_axes[name]].set_title(self.names_to_title[name])

        fig.savefig(os.path.join(self.path, f'{self.file_prefix}{size}_pareto_front.png'))
        fig.savefig(os.path.join(self.path, f'{self.file_prefix}{size}_pareto_front.pgf'))
        plt.close(fig)


