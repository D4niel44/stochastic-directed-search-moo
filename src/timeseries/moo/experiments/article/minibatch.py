import argparse
import os
from itertools import chain
import yaml
import pandas as pd
import numpy as np

from src.timeseries.moo.experiments.config import moea_map
from src.timeseries.moo.experiments.util import load_moea_results_to_exp
from src.timeseries.moo.experiments.experiment import CompositeExperimentBuilder, size_filter, tag_grouping
from src.timeseries.moo.experiments.article.visualization import ResultVisualization
from src.timeseries.moo.experiments.experiment import CompositeExperimentBuilder, Tag

BATCH_SIZE_TO_TITLE = {
    0: 'Sin minibatch',
    512: '512',
    1024: '1024',
}

NAMES_TO_TITLE = {
        'NSGA2': 'NSGA-II',
        'NSGA2 (512)': 'NSGA-II (512)',
        'NSGA2 (1024)': 'NSGA-II (1024)',

        'NSGA3': 'NSGA-III',
        'NSGA3 (512)': 'NSGA-III (512)',
        'NSGA3 (1024)': 'NSGA-III (1024)',

        'MOEAD': 'MOEA/D',
        'MOEAD (512)': 'MOEA/D (512)',
        'MOEAD (1024)': 'MOEA/D (1024)',

        'SMSEMOA': 'SMS-EMOA',
        'SMSEMOA (512)': 'SMS-EMOA (512)',
        'SMSEMOA (1024)': 'SMS-EMOA (1024)',
}

LAYOUT = """
    ABC
    DEF
    GHI
    JKL
"""


NAMES_TO_AXES = {
    'NSGA2': 'A',
    'NSGA2 (512)': 'B',
    'NSGA2 (1024)': 'C',

    'NSGA3': 'D',
    'NSGA3 (512)': 'E',
    'NSGA3 (1024)': 'F',

    'MOEAD': 'G',
    'MOEAD (512)': 'H',
    'MOEAD (1024)': 'I',

    'SMSEMOA': 'J',
    'SMSEMOA (512)': 'K',
    'SMSEMOA (1024)': 'L',
}

def plot_cd_diagram(batch_size_dict, path, recalculate_hv=False, file_prefix=''):
    run_res = []

    for batch_size, size_dict in batch_size_dict.items():
        i = 0
        for size, dict in size_dict.items():
            res_dict = dict['res_dict']
            ref = dict['ref_point']
            n_gen = dict['n_gen']
            for moea in moeas:
                moea_results = res_dict[moea]

                for res in moea_results:
                    run_res.append({
                        'classifier_name': batch_size,
                        'dataset_name': i,
                        'hv': get_hypervolume(res[n_gen - 1]['F'], ref) if recalculate_hv else res[n_gen - 1]['hv'],
                    })
                    i += 1

    runs_df = pd.DataFrame.from_records(run_res)
    print(runs_df)
    runs_df.to_csv(f'{path}seed_results.csv')
    draw_cd_diagram(df_perf=runs_df, key_name='hv', labels=True, path=path, filename=f'{file_prefix}cd_diagram.png')

def mean_std_table(batch_size_dict, path, recalculate_hv=False, file_prefix=''):
    gens_res = []

    

    for batch_size, problem_size_dict in batch_size_dict.items():
        for problem_size, dict in problem_size_dict.items():
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
                'problem size': problem_size,
                'batch size': batch_size
            }
            for moea in moeas:
                times, moea_results = times_dict[moea], res_dict[moea]
                ts = [t[c] for t in times]
                row[moea.__name__] = '{:.2f} ({:.2f})'.format(np.mean(ts), np.std(ts))
            gens_res.append(row)

            row = {
                'metric': 'hv',
                'problem size': problem_size,
                'batch size': batch_size
            }
            for moea in moeas:
                moea_results = res_dict[moea]
                hvs = [get_hypervolume(r[c]['F'], ref) for r in moea_results] if recalculate_hv else [r[c]['hv'] for r in moea_results]
                row[moea.__name__] = '{:.4f} ({:.2E})'.format(np.mean(hvs), np.std(hvs))
            gens_res.append(row)

            row = {
                'metric': 'relative hv',
                'problem size': problem_size,
                'batch size': batch_size
            }
            for moea in moeas:
                moea_results = res_dict[moea]
                hvs = [get_hypervolume(r[c]['F'], ref) for r in moea_results] if recalculate_hv else [r[c]['hv'] for r in moea_results]
                row[moea.__name__] = '{:.4f}'.format(1 - ((max_hv - hv_map[moea]) / max_hv))
            gens_res.append(row)

    gens_res_df = pd.DataFrame.from_records(gens_res).sort_values(['metric', 'problem size', 'batch size'], ascending=False)
    print(gens_res_df)
    title = f'{moea.__name__} results'
    write_text_file(os.path.join(path, f'{file_prefix}{problem_size}_results_table'),
                    latex_table(title, gens_res_df.to_latex(escape=False, index=False)))

# %%
if __name__ == '__main__':
    ## --------------- CFG ---------------
    parser = argparse.ArgumentParser(description='Analyze results of experiment.')
    parser.add_argument('--end', type=int, dest='end_seed', default=None, help='end running the experiment before the given seed (exclusive)')
    parser.add_argument('--use_default_hv', action='store_true', dest='default_hv', help='use the default HV instead of recalculating it')
    parser.add_argument('exp_path', help='path of the experiment config')
    parser.add_argument('--file_prefix', default='', help='prefix to add to file names')
    parser.add_argument('--subfolder_name', default='results', help='name of the subfolder that will contain the output of this script (defaults to results)')

    args = parser.parse_args()

    exp_config = None
    with open(os.path.join(args.exp_path, "config.yaml"), 'r') as stream:
        exp_config = yaml.safe_load(stream)

    print(exp_config)

    exp_builder = CompositeExperimentBuilder()
    exp_builder.set_number_objectives(2)


    for batch_size, cfg in chain.from_iterable(c.items() for c in exp_config['batch_sizes']):
        for size, path in chain.from_iterable(c.items() for c in cfg['moeas']):

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
                exp.set_tag(Tag.ALGORITHM, exp.get_name())
                exp.set_tag(Tag.BATCH_SIZE, batch_size)

                if batch_size != 0:
                    exp._name = exp._name + f" ({batch_size})"
                exp_builder.add_experiment(exp)

    experiments = exp_builder.build()
    output_path = os.path.join(args.exp_path, args.subfolder_name)
    file_prefix = args.file_prefix

    vis = ResultVisualization(experiments, output_path, int(n_repeat), NAMES_TO_TITLE, file_prefix)
    vis.mean_std_table_mb()

    for size in experiments.get_problem_sizes():
        vis.plot_cd_diagram(
            group_key_func=tag_grouping(Tag.BATCH_SIZE),
            group_to_title=BATCH_SIZE_TO_TITLE,
            filter_func=size_filter(size),
            prefix=size+'_',
        )
        vis.plot_median_evo_mb(size_filter(size), prefix=size+'_')
        vis.plot_pareto_front(size, LAYOUT, NAMES_TO_AXES, figsize=(6,8))
