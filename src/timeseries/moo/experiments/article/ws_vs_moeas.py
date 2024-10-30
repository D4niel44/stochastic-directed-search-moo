import sys
import argparse
import os
from itertools import chain
import yaml

from src.timeseries.moo.experiments.config import moea_map
from src.timeseries.moo.experiments.util import load_moea_results_to_exp, load_ws_results_to_exp
from src.timeseries.moo.experiments.experiment import CompositeExperimentBuilder
from src.timeseries.moo.experiments.article.visualization import ResultVisualization

NAMES_TO_TITLE = {
        "NSGA2": "NSGA-II",
        "NSGA3": "NSGA-III",
        "MOEAD": "MOEA/D",
        "SMSEMOA": "SMS-EMOA",
        "WS": "Suma ponderada",
}

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
    vis.plot_median_evo()

    for size in experiments.get_problem_sizes():
        vis.plot_pareto_front(size)

