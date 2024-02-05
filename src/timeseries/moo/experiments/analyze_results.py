import argparse
import os
import yaml

from src.timeseries.moo.experiments.config import moea_map
from src.timeseries.utils.util import write_text_file
from src.timeseries.moo.experiments.utils import get_reference_point, load_results
from src.timeseries.moo.experiments.plot import mean_std_table, plot_cd_diagram, plot_evo, plot_median_evo, plot_pareto_front

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

    times_dict, res_dict = load_results(path, moeas, n_repeat, problem_size)

    ref_point = args.ref_point if args.ref_point is not None else get_reference_point(res_dict, moeas, n_repeat, n_gen)
    recalculate_hv = not args.default_hv

    os.makedirs(output_path, exist_ok=True) 

    write_text_file(os.path.join(output_path, 'reference_point'), str(ref_point))
    plot_evo(res_dict, output_path, moeas, n_repeat, ref_point, n_gen, recalculate_hv, file_prefix)
    plot_median_evo(res_dict, output_path, moeas, ref_point, n_gen, recalculate_hv, file_prefix)
    plot_pareto_front(res_dict, output_path, moeas, n_repeat, n_gen, file_prefix)
    plot_cd_diagram(res_dict, output_path, moeas, ref_point, n_gen, recalculate_hv, file_prefix)
    mean_std_table(times_dict, res_dict, output_path, moeas, ref_point, problem_size, pop_size, n_gen, recalculate_hv, file_prefix)
