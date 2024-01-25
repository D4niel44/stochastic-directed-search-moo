import argparse
import os
import yaml


from timeseries.moo.experiments.config import moea_map, supported_moeas, supported_problem_sizes 

# %%
if __name__ == '__main__':
    ## --------------- CFG ---------------
    parser = argparse.ArgumentParser(description='Create experiment folder and config.')
    parser.add_argument('--moea', metavar='MOEA', nargs='*',
                        choices=[m for m in moea_map.keys()],
                        default=[m for m in moea_map.keys()],
                        dest='moeas',
                        help=f'MOEAs to use in the experiment (supported moeas are: {[m.__name__ for m in supported_moeas]})',)
    parser.add_argument('-g', '--generations', type=int, dest='generations', required=True, help='number of generations (required)')
    parser.add_argument('-s', '--problem_size', dest='problem_size', required=True, 
                        choices=supported_problem_sizes,
                        help=f'size of the problem (required, available sizes are: {supported_problem_sizes})')
    parser.add_argument('-p', '--pop_size', type=int, dest='pop_size', required=True, help='size of the population (required)')
    parser.add_argument('-r', '--num_runs', type=int, dest='num_runs', required=True, help='number of runs of the experiment (required)')
    parser.add_argument('path', help='path to store the experiment folder')

    args = parser.parse_args()

    p = os.path.join(args.path, f'exp_{args.problem_size}_p{args.pop_size}_g{args.generations}_r{args.num_runs}')
    os.makedirs(p)

    config = {
        'moeas': args.moeas,
        'problem_size': args.problem_size,
        'population_size': args.pop_size,
        'generations': args.generations,
        'number_runs': args.num_runs,
    }

    with open(os.path.join(p, 'config.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

