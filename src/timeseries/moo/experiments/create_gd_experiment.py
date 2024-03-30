import argparse
import os
import yaml


from src.timeseries.moo.experiments.config import moea_map, supported_moeas, supported_problem_sizes 

def generate_uniform_combinations(n):
    # Include 0 and 1 in combinations
    return [i/n for i in range(n + 1)]
# %%
if __name__ == '__main__':
    ## --------------- CFG ---------------
    parser = argparse.ArgumentParser(description='Create experiment folder and config.')
    parser.add_argument('-s', '--problem_size', dest='problem_size', required=True, 
                        choices=supported_problem_sizes,
                        help=f'size of the problem (required, available sizes are: {supported_problem_sizes})')
    parser.add_argument('--initial_weight_moea', metavar='MOEA',
                        choices=[m for m in moea_map.keys()],
                        default=['SMSEMOA'],
                        dest='moea',
                        help=f'MOEA to use to init the experiment weights (supported moeas are: {[m.__name__ for m in supported_moeas]})',)
    parser.add_argument('--initial_weight_path', default=None, help='path to load the moea from') 
    parser.add_argument('-n', '--number_combinations', choices=[5, 10, 20, 50, 99, 100], type=int, dest='number_combinations', required=True, help='number of different combination of weigth for the sum of QCR and QER')
    parser.add_argument('--epochs', type=int, dest='number_epochs', default=5, help='number of epochs to train')
    parser.add_argument('--skip_train_metrics', action='store_true', help="don't store metrics each generation using the train dataset (only the validation dataset will be used)")
    parser.add_argument('--normalize_loss', action='store_true', help="use normalized loss during training")
    parser.add_argument('path', help='path to store the experiment folder')

    args = parser.parse_args()

    p = os.path.join(args.path, f'exp_{args.problem_size}_n{args.number_combinations}_norm{args.normalize_loss}_e{args.number_epochs}{"_iw" if args.initial_weight_path is not None else ""}')
    os.makedirs(p)

    config = {
        'algorithm': 'GD',
        'problem_size': args.problem_size,
        'combinations': generate_uniform_combinations(args.number_combinations),
        'epochs': args.number_epochs,
        'skip_train_metrics': args.skip_train_metrics,
        'normalize_loss': args.normalize_loss,
    }
    if args.initial_weight_path is not None:
        config['initial_weight'] = {
            'path': args.initial_weight_path,
            'moea': args.moea,
        }

    with open(os.path.join(p, 'config.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

