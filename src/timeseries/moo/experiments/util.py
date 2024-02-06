import os
from itertools import chain
from functools import partial
from multiprocessing import Pool, RLock
import sys

import joblib
from tqdm import tqdm
import time

def _parallel_load(path, n_repeat, problem_size, m):
    (i, moea) = m
    results = []
    print_warning = False
    for seed in tqdm(range(n_repeat), desc=f'Loading {moea.__name__} results', leave=True, colour='green', position=i, lock_args=None, file=sys.stdout):
        res = joblib.load(os.path.join(path, f'{moea.__name__}_{problem_size}_results_{seed}.z'))
        if len(res) == 3: #results without training metrics
            # (sol, times, metrics) - sol is discarded
            results.append(res[1:])
        elif len(res) == 4: #results with training metrics
            # (sol, times, metrics, train_metrics) - sol and train_metrics are discarded
            results.append(res[1:3])
        else:
            raise AttributeError("unsuported result length")
        if 'F' not in res[2][-1]:
            # For backwards compatibility with older experiments that didn't save F per generation.
            # Add the last generation F to the metrics, and warn that this result does not support F per generation.
            # TODO: Evaluate X on validation dataset in order to get F.
            print_warning = True
            res[2][-1]['F'] = res[0].get('F')
    return moea, results, print_warning
 
def load_results(path, moeas, n_repeat, problem_size):
    res_dict = {}
    times_dict = {}
    print_warning = False
    tqdm.set_lock(RLock())
    p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    tasks = p.map(partial(_parallel_load, path, n_repeat, problem_size), enumerate(moeas))
    for moea, results, warning in tasks:
        times_dict[moea] = [r[0] for r in results]
        res_dict[moea] = [r[1] for r in results]
        print_warning = print_warning | warning
    if print_warning:
        print(f'WARNING: The results in path {path} do not support validation F per generation')
    return times_dict, res_dict

def get_reference_point(res_dict):
    max_f1 = 0
    max_f2 = 0
    for res in chain.from_iterable(res_dict.values()):
        F = res[-1]['F']
        max_f1 = max(max_f1, max(F[:, 0]))
        max_f2 = max(max_f2, max(F[:, 1]))
    return max_f1 * 1.1, max_f2 * 1.1


def parse_reference_point_arg(str):
    return [float(n) for n in str.split(',')]
