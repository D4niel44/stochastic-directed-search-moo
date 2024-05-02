import os
from itertools import chain
from functools import partial
from multiprocessing import Pool, RLock
import sys

import joblib
from tqdm import tqdm
import yaml

from src.timeseries.utils.moo import get_hypervolume

def _parallel_load(path, n_repeat, problem_size, m):
    (i, moea) = m
    results = []
    print_warning = False
    for seed in tqdm(range(n_repeat), desc=f'Loading {moea.__name__} results', leave=True, colour='green', position=i, lock_args=None, file=sys.stdout):
        res = load_result_file(path, moea, problem_size, seed)
        # Old results are saved in tuple
        if type(res) is tuple:
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
        else:
            # clear res.opt to save memory
            res.opt = None
            results.append(res)
    return moea, results, print_warning

def load_result_file(path, moea, problem_size, seed):
    return joblib.load(os.path.join(path, f'{moea.__name__}_{problem_size}_results_{seed}.z'))
 
def load_results(path, moeas, n_repeat, problem_size, parallelize = True):
    res_dict = {}
    times_dict = {}
    print_warning = False
    if parallelize:
        tqdm.set_lock(RLock())
        p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        tasks = p.map(partial(_parallel_load, path, n_repeat, problem_size), enumerate(moeas))
    else:
        tasks = map(partial(_parallel_load, path, n_repeat, problem_size), enumerate(moeas))
    for moea, results, warning in tasks:
        if type(results[0]) is tuple:
            times_dict[moea] = [r[0] for r in results]
            res_dict[moea] = [r[1] for r in results]
        else:
            times_dict[moea] = [r.times for r in results]
            res_dict[moea] = [r.metrics for r in results]
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


def load_config(path):
    with open(os.path.join(path, "config.yaml"), 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def get_best_seed_moea(moea_results, ref_point):
    best_idx = 0
    best_hv = 0
    for i, r in enumerate(moea_results):
        hv = get_hypervolume(r[-1]['F'], ref_point)
        if hv > best_hv:
            best_idx = i
            hv = best_hv
    return best_idx

def get_best_f_moea(moea_results, ref_point):
    best_idx = get_best_seed_moea(moea_results, ref_point)
    return moea_results[best_idx][-1]['F']


def nonlinear_weights_selection(lambdas, k, n):
    return [(l / k) ** n for l in lambdas]
