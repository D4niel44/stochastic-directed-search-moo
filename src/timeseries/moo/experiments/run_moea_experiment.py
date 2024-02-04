# %%
import argparse
import os
import time

import yaml
import joblib
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3 
from pymoo.algorithms.moo.moead import MOEAD 
from pymoo.algorithms.moo.sms import SMSEMOA 
from pymoo.core.callback import Callback

from src.timeseries.moo.sds.config import sds_cfg
from src.timeseries.moo.core.harness import get_model_and_params, get_ts_problem
from src.timeseries.moo.sds.utils.bash import get_input_args
from src.timeseries.moo.sds.utils.indicators import metrics_of_pf
from src.timeseries.moo.sds.utils.util import get_from_dict
from src.timeseries.utils.moo import sort_1st_col
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from src.timeseries.moo.experiments.config import moea_map

class MetricsCallback(Callback):

    def __init__(self, problem, t0, ref_point, skip_train_metrics=False) -> None:
        super().__init__()
        self.problem = problem
        self.t0 = t0
        self.data["metrics"] = []
        self.data["times"] = []
        self.data["train_metrics"] = []
        self.ref_point = ref_point
        self.skip_train_metrics = skip_train_metrics

    def notify(self, algorithm, **kwargs):
        X = algorithm.pop.get("X")
        F = problem.eval_individuals(X, 'valid')
        X_moea_sorted, F_moea_sorted = sort_1st_col(X, F)
        moea_metrics = metrics_of_pf(F_moea_sorted, ref=self.ref_point)
        self.data["metrics"].append(moea_metrics)
        self.data["times"].append(time.time() - t0)

        if not skip_train_metrics:
            train_metrics = metrics_of_pf(algorithm.pop.get("F"), ref=self.ref_point)
            self.data["train_metrics"].append(train_metrics)

def optimal_reference_point(problem_size):
    if problem_size == 'small':
        return [6.604550218582154, 1.7242103934288027]
    if problem_size == 'medium':
        return [78.61612014770509, 16.727188873291016]
    return [2., 2.]

# %%
if __name__ == '__main__':
    ## --------------- CFG ---------------
    parser = argparse.ArgumentParser(description='Run experiment.')
    parser.add_argument('--start', type=int, dest='start_seed', default=0, help='start running the experiment at the given seed')
    parser.add_argument('--end', type=int, dest='end_seed', default=None, help='end running the experiment before the given seed (exclusive)')
    parser.add_argument('path', help='path to store the experiment folder')

    args = parser.parse_args()

    config = None
    with open(os.path.join(args.path, "config.yaml"), 'r') as stream:
        config = yaml.safe_load(stream)

    print(config)
    project = 'snp'
    problem_size = config['problem_size']
    moeas = [moea_map[m] for m in config['moeas']]
    n_repeat = args.end_seed if args.end_seed is not None else config['number_runs']
    pop_size = config['population_size']
    n_gen = config['generations']
    skip_train_metrics = config['skip_train_metrics']
    ## ------------------------------------

    sds_cfg['model']['ix'] = get_input_args()['model_ix']
    sds_cfg['model']['ix'] = 5
    sds_cfg['problem']['split_model'] = problem_size
    sds_cfg['problem']['limits'] = None
    sds_cfg['sds']['max_increment'] = None if sds_cfg['problem']['split_model'] == 'medium' else 0.05
    sds_cfg['sds']['step_size'] = 2.5e-2 if sds_cfg['problem']['split_model'] == 'medium' else 5e-3

    print('Model ix: {}'.format(get_from_dict(sds_cfg, ['model', 'ix'])))

    model_params, results_folder = get_model_and_params(sds_cfg, project)
    problem = get_ts_problem(sds_cfg, model_params, test_ss=False)


    # %% Solve N times with MOEA (take measurements along generations)
    for seed in range(args.start_seed, n_repeat):
        for moea in moeas:
            t0 = time.time()
            if moea.__name__ == "MOEAD":
                problem.n_ieq_constr = 0
                problem.constraints_limits = None
                algorithm = moea(
                    n_offsprings=pop_size,
                    ref_dirs=get_reference_directions("energy", problem.n_obj, pop_size),
                    sampling=FloatRandomSampling(),
                    crossover=SBX(prob=0.9, eta=15),
                    mutation=PolynomialMutation(eta=20),
                )
            else:
                algorithm = moea(
                    pop_size=pop_size,
                    n_offsprings=pop_size,
                    ref_dirs=get_reference_directions("energy", problem.n_obj, pop_size),
                    sampling=FloatRandomSampling(),
                    crossover=SBX(prob=0.9, eta=15),
                    mutation=PolynomialMutation(eta=20),
                    eliminate_duplicates=True
                )

            termination = get_termination("n_gen", n_gen)
            res = minimize(problem,
                           algorithm,
                           termination,
                           seed=seed,
                           save_history=False,
                           verbose=True,
                           callback=MetricsCallback(problem, t0, optimal_reference_point(problem_size)))

            if skip_train_metrics:
                joblib.dump(
                 (res.opt, res.algorithm.callback.data["times"], res.algorithm.callback.data["metrics"]),
                 os.path.join(args.path, f'{moea.__name__}_{sds_cfg["problem"]["split_model"]}_results_{seed}.z'),
                 compress=9)
            else:
                joblib.dump(
                 (res.opt, res.algorithm.callback.data["times"], res.algorithm.callback.data["metrics"], res.algorithm.callback.data["train_metrics"]),
                 os.path.join(args.path, f'{moea.__name__}_{sds_cfg["problem"]["split_model"]}_results_{seed}.z'),
                 compress=9)
        