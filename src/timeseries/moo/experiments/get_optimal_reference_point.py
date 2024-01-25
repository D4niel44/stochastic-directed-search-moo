# %%
import os
import time

import joblib
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3 
from pymoo.algorithms.moo.moead import MOEAD 
from pymoo.algorithms.moo.sms import SMSEMOA 
from pymoo.core.callback import Callback
from tabulate import tabulate

from src.models.compare.winners import wilcoxon_significance
from src.timeseries.moo.core.harness import get_model_and_params, get_ts_problem, get_continuation_method, \
    run_cont_problem
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
import numpy as np
import matplotlib.pyplot as plt

from src.timeseries.utils.util import write_text_file, latex_table
from src.timeseries.utils.critical_difference import draw_cd_diagram


if __name__ == '__main__':
    ## --------------- CFG ---------------
    problem_size = 'small'
    moeas = [NSGA2, NSGA3, MOEAD, SMSEMOA]
    path = 'output_small_new/'
    n_repeat = 20
    pop_size = 200
    n_gen = 250

    res_dict = {}
    times_dict = {}
    sol_dict = {}
    for i, moea in enumerate(moeas):
        results = [joblib.load(f'{path}{moea.__name__}_{problem_size}_results_{seed}.z') for seed in range(n_repeat)]
        sol_dict[moea] = [r[0] for r in results]
        times_dict[moea] = [r[1] for r in results]
        res_dict[moea] = [r[2] for r in results]

    max_f1 = 0
    max_f2 = 0
    for seed in range(n_repeat):
        for i, moea in enumerate(moeas):
            sol = sol_dict[moea]
            F = sol[seed].get("F")
            max_f1 = max(max_f1, max(F[:, 0]))
            max_f2 = max(max_f2, max(F[:, 1]))

    print(f'Optimal reference point [{max_f1 * 1.1}, {max_f2 * 1.1}]')