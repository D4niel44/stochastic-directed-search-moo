import os
import joblib


def load_results(path, moeas, n_repeat, problem_size):
    res_dict = {}
    times_dict = {}
    for moea in moeas:
        results = []
        for seed in range(n_repeat):
            res = joblib.load(os.path.join(path, f'{moea.__name__}_{problem_size}_results_{seed}.z'))
            if len(res) == 3: #results without training metrics
                # (sol, times, metrics) - sol is discarded
                results.append(res[1:])
            elif len(res) == 4: #results with training metrics
                # (sol, times, metrics, train_metrics) - sol and train_metrics are discarded
                results.append(res[1:3])
            else:
                raise AttributeError("unsuported result length")
        times_dict[moea] = [r[0] for r in results]
        res_dict[moea] = [r[1] for r in results]
    return times_dict, res_dict

def get_reference_point(res_dict, moeas, n_repeat, n_gen):
    max_f1 = 0
    max_f2 = 0
    for seed in range(n_repeat):
        for i, moea in enumerate(moeas):
            sol = res_dict[moea]
            F = sol[seed][n_gen - 1]['F']
            max_f1 = max(max_f1, max(F[:, 0]))
            max_f2 = max(max_f2, max(F[:, 1]))
    return max_f1 * 1.1, max_f2 * 1.1
