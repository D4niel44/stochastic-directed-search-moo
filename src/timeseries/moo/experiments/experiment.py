from abc import ABC, abstractmethod
from functools import cache
from typing import List, Any, Tuple

import numpy as np

from src.timeseries.utils.moo import get_hypervolume
from src.timeseries.moo.experiments.moea_result import MoeaResult


class ExperimentResult(ABC):
    @abstractmethod
    def get_time(self) -> float:
        pass

    # Should return a numpy array.
    @abstractmethod
    def get_evaluation(self) -> Any:
        pass

    @abstractmethod
    def get_evaluation_per_generation(self) -> Any:
        pass

    @abstractmethod
    def get_generations(self) -> int:
        pass

    @cache
    def compute_hv_per_generation(self, ref_point):
        return [get_hypervolume(eval, ref_point) for eval in self.get_evaluation_per_generation()]

    @cache
    def compute_hv(self, ref_point) -> float:
        return get_hypervolume(self.get_evaluation(), ref_point)

    def get_max_per_f(self):
        F = self.get_evaluation()
        return [max(F[:, i]) for i in range(F.shape[1])]

class MoeaExperimentResult(ExperimentResult):
    def __init__(self, moea_result: MoeaResult):
        self._res = moea_result

    def get_time(self) -> float:
        # Get last generation time, as a list
        return self._res.times[-1]

    def get_evaluation(self):
        return self._res.metrics[-1]['F']

    def get_evaluation_per_generation(self) -> Any:
        return [m['F'] for m in self._res.metrics]

    def get_generations(self) -> int:
        return len(self._res.metrics)

class WeightedSumExperimentResult(ExperimentResult):
    def __init__(self, serialized_res):
        self._res = serialized_res

        eval = []
        times = []
        for res in serialized_res:
            qcr = res['val_quantile_coverage_risk']
            qer = res['val_quantile_estimation_risk']
            eval.append([qcr[-1], qer[-1]])
            times.append(res['time'])

        self._eval = np.array(eval)
        self._times = np.array(eval)

    def get_time(self) -> float:
        return sum(self._times)

    def get_evaluation(self):
        return self._eval

    # Weighted sum is not a multi gen algorithm.
    def get_generations(self) -> int:
        return 1

    def get_evaluation_per_generation(self) -> Any:
        return [self.get_evaluation]

# Multiple executions of a single algo & size
class Experiment:
    def __init__(self, algo_name: str, size: str, results: List[ExperimentResult]):
        self._name = algo_name
        self._size = size
        self._res = results

        self._gens = self._get_and_validate_generations()

    def _get_and_validate_generations(self) -> int:
        generations = self._res[0].get_generations()
        for res in self._res:
            if res.get_generations() != generations:
                raise ValueError("Two results in same experiment have different number of generations")

        return generations

    def get_generations(self) -> int:
        return self._gens

    def is_multi_gen_exp(self) -> bool:
        return self.get_generations() > 1

    def get_name(self) -> str:
        return self._name

    def get_problem_size(self) -> str:
        return self._size

    def get_mean_std_time(self) -> Tuple[float, float]:
        concatenated_times = []
        for res in self._res:
            concatenated_times.append(res.get_time())

        return np.mean(concatenated_times), np.std(concatenated_times)

    def get_mean_std_hv(self, ref_point: float) -> Tuple[float, float]:
        hvs = [res.compute_hv(ref_point) for res in self._res]
        return np.mean(hvs), np.std(hvs)

    def get_median_result(self, ref_point: float) -> ExperimentResult:
        sorted_results = sorted(self._res, key=lambda r: r.compute_hv(ref_point))
        return sorted_results[len(sorted_results)//2]

    def get_result(self, idx) -> ExperimentResult:
        return self._res[idx]

    def __len__(self):
        return len(self._res)

    def __iter__(self):
        for r in self._res:
            yield r

    def get_max_per_f(self):
        max_f = self._res[0].get_max_per_f()
        for r in self._res[1:]:
            for i, r_max_f in enumerate(r.get_max_per_f()):
                max_f[i] = max(max_f[i], r_max_f)
        return max_f


# Holds experiments of multiple algos and sizes
class CompositeExperiment:

    # Do not instantiate directly, use the Builder below instead
    def __init__(self, m, n_obj, gen_map):
        self._map = m
        self._n_obj = n_obj
        self._gen_map = gen_map

    def n_gen(self, size):
        return self._n_gen[size]

    # Gets the problem sizes.
    def get_problem_sizes(self):
        problem_sizes = set()
        for _, size, _ in self.exp_iter():
            problem_sizes.add(size)

        return problem_sizes


    # Iterates over each expriments. Yields (name, size, exp) tuples.
    def exp_iter(self):
        for name, size_map in self._map.items():
            for size, exp_list in size_map.items():
                for  exp in exp_list:
                    yield (name, size, exp)

    # Iterates over experiments of a given size. Yields (name, exp) tuples.
    def size_iter(self, size: str):
        for name, size_map in self._map.items():
            for exp in size_map[size]:
                yield (name, exp)

    # Iterates over experiments with the same name. Yields (size, exp) tuples.
    def name_iter(self, name: str):
        for size, exp_list in self._map[name].items():
            for exp in exp_list:
                yield (size, exp)

    # Computes the ref point for experiments of a given size.
    @cache
    def compute_reference_point(self, size: str, scale=1.1):
        ref_point = [0 for _ in range(self._n_obj)]

        for _, exp in self.size_iter(size):
            exp_max_f = exp.get_max_per_f()
            for i, v in enumerate(ref_point):
                ref_point[i] = max(v, exp_max_f[i])

        # Return scaled
        return tuple([v * scale for v in ref_point])

class CompositeExperimentBuilder:
    def __init__(self):
        self._map = {}
        self._n_obj = 0
        self._gen_map = {}

    def set_number_objectives(self, n_obj):
        self._n_obj = n_obj

        return self

    def set_number_generations(self, size, n_gen):
        self._gen_map[size] = n_gen

        return self

    def add_experiment(self, exp: Experiment):
        name = exp.get_name()
        size = exp.get_problem_size()
        if name not in self._map:
            self._map[name] = {}

        if size not in self._map[name]:
            self._map[name][size] = []

        self._map[name][size].append(exp)

        return self

    def build(self) -> CompositeExperiment:
        return CompositeExperiment(self._map, self._n_obj, self._gen_map)







