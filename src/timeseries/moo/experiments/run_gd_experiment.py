# %%
import argparse
import os
import time
import copy
import tensorflow as tf

import yaml
import joblib
from pymoo.core.callback import Callback

from src.timeseries.moo.sds.config import sds_cfg
from src.timeseries.moo.core.harness import get_model_and_params
from src.timeseries.moo.sds.utils.bash import get_input_args
from src.timeseries.moo.sds.utils.indicators import metrics_of_pf
from src.timeseries.moo.sds.utils.util import get_from_dict
from src.timeseries.utils.moo import sort_1st_col

from src.models.attn.nn_funcs import QuantileLossCalculator

from src.sds.nn.utils import reconstruct_weights, params_conversion_weights, batch_from_list_or_array, \
    predict_from_batches, get_one_output_model, split_model

def batch_base_inputs(x_train, x_valid, base_batch_size):
        x_train_batches = batch_from_list_or_array(x_train, batch_size=base_batch_size)
        x_valid_batches = batch_from_list_or_array(x_valid, batch_size=base_batch_size)
        return x_train_batches, x_valid_batches

def get_quantile_loss_calculator_from_model(model):
    quantiles = copy.copy(model.quantiles)
    output_size = copy.copy(model.output_size)

    quantile_loss_calculator = QuantileLossCalculator(quantiles, output_size)
    return quantile_loss_calculator


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

def get_intermediate_layers(problem_size):
    if problem_size == 'medium':
        intermediate_layers = ['layer_normalization_36', 'time_distributed_144']
    elif problem_size == 'small':
        intermediate_layers = ['layer_normalization_40']
    elif problem_size == 'large':
        intermediate_layers = ['layer_normalization_4', 'layer_normalization_5', 'tf.math.reduce_sum_1', 'tf.math.reduce_sum_2', 'time_distributed_144']
    else:
        raise NotImplementedError
    return intermediate_layers

def loss_function(loss_func, weight, quantile_ix):
    def loss(y, y_pred):
        loss_per_quantile = loss_func(y, y_pred)
        loss_per_func = loss_per_quantile[quantile_ix]
        ret = weight * loss_per_func[0] + (1 - weight) * loss_per_func[1]
        return ret
    return loss

def qcr(quantile_loss_calculator, quantile_ix):
    def quantile_coverage_risk(y, y_pred):
        loss_per_quantile = quantile_loss_calculator.quantile_loss_per_q_moo(y, y_pred)
        loss_per_func = loss_per_quantile[quantile_ix]
        return loss_per_func[0]
    return quantile_coverage_risk

def qer(quantile_loss_calculator, quantile_ix):
    def quantile_estimation_risk(y, y_pred):
        loss_per_quantile = quantile_loss_calculator.quantile_loss_per_q_moo(y, y_pred)
        loss_per_func = loss_per_quantile[quantile_ix]
        return loss_per_func[1]
    return quantile_estimation_risk


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
    combinations = config['combinations']
    skip_train_metrics = config['skip_train_metrics']
    normalize_loss = config['normalize_loss']
    epochs = config['epochs']
    ## ------------------------------------

    sds_cfg['model']['ix'] = get_input_args()['model_ix']
    sds_cfg['model']['ix'] = 5
    sds_cfg['problem']['split_model'] = problem_size
    sds_cfg['problem']['limits'] = None
    sds_cfg['sds']['max_increment'] = None if sds_cfg['problem']['split_model'] == 'medium' else 0.05
    sds_cfg['sds']['step_size'] = 2.5e-2 if sds_cfg['problem']['split_model'] == 'medium' else 5e-3

    print('Model ix: {}'.format(get_from_dict(sds_cfg, ['model', 'ix'])))

    model_params, results_folder = get_model_and_params(sds_cfg, project)
    model = get_one_output_model(model_params['model'].model, 'td_quantiles')
    models = split_model(model, get_intermediate_layers(problem_size), compile=False)

    base_model = models['base_model']
    trainable_model = models['trainable_model']

    base_model.compile()
    quantile_loss_calculator = get_quantile_loss_calculator_from_model(model_params['model'])
    adam = tf.keras.optimizers.Adam(learning_rate=model_params['model'].learning_rate, clipnorm=model_params['model'].max_gradient_norm)

    initial_weights = trainable_model.get_weights()

    if normalize_loss:
        loss_f = quantile_loss_calculator.quantile_loss_per_q_moo
    else:
        loss_f = quantile_loss_calculator.quantile_loss_per_q_moo_no_normalization

    results = []
    for weight in combinations:
        print(f"Starting training with sum weight {weight}")
        trainable_model.set_weights(initial_weights)
        loss = loss_function(loss_f, weight, sds_cfg['problem']['quantile_ix'])
        trainable_model.compile(
            loss=loss,
            metrics=[
                qcr(quantile_loss_calculator, sds_cfg['problem']['quantile_ix']),
                qer(quantile_loss_calculator, sds_cfg['problem']['quantile_ix']),
            ],
            optimizer=adam, 
        )

        with tf.device('/device:GPU:0'):
            x_base_train, x_base_valid = batch_base_inputs(
                model_params['datasets']['train']['x'],
                model_params['datasets']['valid']['x'],
                sds_cfg['problem']['base_batch_size']
            )
            x_train = predict_from_batches(base_model, x_base_train,
                                            to_numpy=False,
                                            concat_output=True,
                                            use_gpu=True)
            x_valid = predict_from_batches(base_model, x_base_valid,
                                            to_numpy=False,
                                            concat_output=True,
                                            use_gpu=True)
            #trainable_model.fit()
            res = trainable_model.fit(
                x=x_train,
                y=model_params['datasets']['train']['y'],
                batch_size=sds_cfg['problem']['moo_batch_size'], 
                epochs=epochs,
                verbose=2,
                validation_data=(x_valid, model_params['datasets']['valid']['y']),
            )

            results.append(res.history)
            trainable_model.save(os.path.join(args.path, f'model_w{weight}.keras'))

    joblib.dump(
        results,
        os.path.join(args.path, f'results.z'),
        compress=3)
        