# %%
import argparse
import os
import copy
import tensorflow as tf

import yaml
import numpy as np
import joblib

from src.timeseries.moo.sds.config import sds_cfg
from src.timeseries.moo.core.harness import get_model_and_params
from src.timeseries.moo.sds.utils.util import get_from_dict
from src.timeseries.moo.experiments.util import nonlinear_weights_selection
from src.timeseries.moo.experiments.util import load_results, load_config, get_reference_point, get_best_seed_moea, load_result_file
from src.timeseries.moo.experiments.config import moea_map
from src.timeseries.utils.moo import sort_1st_col
from src.timeseries.utils.util import write_text_file

from src.models.attn.nn_funcs import QuantileLossCalculator

from src.sds.nn.utils import batch_from_list_or_array, predict_from_batches, get_one_output_model, split_model, params_conversion_weights, reconstruct_weights

class WeightedSumFineTuning():
    
    def __init__(self, sds_cfg, project, problem_size):
        self.trainable_model_batch_size = sds_cfg['problem']['moo_batch_size']
        self.quantile_ix = sds_cfg['problem']['quantile_ix'] 

        model_params, results_folder = get_model_and_params(sds_cfg, project)
        model = get_one_output_model(model_params['model'].model, 'td_quantiles')
        models = split_model(model, get_intermediate_layers(problem_size), compile=False)

        self.base_model = models['base_model']
        self.trainable_model = models['trainable_model']
        self.base_model.compile()
        self.quantile_loss_calculator = get_quantile_loss_calculator_from_model(model_params['model'])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=model_params['model'].learning_rate, clipnorm=model_params['model'].max_gradient_norm)
        self.initial_weights = self.trainable_model.get_weights()

        with tf.device('/device:GPU:0'):
            x_base_train, x_base_valid = batch_base_inputs(
                    model_params['datasets']['train']['x'],
                    model_params['datasets']['valid']['x'],
                    sds_cfg['problem']['base_batch_size']
            )
            self.x_train = predict_from_batches(self.base_model, x_base_train,
                                                to_numpy=False,
                                                concat_output=True,
                                                use_gpu=True)
            self.x_valid = predict_from_batches(self.base_model, x_base_valid,
                                                to_numpy=False,
                                                concat_output=True,
                                                use_gpu=True)
        
        self.y_train = model_params['datasets']['train']['y']
        self.y_valid = model_params['datasets']['valid']['y']

    def get_initial_weight_params(self):
        _, params = params_conversion_weights(self.initial_weights)
        return params
    
    def train(self, weight, epochs=5, normalize_loss=None, standarize_loss=False, initial_weights=None, save_path=None, metrics=[]):
        print(f"Starting training with sum weight {weight}")

        self.trainable_model.set_weights(initial_weights if initial_weights is not None else self.initial_weights)

        if standarize_loss:
            loss_f = self.quantile_loss_calculator.quantile_loss_per_q_moo
        else:
            loss_f = self.quantile_loss_calculator.quantile_loss_per_q_moo_no_normalization
        
        if normalize_loss:
            loss = loss_function(
                loss_f,
                weight,
                self.quantile_ix,
                normalizers=[
                    normalizer(*normalize_loss[0]),
                    normalizer(*normalize_loss[1]),
                ],
            )
        else:
            loss = loss_function(loss_f, weight, self.quantile_ix)

        self.trainable_model.compile(
            loss=loss,
            metrics=[
                qcr(self.quantile_loss_calculator, self.quantile_ix),
                qer(self.quantile_loss_calculator, self.quantile_ix),
            ] + metrics,
            optimizer=self.optimizer,
        )

        with tf.device('/device:GPU:0'):
            res = self.trainable_model.fit(
                x=self.x_train,
                y=self.y_train,
                batch_size=self.trainable_model_batch_size, 
                epochs=epochs,
                verbose=2,
                validation_data=(self.x_valid, self.y_valid),
            )
        
        if save_path is not None:
            self.trainable_model.save(save_path)
        
        return res

def batch_base_inputs(x_train, x_valid, base_batch_size):
        x_train_batches = batch_from_list_or_array(x_train, batch_size=base_batch_size)
        x_valid_batches = batch_from_list_or_array(x_valid, batch_size=base_batch_size)
        return x_train_batches, x_valid_batches

def get_quantile_loss_calculator_from_model(model):
    quantiles = copy.copy(model.quantiles)
    output_size = copy.copy(model.output_size)

    quantile_loss_calculator = QuantileLossCalculator(quantiles, output_size)
    return quantile_loss_calculator


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

def normalizer(min, max):
    def func(x):
        return min_max_scale(x, min, max)
    return func

def min_max_scale(x, min, max):
    return (x - min) / (max - min)

def no_normalization(x):
    return x

def loss_function(loss_func, weight, quantile_ix, normalizers = [no_normalization, no_normalization]):
    def loss(y, y_pred):
        loss_per_quantile = loss_func(y, y_pred)
        loss_per_func = loss_per_quantile[quantile_ix]
        ret = weight * normalizers[0](loss_per_func[0]) + (1 - weight) * normalizers[1](loss_per_func[1])
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

def qcl(quantile_loss_calculator, quantile_ix):
    def quantile_coverage_loss(y, y_pred):
        loss_per_quantile = quantile_loss_calculator.quantile_loss_per_q_moo_no_normalization(y, y_pred)
        loss_per_func = loss_per_quantile[quantile_ix]
        return loss_per_func[0]
    return quantile_coverage_loss

def qel(quantile_loss_calculator, quantile_ix):
    def quantile_estimation_loss(y, y_pred):
        loss_per_quantile = quantile_loss_calculator.quantile_loss_per_q_moo_no_normalization(y, y_pred)
        loss_per_func = loss_per_quantile[quantile_ix]
        return loss_per_func[1]
    return quantile_estimation_loss

def get_initial_weights(path, moea):
    config = load_config(path)
    _, res_dict = load_results(path, [moea], config['number_runs'], config['problem_size'], parallelize=False)

    ref_point = get_reference_point(res_dict)
    best_seed = get_best_seed_moea(res_dict[moea], ref_point)

    (sol, *_) = load_result_file(path, moea, config['problem_size'], best_seed)

    X, F = sol.get('X'), sol.get('F')
    X_sorted, _ = sort_1st_col(X, F)
    return np.flip(X_sorted, 0)

def run():
    ## --------------- CFG ---------------
    parser = argparse.ArgumentParser(description='Run experiment.')
    parser.add_argument('--start', type=int, dest='start_seed', default=0, help='start running the experiment at the given seed')
    parser.add_argument('--end', type=int, dest='end_seed', default=None, help='end running the experiment before the given seed (exclusive)')
    parser.add_argument('path', help='path to store the experiment folder')

    args = parser.parse_args()

    config = None
    with open(os.path.join(args.path, "config.yaml"), 'r') as stream:
        config = yaml.safe_load(stream)

    project = 'snp'
    problem_size = config['problem_size']
    combinations = config['combinations']
    skip_train_metrics = config['skip_train_metrics']
    normalize_loss = config['normalize_loss']
    epochs = config['epochs']
    ## ------------------------------------

    sds_cfg['model']['ix'] = 5
    sds_cfg['problem']['split_model'] = problem_size
    sds_cfg['problem']['limits'] = None
    sds_cfg['sds']['max_increment'] = None if sds_cfg['problem']['split_model'] == 'medium' else 0.05
    sds_cfg['sds']['step_size'] = 2.5e-2 if sds_cfg['problem']['split_model'] == 'medium' else 5e-3

    print('Model ix: {}'.format(get_from_dict(sds_cfg, ['model', 'ix'])))

    model = WeightedSumFineTuning(sds_cfg, project, problem_size)

    if 'initial_weight' in config:
        initial_X = get_initial_weights(config['initial_weight']['path'], moea_map[config['initial_weight']['moea']])
        params = model.get_initial_weight_params()
        initial_weights = [reconstruct_weights(X, params) for i, X in enumerate(initial_X) if i % len(initial_X) // len(combinations) == 0]

    if 'nonlinear_weight_params' in config:
        k = config['nonlinear_weight_params']['k']
        n = config['nonlinear_weight_params']['n']
        weights = nonlinear_weights_selection(combinations, k, n)
    else:
        weights = combinations
    write_text_file(os.path.join(args.path, 'ws_weights'), str(weights))

    if normalize_loss:
        res0 = model.train(
            0.0,
            epochs,
            normalize_loss=None,   
            initial_weights=(initial_weights[i] if 'initial_weight' in config else None),
            save_path=os.path.join(args.path, f'model_w{0.0}.keras'),
            metrics=[
                qcl(model.quantile_loss_calculator, model.quantile_ix),
                qel(model.quantile_loss_calculator, model.quantile_ix),
            ],
        )
        res1 = model.train(
            1.0,
            epochs,
            normalize_loss=None,   
            initial_weights=(initial_weights[i] if 'initial_weight' in config else None),
            save_path=os.path.join(args.path, f'model_w{1.0}.keras'),
            metrics=[
                qcl(model.quantile_loss_calculator, model.quantile_ix),
                qel(model.quantile_loss_calculator, model.quantile_ix),
            ],
        )
        min_f1 = res1.history['quantile_coverage_loss'][-1]
        max_f1 = res0.history['quantile_coverage_loss'][-1]
        min_f2 = res0.history['quantile_estimation_loss'][-1]
        max_f2 = res1.history['quantile_estimation_loss'][-1]
        limits = [(min_f1, max_f1), (min_f2, max_f2)]
        print(f'Normalizer limits {limits}')
        write_text_file(os.path.join(args.path, 'normalizer_limits'), str(limits))
    else:
        limits = None

    results = []
    for i, weight in enumerate(weights):
        if normalize_loss and weight == 0.0:
            res = res0
        if normalize_loss and weight == 1.0:
            res = res1
        else:
            res = model.train(
                weight,
                epochs,
                normalize_loss=limits,   
                initial_weights=(initial_weights[i] if 'initial_weight' in config else None),
                save_path=os.path.join(args.path, f'model_w{weight}.keras'),
            )
        results.append(res.history)
        
    joblib.dump(
        results,
        os.path.join(args.path, f'results.z'),
        compress=3)
        
if __name__ == '__main__':
    run()
