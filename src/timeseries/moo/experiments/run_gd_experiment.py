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
from src.timeseries.moo.sds.utils.util import get_from_dict

from src.models.attn.nn_funcs import QuantileLossCalculator

from src.sds.nn.utils import batch_from_list_or_array, predict_from_batches, get_one_output_model, split_model

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
    
    def train(self, weight, epochs=5, normalize_loss=False, initial_weights=None, save_path=None):
        print(f"Starting training with sum weight {weight}")

        self.trainable_model.set_weights(initial_weights if initial_weights is not None else self.initial_weights)

        if normalize_loss:
            loss_f = self.quantile_loss_calculator.quantile_loss_per_q_moo
        else:
            loss_f = self.quantile_loss_calculator.quantile_loss_per_q_moo_no_normalization
        loss = loss_function(loss_f, weight, self.quantile_ix)

        self.trainable_model.compile(
            loss=loss,
            metrics=[
                qcr(self.quantile_loss_calculator, self.quantile_ix),
                qer(self.quantile_loss_calculator, self.quantile_ix),
            ],
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

    results = []
    for weight in combinations:
        res = model.train(
            weight,
            epochs,
            normalize_loss=normalize_loss,   
            initial_weights=None,
            save_path=os.path.join(args.path, f'model_w{weight}.keras'),
        )
        results.append(res.history)
        
    joblib.dump(
        results,
        os.path.join(args.path, f'results.z'),
        compress=3)
        
if __name__ == '__main__':
    run()
