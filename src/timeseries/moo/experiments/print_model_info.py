# %%
from src.timeseries.moo.sds.config import sds_cfg
from src.timeseries.moo.core.harness import get_model_and_params
from src.timeseries.moo.sds.utils.bash import get_input_args
from src.timeseries.moo.sds.utils.util import get_from_dict

from keras.utils.vis_utils import plot_model

# %%
if __name__ == '__main__':
    project = 'snp'
    sds_cfg['model']['ix'] = get_input_args()['model_ix']
    sds_cfg['model']['ix'] = 5
    #sds_cfg['problem']['split_model'] = problem_size
    sds_cfg['problem']['limits'] = None
    sds_cfg['sds']['max_increment'] = None if sds_cfg['problem']['split_model'] == 'medium' else 0.05
    sds_cfg['sds']['step_size'] = 2.5e-2 if sds_cfg['problem']['split_model'] == 'medium' else 5e-3

    print('Model ix: {}'.format(get_from_dict(sds_cfg, ['model', 'ix'])))

    model_params, results_folder = get_model_and_params(sds_cfg, project)

    with open('./output/model_params.txt', 'w') as f:
        model_params['model'].model.summary(print_fn=lambda x: f.write(x + '\n'))

    plot_model(model_params['model'].model, to_file='./output/model_plot.png', show_shapes=True, show_layer_names=True)

    print(model_params['model'].model.get_layer('tf.math.reduce_sum_2').output)
