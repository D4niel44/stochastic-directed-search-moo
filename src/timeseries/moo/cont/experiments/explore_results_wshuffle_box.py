import itertools
import os

import joblib
import numpy as np
import pandas as pd
import plotly.io as pio
from plotly.subplots import make_subplots
from tabulate import tabulate

from src.models.compare.winners import wilcoxon_rank, wilcoxon_significance, kruskal_significance
from src.timeseries.moo.cont.core.harness import plot_2D_pf
from src.timeseries.moo.cont.utils.results import get_compiled_dict, df_from_dict, compile_metrics, combine_means_stds, \
    adapt_runs
from src.timeseries.moo.cont.utils.util import get_from_dict
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.moo import sort_arr_1st_col
from src.timeseries.utils.util import concat_mean_std_df, write_text_file, latex_table
from src.utils.plot import plot_bidir_2D_points_vectors, \
    plot_pfs, bar_plots_with_errors, bar_plot_3axes_with_errors, plot_2D_points_traces, plot_boxes, \
    plot_2d_grouped_traces, marker_names, color_sequences, plot_radar, set_fig_font_scale, box_plot_colors
import plotly.graph_objs as go
import matplotlib as plt

pio.renderers.default = "browser"


def pad_str_num_cols(df, round_dig=3, thold=1e-1):
    for col in df.columns:

        if max(df[col]) < thold:
            df[col] = df[col].apply(lambda x: "{:.2e}".format(x))
        else:
            digits = df[col].round(round_dig).astype(str).str.split('.')
            df_dig = pd.DataFrame(digits.to_list(), columns=['i', 'd'], index=digits.index)
            df_dig['i'] = df_dig['i'].str.rjust(df_dig['i'].str.len().max(), fillchar=' ')
            df_dig['d'] = df_dig['d'].str.ljust(df_dig['d'].str.len().max(), fillchar='0')
            df[col] = df_dig['i'] + '.' + df_dig['d']


if __name__ == '__main__':

    # %%

    general_cfg = {'save_plot': False,
                   'save_results': False,
                   'show_title': False,
                   'experiment_name': 'model_ix',
                   'plot_individual_pf': False,
                   'color_per_subset': True,
                   'plot_only_important': False,
                   'save_latex': False,
                   }

    project = 'snp'

    files_cfgs = [{'folder': 'step_eps', 'experiment': '2_ix_4_it_9', 'prefix': 'step_eps:', 'x_title': 'step size'},
                  {'folder': 'type_in_pf_eps_type_in_pf_eps_type_in_pf_eps', 'experiment': '2_ix_15_it_4', 'prefix': 'in_pf_', 'x_title':'stop criteria'},
                  {'folder': 'eps', 'experiment': '2_ix_9_it_1', 'prefix': 'eps:', 'x_title': 'beta'},
                  {'folder': 'moo_batch_size', 'experiment': '2_ix_8_it_6', 'prefix': 'moo_batch_size:', 'x_title': 'batch size'},
                  # {'folder': 'batch_ratio_stop_criteria', 'experiment': '2_ix_7_it_2', 'prefix': 'steps_eps:', 'x_title': 'beta'},
                  {'folder': 'model_ix', 'experiment': '10_it_6', 'prefix': 'ix:', 'x_title': 'training'},
                  {'folder': 'split_model', 'experiment': '2_ix_2_it_woX3', 'prefix': 'split_model:', 'x_title': 'MOP problem'},
                  ]

    file_cfg = files_cfgs[0]
    # [
    #     # ('type_in_pf_eps_type_in_pf_eps_type_in_pf_eps', '2_ix_15_it_4'),
    #     # ('eps', '2_ix_9_it_1'),
    #     # ('moo_batch_size', '2_ix_8_it_6'),
    #     # ('batch_ratio_stop_criteria', '2_ix_7_it_2'),
    #     # ('model_ix', '10_it_6'),
    #     # ('split_model', '2_ix_2_it_woX3'),
    # ]

    base_path = os.path.join(get_result_folder({}, project), 'experiments', general_cfg['experiment_name'])
    results_folder = os.path.join(get_result_folder({}, project), 'experiments', file_cfg['folder'])
    cont_results = joblib.load(os.path.join(get_result_folder({}, project), 'experiments', file_cfg['folder'], file_cfg['experiment']) + '.z')
    lbls = [', '.join(['{}:{}'.format(k, v) for k, v in d.items()]) for d in cont_results['exp_lbl']]

    lbls = [s.replace(file_cfg['prefix'], '') for s in lbls]

    x_title = file_cfg['x_title']
    performance_title = file_cfg['x_title']

    print(lbls)
    # %% Use only one experiment file
    if isinstance(cont_results['exp_results'][0]['results'], list):
        subset_metrics_dfs = [[r['subset_metrics'] for r in res['metrics']] for res in cont_results['exp_results']]
        pred_corr_metrics_dfs = [[r['pred_corr_metrics'] for r in res['metrics']] for res in
                                 cont_results['exp_results']]
        times_dfs = [[r['times'] for r in res['metrics']] for res in cont_results['exp_results']]
    else:
        subset_metrics_dfs = [get_from_dict(res, ['metrics', 'subset_metrics']) for res in cont_results['exp_results']]
        pred_corr_metrics_dfs = [get_from_dict(res, ['metrics', 'pred_corr_metrics']) for res in
                                 cont_results['exp_results']]
        times_dfs = [get_from_dict(res, ['metrics', 'times']) for res in cont_results['exp_results']]

        # %%
    items_cfg = {
        'pred_norm': ['pred_corr_metrics', 'pred_norm'],
        'corr_norm': ['pred_corr_metrics', 'corr_norm'],
        'descent_norm': ['pred_corr_metrics', 'descent_norm'],
        'corrector steps': ['pred_corr_metrics', 'n_correctors'],
        'train hv': ['subset_metrics', 'train', 'hv'],
        'valid hv': ['subset_metrics', 'valid', 'hv'],
        'train dist': ['subset_metrics', 'train', 'distances'],
        'valid dist': ['subset_metrics', 'valid', 'distances'],
        'predictor f evals': ['evaluations', 'f', 'predictor'],
        'corrector f evals': ['evaluations', 'f', 'corrector'],
        'descent f evals': ['evaluations', 'f', 'descent'],
        'predictor g evals': ['evaluations', 'grad', 'predictor'],
        'corrector g evals': ['evaluations', 'grad', 'corrector'],
        'descent g evals': ['evaluations', 'grad', 'descent'],
    }

    results_items = {}
    for k, v in items_cfg.items():
        results_items[k] = [[get_from_dict(r, v) for r in res['results']] for res in cont_results['exp_results']]

    results_items['exec time'] = [[r['exec_time'] for r in res['results']] for res in cont_results['exp_results']]
    results_items['descent steps'] = [[len(l) for l in res] for res in results_items['descent_norm']]
    # merge experiment runs in a list of all runs per experiment
    flatten_exp_results = dict([(k, [(list(itertools.chain(*exp)) if isinstance(exp[0], list) else exp) for exp in v]) \
                                for k, v in results_items.items()])

    # %%
    plot_cfgs = [{'keys': ['pred_norm', 'corr_norm', ], 'color_position': 'auto',
                  'secondary_y': True, 'color_ixs': [0, 1, 3]},
                 {'keys': ['train hv', 'valid hv'], 'color_position': 'auto',
                  'secondary_y': False, 'boxmode': 'overlay', 'y_title': 'hypervolume'},
                 {'keys': ['train dist', 'valid dist'], 'color_position': 'auto',
                  'secondary_y': False, 'y_title': 'distance'},
                 {'keys': ['corrector steps', 'descent steps'], 'color_position': 'auto',
                  'secondary_y': False, 'y_title': 'no. steps'},

                 {'keys': ['predictor f evals', 'corrector f evals', 'descent f evals', 'predictor g evals',
                           'corrector g evals', 'descent g evals'],
                  'secondary_y': False, 'type': 'bar'},
                 ]

    wilcoxon_matrix = {}
    for plot_cfg in plot_cfgs:
        metrics = dict([(k, flatten_exp_results[k]) for k in plot_cfg['keys']])
        plot_cfg['wilcoxon'], plot_cfg['color_text'] = {}, {}
        plot_cfg['color'] = {}
        plot_cfg['metrics'] = metrics
        for key, list_of_lists in metrics.items():
            plot_cfg['wilcoxon'][key] = wilcoxon_significance(list_of_lists, labels=lbls, p_thold=0.05)
            wilcoxon_rank = (plot_cfg['wilcoxon'][key].sum(axis=1)).astype(int)
            plot_cfg['color'][key] = wilcoxon_rank
            text = np.array(wilcoxon_rank) / (len(wilcoxon_rank) - 1)
            # text = ['{:.2f}'.format(abs(t)) for t in text]
            plot_cfg['color_text'][key] = text

    # %%
    template = 'plotly_white'
    if not general_cfg['plot_only_important']:
        for plot_cfg in plot_cfgs:
            if plot_cfg.get('type', 'box') == 'box':
                box_plot_colors(plot_cfg,
                                labels=lbls,
                                x_title=x_title,
                                color_label_pos=plot_cfg.get('color_position'),
                                y_title=plot_cfg.get('y_title', None),
                                secondary_y=plot_cfg.get('secondary_y', False))
            else:
                compiled_dict = dict([(k, {'mean': np.mean(np.array(flatten_exp_results[k]), axis=1), \
                                           'std': np.std(np.array(flatten_exp_results[k]), axis=1)}) \
                                      for k in plot_cfg['keys']])
                compiled_dict['lbls'] = lbls
                bar_plots_with_errors(compiled_dict,
                                      secondary_y=plot_cfg.get('secondary_y', False),
                                      title=plot_cfg.get('title', None))


    # %%
    # flatten_exp_results = subset_metrics_dfs

    # if not general_cfg['plot_only_important']:
    #     for plot_cfg in plot_cfgs:
    #         if plot_cfg.get('type', 'box') == 'box':
    #             plot_boxes(dict([(k, flatten_exp_results[k]) for k in plot_cfg['keys']]),
    #                        lbls,
    #                        color_ixs=plot_cfg.get('color_ixs', None),
    #                        secondary_y=plot_cfg.get('secondary_y', False),
    #                        plot_title=False,
    #                        y_title=plot_cfg.get('y_title', False),
    #                        x_title=x_title,
    #                        label_scale=1.8,
    #                        boxmode=plot_cfg.get('boxmode', 'group'),
    #                        title=plot_cfg.get('title', None))
    #
    #             # wilcoxon_rank()
    #         else:
    #             compiled_dict = dict([(k, {'mean': np.mean(np.array(flatten_exp_results[k]), axis=1), \
    #                                        'std': np.std(np.array(flatten_exp_results[k]), axis=1)}) \
    #                                   for k in plot_cfg['keys']])
    #             compiled_dict['lbls'] = lbls
    #             bar_plots_with_errors(compiled_dict,
    #                                   secondary_y=plot_cfg.get('secondary_y', False),
    #                                   title=plot_cfg.get('title', None))

    # %%
    # import plotly.express as px
    #
    # fig = px.imshow(wilcoxon_rank_df)
    # fig.show()
    # print(wilcoxon_rank_df)

    # %%
    if general_cfg['color_per_subset']:
        try:
            exp_lbls = [d['type'] for d in cont_results['exp_lbl']]
            group_ix, j = 0, 0
            colors = [color_sequences[group_ix][j]]
            marker_centroids = [marker_names[0]]
            for i in range(1, len(exp_lbls)):
                if exp_lbls[i] != exp_lbls[i - 1]:
                    group_ix += 1
                    j = 0
                else:
                    j += 1
                colors.append(color_sequences[group_ix][j])
                marker_centroids.append(marker_names[group_ix])
        except Exception as e:
            colors = None
            marker_centroids = marker_names[:len(cont_results['exp_lbl'])]
    else:
        colors = None
        marker_centroids = marker_names[:len(cont_results['exp_lbl'])]

    print(colors, marker_centroids)
    # %%
    # Hv
    titles = ['Hypervolume', 'Distance', 'Predictor and corrector norm', 'Function and gradient evals',
              'Corrector steps']
    dfs = [subset_metrics_dfs, subset_metrics_dfs, pred_corr_metrics_dfs, pred_corr_metrics_dfs, pred_corr_metrics_dfs]
    secondary_ys = [False, False, True, False, False]
    get_values = [{'train hv': {'mean': ['train', 'hv']},
                   'valid hv': {'mean': ['valid', 'hv']},
                   },
                  {'train dist': {'mean': ['train', 'mean norm'],
                                  'std': ['train', 'std norm'],
                                  'count': ['train', 'count']},
                   'valid dist': {'mean': ['valid', 'mean norm'],
                                  'std': ['valid', 'std norm'],
                                  'count': ['valid', 'count']},
                   },
                  {'predictor norm': {'mean': ['predictor', 'mean norm'],
                                      'std': ['predictor', 'std norm'],
                                      'count': ['predictor', 'count']},
                   'corrector norm': {'mean': ['corrector', 'mean norm'],
                                      'std': ['corrector', 'std norm'],
                                      'count': ['corrector', 'count']},
                   },
                  {'predictor f evals': {'mean': ['predictor', 'f_evals']},
                   'predictor g evals': {'mean': ['predictor', 'grad_evals']},
                   'corrector f evals': {'mean': ['corrector', 'f_evals']},
                   'corrector g evals': {'mean': ['corrector', 'grad_evals']},
                   'descent f evals': {'mean': ['descent', 'f_evals']},
                   'descent g evals': {'mean': ['descent', 'grad_evals']},
                   },
                  {'corrector steps': {'mean': ['corrector', 'mean per step'],
                                       'std': ['corrector', 'std per step'],
                                       'count': ['corrector', 'count']},
                   'descent steps': {'mean': ['descent', 'count']},
                   }
                  ]

    compiled_dfs = []
    for title, df, sy, value in zip(titles, dfs, secondary_ys, get_values):
        compiled_dict = get_compiled_dict(df, lbls, value)
        compiled_dict = adapt_runs(compiled_dict)
        compiled_df = df_from_dict(compiled_dict)
        pad_str_num_cols(compiled_df, round_dig=2)
        cols = np.unique([c.replace('_std', '') for c in compiled_df.columns])
        latex_df = concat_mean_std_df(compiled_df, cols)
        compiled_dfs.append(latex_df)
        print(tabulate(latex_df, headers='keys', tablefmt='psql'))

        if general_cfg['save_latex']:
            write_text_file(os.path.join(results_folder, f"{title.replace(' ', '_')}"),
                            latex_table(title,
                                        latex_df.to_latex(escape=False, column_format='r' + 'r' * latex_df.shape[1])))

        # if not general_cfg['plot_only_important']:
        #     bar_plots_with_errors(compiled_dict, title=title, secondary_y=sy)

    # %%
    # Exec Time
    get_values = [{'f(x)': {'mean': ['f(x)', 'mean (s)']},
                   'J(x)': {'mean': ['J(x)', 'mean (s)']},
                   'execution time': {'mean': ['execution', 'mean (s)']}},
                  {'predictor f evals': {'mean': ['predictor', 'f_evals']},
                   'predictor g evals': {'mean': ['predictor', 'grad_evals']},
                   'corrector f evals': {'mean': ['corrector', 'f_evals']},
                   'corrector g evals': {'mean': ['corrector', 'grad_evals']},
                   },
                  {'train hv': {'mean': ['train', 'hv']},
                   'valid hv': {'mean': ['valid', 'hv']},
                   },
                  ]

    compiled_dict, compiled_df = compile_metrics(lbls, [times_dfs, pred_corr_metrics_dfs, subset_metrics_dfs],
                                                 get_values)

    tot_f = np.array(compiled_dict['predictor f evals']['mean']) + np.array(compiled_dict['corrector f evals']['mean'])
    tot_g = np.array(compiled_dict['predictor g evals']['mean']) + np.array(compiled_dict['corrector g evals']['mean'])
    factor = np.array(compiled_dict['J(x)']['mean']) / np.array(compiled_dict['f(x)']['mean'])
    weighted_f_evals = tot_f + tot_g * factor
    compiled_dict['weighted f evals'] = {'mean': np.round(weighted_f_evals, 0).astype(int)}

    if isinstance(cont_results['exp_results'][0]['results'], list):
        mu = np.vstack([np.array(compiled_dict['train hv']['mean']), np.array(compiled_dict['valid hv']['mean'])]).T
        st = np.vstack([np.array(compiled_dict['train hv']['std']), np.array(compiled_dict['valid hv']['std'])]).T
        cnt = np.ones_like(mu) * len(subset_metrics_dfs[0])
        muc, stc = combine_means_stds(mu, st, cnt)
        compiled_dict['mean hv'] = {'mean': muc, 'std': stc}
    else:
        hvs = np.vstack([np.array(compiled_dict['train hv']['mean']), np.array(compiled_dict['valid hv']['mean'])])
        compiled_dict['mean hv'] = {'mean': np.mean(hvs, axis=0), 'std': np.std(hvs, axis=0)}

    compiled_df = df_from_dict(compiled_dict)

    print_df = compiled_df.loc[:, ['execution time', 'weighted f evals', 'mean hv']]
    print(tabulate(print_df, headers='keys', tablefmt='psql'))

    plot_dict = {}
    for key in ['lbls', 'execution time', 'weighted f evals', 'mean hv']:
        plot_dict[key] = compiled_dict[key]

    bar_plot_3axes_with_errors(plot_dict, title='Cpu time, weighted f evals, and hv')

    # %% Performance
    performance = (compiled_dict['mean hv']['mean'] / max(compiled_dict['mean hv']['mean'])) * 100 - (
            compiled_dict['weighted f evals']['mean'] / max(compiled_dict['weighted f evals']['mean']))

    performance = performance / max(performance)

    plot_dict = {'lbls': lbls,
                 'performance': {
                     'mean': performance},
                 }
    # bar_plots_with_errors(plot_dict, title='Performance', secondary_y=False)

    data = np.array([compiled_dict['weighted f evals']['mean'] / 1000, compiled_dict['mean hv']['mean'],
                     compiled_dict['execution time']['mean']]).T
    performance = pd.DataFrame(data, columns=['weighted f evals', 'mean hv', 'execution time'], index=lbls)
    # data = [d.reshape(1, -1) for d in data]

    evals_max, hv_max, et_max = 3.9, 3.41, 1200
    hv_min = 3.1
    performance_norm = pd.DataFrame()
    performance_norm['norm weighted f evals'] = (performance['weighted f evals'] / evals_max) * 100
    performance_norm['norm exec time'] = (performance['execution time'] / et_max) * 100
    performance_norm['norm mean HVs'] = ((performance['mean hv'] - hv_min) / (hv_max - hv_min)) * 100
    print(tabulate(performance_norm, headers='keys', tablefmt='psql'))

    data = [d.reshape(1, -1) for d in performance_norm.values]

    # %%
    keys = ['corrector g evals', 'predictor g evals', 'corrector f evals', 'predictor f evals']

    fg_evals = {key: np.array(results_items[key]) for key in keys}
    tot_fg_evals = {'tot_g': fg_evals['predictor g evals'] + fg_evals['corrector g evals'],
                    'tot_f': fg_evals['predictor f evals'] + fg_evals['corrector f evals']}

    weighted_f = [f + g * c for f, g, c in zip(tot_fg_evals['tot_f'], tot_fg_evals['tot_g'], factor)]

    keys = ['train hv', 'valid hv']

    hvs = {key: np.array(results_items[key]) for key in keys}
    mean_hv = [(t + v) / 2 for t, v in zip(hvs['train hv'], hvs['valid hv'])]

    evals_max, hv_max, et_max = 4.6, 3.41, 1200
    hv_min = 3.1

    norm_f = np.array([(f * 100) / (evals_max * 1000) for f in weighted_f])
    norm_hv = np.array([(h - hv_min) / (hv_max - hv_min) * 100 for h in mean_hv])
    norm_et = np.array([(np.array(r) / et_max) * 100 for r in results_items['exec time']])

    performance_norm = pd.DataFrame()

    keys = ['norm weighted f evals', 'norm exec time', 'norm mean HVs']
    for key, arr in zip(keys, [norm_f, norm_et, norm_hv]):
        performance_norm[key] = np.mean(arr, axis=-1)
        performance_norm[key + '_std'] = np.std(arr, axis=-1)

    performance_norm.index = lbls

    # %%

    pad_str_num_cols(performance_norm, round_dig=2)
    print(performance_norm)
    cols = np.unique([c.replace('_std', '') for c in performance_norm.columns])
    latex_df = concat_mean_std_df(performance_norm, cols)
    print(tabulate(latex_df, headers='keys', tablefmt='psql'))

    if general_cfg['save_latex']:
        write_text_file(os.path.join(results_folder, 'main_table.txt'),
                        latex_table('Results comparison', latex_df.to_latex(escape=False)))

    # %%

    # markersymbols = None
    # centroidsymbols = marker_names[:len(colors_ixs)]

    plot_2d_grouped_traces([norm_f, norm_hv],
                           names=lbls,
                           markersizes=12,
                           colors=colors,
                           markersymbols=None,
                           centroidsymbols=marker_centroids,
                           axes_labels=['Normalized weighted function evals', 'Normalized mean hv'],
                           label_scale=1.8,
                           # title=performance_title,
                           legend_title=performance_title,
                           )

    # plot_2D_points_traces(data, names=lbls, color_ixs=colors_ixs,
    #                       axes_labels=['normalized f evals', 'normalized hypervolume'])

    # %%
    # norm_f_arr = 100 - np.array(norm_f)
    # norm_hv_arr = np.array(norm_hv)
    # norm_et_arr = 100 - np.array(norm_et)
    # radar_data = np.array([norm_f_arr, norm_hv_arr, norm_et_arr])
    #
    # categories = ['Normalized weighted function evals', 'Normalized mean hv', 'Normalized execution time']
    # mean_radar_data = np.mean(radar_data, axis=-1)
    #
    # plot_radar(data=mean_radar_data,
    #            categories=categories,
    #            names=lbls,
    #            colors=colors,
    #            markersymbols=marker_centroids)

    # %%
    img_path = os.path.join(base_path, 'img')
    if isinstance(cont_results['exp_results'][0]['results'], list):
        Fs = [sort_arr_1st_col(res['results'][0]['population']['F']) for res in cont_results['exp_results']]
        fx_inis = [res['results'][0]['independent'][0]['descent']['ini_fx'].reshape(1, -1) for res in
                   cont_results['exp_results']]
    else:
        Fs = [sort_arr_1st_col(res['results']['population']['F']) for res in cont_results['exp_results']]
        fx_inis = [res['results']['independent'][0]['descent']['ini_fx'].reshape(1, -1) for res in
                   cont_results['exp_results']]
    names = lbls + [l + '_ini' for l in lbls]

    if not general_cfg['plot_only_important']:
        plot_2D_pf(Fs, fx_inis, names, general_cfg['save_plot'], os.path.join(img_path, 'pfs'),
                   f_markersize=5,
                   f_mode='markers+lines',
                   colors_ixs=colors * 2 if colors is not None else None)

        plot_pfs(Fs, fx_inis, lbls)

    # %% Plot individual
    if general_cfg['plot_individual_pf']:
        if isinstance(cont_results['exp_results'][0]['results'], list):
            plot_results = [[p['population'] for p in res['results'][0]['independent']] for res in
                            cont_results['exp_results']]
            plot_descent = [[p['descent'] for p in res['results'][0]['independent']] for res in
                            cont_results['exp_results']]
        else:
            plot_results = [[p['population'] for p in res['results']['independent']] for res in
                            cont_results['exp_results']]
            plot_descent = [[p['descent'] for p in res['results']['independent']] for res in
                            cont_results['exp_results']]

        for res, descent in zip(plot_results, plot_descent):
            plot_bidir_2D_points_vectors(res,
                                         descent=descent,
                                         markersize=6,
                                         plot_ps=False,
                                         )