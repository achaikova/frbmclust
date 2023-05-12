import argparse
import os

import numpy as np
import pandas as pd
from vos import Client

# from vos import Client

import brightness_temp
import config
import constants
import correlation
import skycooord
from clustering_plots import ward_linkage, sub_nums


def enable_latex():

    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('text', usetex=True)
    rc('font', size=14.0)
    rc("axes", labelsize=14.0)
    rc("xtick", labelsize=14.0)
    rc("xtick", direction="in")
    rc("ytick", labelsize=14.0)
    rc("ytick", direction="in")
    rc("text.latex", preamble=r'\renewcommand{\familydefault}{\sfdefault} \usepackage{cmbright} \usepackage{textcomp}')
    rc('xtick', labelsize=14.0)
    rc('ytick', labelsize=14.0)

    # plt.rcParams['figure.constrained_layout.use'] = True
    # fig = plt.figure(figsize=(12. / 2.54, 7. / 2.54))
    # fig.set_constrained_layout_pads(w_pad=0.05 / 2.54, h_pad=0.05 / 2.54, hspace=0, wspace=0)


def get_features(catalog, with_repeaters, with_sub_num):
    """
    Get relevant catalog parameters for bursts
    :param catalog (pandas.DataFrame):
    :param with_repeaters (bool):
        if True return parameters for all bursts, else only for non-repeaters
    :param with_sub_num (bool):
        if True return parameters for each burst, else for each sub-burst
    :return:
        pandas.DataFrame
    """
    chime_features_names = catalog[['tns_name', 'previous_name', 'repeater_name', 'bonsai_snr', 'bonsai_dm',
       'snr_fitb', 'dm_fitb', 'bc_width', 'flux', 'fluence', 'sub_num',
       'sp_idx', 'sp_run', 'high_freq', 'low_freq', 'peak_freq']]
    if not with_repeaters:
        chime_features_names = chime_features_names[chime_features_names['repeater_name'] == '-9999']

    chime_features = chime_features_names.drop(columns=['tns_name', 'previous_name', 'repeater_name'])

    corr = chime_features.corr()

    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= 0.9:
                if columns[j]:
                    columns[j] = False
    selected_columns = chime_features.columns[columns]
    chime_features = chime_features[selected_columns]
    chime_features['tns_name'] = chime_features_names['tns_name']
    chime_features['repeater_name'] = chime_features_names['repeater_name']

    chime_features.loc[chime_features.repeater_name == '-9999', 'repeater_name'] = 0
    chime_features.loc[chime_features.repeater_name != 0, 'repeater_name'] = 1
    chime_features['repeater_name'] = chime_features['repeater_name'].astype('float')
    if with_sub_num:
        chime_features_sub_nums_res = chime_features.drop(columns=['sub_num', 'sp_idx', 'sp_run', 'high_freq',
                                                                   'low_freq', 'peak_freq'])
        chime_features_sub_nums_res = chime_features_sub_nums_res.drop_duplicates(ignore_index=True)
        chime_features_sub_nums_res['sub_num'] = chime_features_sub_nums_res['tns_name'].map(
            chime_features['tns_name'].value_counts(sort=False))
        return chime_features_sub_nums_res.dropna()
    else:
        chime_features_every_burst_res = chime_features.drop(columns=['sub_num'])
        return chime_features_every_burst_res.dropna()


def get_cluster_indices(labels):
    """
    Get indices of clusters
    :param labels (list):
    :return:
        list
    """
    classes = np.unique(labels)
    clusters = [np.where(labels == i)[0] for i in classes]
    return clusters


def plot_brightness_temp(ordered_frb_names, cluster_labels, catalog, emission_freq, precalc_z):
    """
    Plot brightness temperature distribution by clusters
    :param ordered_frb_names:
    :param cluster_labels:
    :param catalog:
    :param emission_freq (int):
    """
    tb_params = brightness_temp.get_tb_params(ordered_frb_names, cluster_labels, catalog, emission_freq, precalc_z)
    brightness_temp.calc_brightness_temperature(tb_params)


def plot_clusters(preprocessed, ts_type, catalog):
    """
    Clusterize FRB profiles, plot resulting cluster profiles and catalog parameters distribution
    :param preprocessed (bool):
    :param ts_type (str):
        ts, cal_ts
    :param catalog (pandas.DataFrame):
    :return:
        list, list
    """
    frb_corr = correlation.FRBcorr(preprocessed)
    profile_arr = [frb_corr.data[frb][ts_type] for frb in frb_corr.data.keys()]
    init_corr = frb_corr.calc_corr(profile_arr)
    init_corr_pd = pd.DataFrame(init_corr)
    init_corr_pd.set_axis(list(frb_corr.data.keys()), axis=1, inplace=True)

    ordered_frb_names, ordered_ts_pad = frb_corr.get_ordered_frb_names(init_corr_pd, ts_type)
    cur_corr = pd.DataFrame(frb_corr.calc_corr(ordered_ts_pad))
    cur_corr.set_axis(ordered_frb_names, axis=1, inplace=True)
    chime_features_sub_nums_all = get_features(catalog, with_repeaters=True, with_sub_num=True)
    corresponding_sub_nums = chime_features_sub_nums_all[
        chime_features_sub_nums_all['tns_name'].isin(ordered_frb_names)]
    reindexed_corresponding_sub_nums = corresponding_sub_nums.set_index('tns_name').reindex(
        cur_corr.columns)

    cluster_labels = ward_linkage(cur_corr, threshold_ward=4, maxclust_ward=2)
    sub_nums(reindexed_corresponding_sub_nums, cluster_labels)

    frb_corr.plot_cluster_mean_compare(get_cluster_indices(cluster_labels))
    frb_corr.plot_cluster_mean(get_cluster_indices(cluster_labels))

    return ordered_frb_names, cluster_labels


def plot_skycoord(ordered_frb_names, cluster_labels, catalog):
    """
    Plot aitoff projection and equatorial coordinates and time of arrival histogram
    :param ordered_frb_names (list):
    :param cluster_labels (list): 
    :param catalog (pandas.DataFrame): 
    """
    coord = skycooord.CelestialCoord(ordered_frb_names, cluster_labels, catalog, save=True)
    coord.plot()
    coord.hist()


def profile_classification(args):
    """
    Get bursts classification based on their morphology, plot corresponding parameters distributions
    :param args: 
    """
    if not os.path.exists(config.PLOTS_DIR):
        os.makedirs(config.PLOTS_DIR)
    if not os.path.exists(config.RESOURCES_DIR):
        os.makedirs(config.RESOURCES_DIR)
    catalog = pd.read_csv(config.CHIME_FRB_CATALOG)
    if args.download:
        read_bursts(catalog)
    ordered_frb_names, cluster_labels = plot_clusters(args.preprocessed, constants.TS_TYPE, catalog)
    plot_skycoord(ordered_frb_names, cluster_labels, catalog)
    plot_brightness_temp(ordered_frb_names, cluster_labels, catalog, args.emission_freq, args.precalc_z)


def read_bursts(catalog):
    """
    Download waterfall data
    :param catalog (pandas.DataFrame): 
    """
    for frb_name in catalog['tns_name']:
        filename = f'{frb_name}_waterfall.h5'
        if not os.path.exists(config.RESOURCES_DIR / filename):
            try:
                Client().copy('/'.join((config.WATERFALL_STORAGE, filename)), config.RESOURCES_DIR / filename)
            except Exception as e:
                print(f"Could not download file {filename}")
                print(e)
                if os.path.exists(config.RESOURCES_DIR / filename):
                    os.remove(config.RESOURCES_DIR / filename)


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--download', help='Download waterfall data', action='store_true', dest='download')

parser.add_argument('--use-preprocessed', help='Use preprocessed signals', action='store_true',
                    dest='preprocessed')
parser.add_argument('-ef', '--emission_freq',
                    help='Set emission frequency used in brightness temperature formula (MHz). Default value is 600 MHz',
                    nargs=1,
                    default=600,
                    dest='emission_freq')
parser.add_argument('--use-precalculated-z', help='Use precalculated red-shift values', action='store_true',
                    dest='precalc_z')
parser.add_argument('--use-latex', help='Enable LaTeX support for plots', action='store_true', dest='use_latex')
parser_args = parser.parse_args()

if parser_args.use_latex:
    enable_latex()

profile_classification(parser_args)
