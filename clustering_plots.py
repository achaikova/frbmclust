import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import copy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import constants as const
import config


def ward_linkage(distances, threshold_ward, maxclust_ward):
    """
    Clusterize data using ward linkage
    :param distances (list),
    :param threshold_ward (int),
    :param maxclust_ward (int),
    :return:
        list
    """
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=const.Plotting.FIGSIZE_CLUSTERING)
    dissimilarity = 1 - abs(distances)
    z_ward = linkage(squareform(dissimilarity), 'ward')
    _ = dendrogram(z_ward, orientation='top',
                   leaf_rotation=90, no_labels=True, color_threshold=threshold_ward)
    # Clusterize the data
    cluster_labels = fcluster(z_ward, maxclust_ward, criterion='maxclust')
    counts = np.unique(cluster_labels, return_counts=True)[1]
    title = f'Ward linkage\n'
    for ind, i in enumerate(counts):
        title += f' Cluster {ind + 1}={i}'
    plt.title(title)
    plt.savefig(f'{config.PLOTS_DIR}/ward_linkage.{const.Plotting.FORMAT}', dpi=300)
    cluster_labels_pd = pd.DataFrame(cluster_labels)
    cluster_labels_pd.set_axis(distances.columns, axis=0, inplace=True)
    cluster_labels_pd.to_csv(f'{config.PLOTS_DIR}/frb_clusters.csv')
    return cluster_labels


def sub_nums(sub_nums_pd, labels_corr):
    """
    Plot catalog parameters (bc_width, flux) distribution according to the resulting clustering
    :param sub_nums_pd (pandas.DataFrame),
    :param labels_corr (list),
    """
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(12, 12))
    tmp_sub_nums = sub_nums_pd.copy()
    tmp_sub_nums['cluster_labels'] = labels_corr
    g_sn = sns.PairGrid(tmp_sub_nums, hue='cluster_labels',
                        vars=['bc_width', 'flux'], palette='bright', diag_sharey=False)
    g_sn.map_diag(sns.kdeplot, common_norm=False, shade=True)
    g_sn.map_offdiag(sns.scatterplot)
    g_sn.add_legend()
    plt.savefig(f'{config.PLOTS_DIR}/bc_width_flux_distribution.{const.Plotting.FORMAT}', dpi=300)
