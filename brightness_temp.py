import config
from frb.dm import igm
from cosmocalc import cosmocalc
from astropy.cosmology import Planck18
from astropy import units as u
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import constants as const
import numpy as np


def get_tb_params(frb_names, cluster_labels, catalog, emission_freq, precalc_z):
    """
    Get catalog parameters used for calculating brightness temperature

    :param frb_names (list),
    :param cluster_labels (list),
    :param catalog (pandas.DataFrame),
    :param emission_freq (int),
    :returns:
        pandas.DataFrame
    """
    if precalc_z:
        z_da_precalc = pd.read_csv(config.Z_DA_PRECALC)
    rows_frb = []
    for i, name in enumerate(frb_names):
        d = {}
        dm = catalog[catalog['tns_name'] == name]['dm_fitb'].tolist()[0]
        z = z_da_precalc[z_da_precalc['tns_name'] == name]['z'].tolist()[0] if precalc_z else igm.z_from_DM(
            dm * u.pc / u.cm ** 3,
            Planck18)
        dA = z_da_precalc[z_da_precalc['tns_name'] == name]['dA'].tolist()[0] if precalc_z else \
            cosmocalc(z, const.BrightnessTemperature.H0, const.BrightnessTemperature.WM,
                      const.BrightnessTemperature.WV)['DA_Mpc'] / 10 ** 3

        d.update({'tns_name': name,
                  'dm': dm,
                  'flux': catalog[catalog['tns_name'] == name]['flux'].tolist()[0],
                  'emission_freq': emission_freq,
                  'bc_width': catalog[catalog['tns_name'] == name]['bc_width'].tolist()[0],
                  'cluster_label': cluster_labels[i],
                  'z': z,
                  'dA': dA
                  })
        rows_frb.append(d)
    tb_params = pd.DataFrame(rows_frb,
                             columns=['tns_name', 'dm', 'flux', 'emission_freq', 'bc_width', 'cluster_label', 'dA',
                                      'z'])
    tb_params.set_index('tns_name', inplace=True)
    return tb_params


def calc_brightness_temperature(tb_params):
    """
    Calculate brightness temperature

    :param tb_params (pandas.Dataframe),
    """
    for i in tb_params.index:
        tb_params.at[i, 'Tb'] = (tb_params.loc[i, 'flux'] * (tb_params.loc[i, 'dA'] ** 2)) / \
                                (((tb_params.loc[i, 'bc_width'] * (10 ** 3)) /
                                  (1 + tb_params.loc[i, 'z'])) ** 2 * (
                                             tb_params.loc[i, 'emission_freq'] / (10 ** 3)) ** 2)
    tb_params['Tb'] = np.float64(tb_params['Tb'] * const.BrightnessTemperature.TB_CONST)
    plt.figure(figsize=const.Plotting.FIGSIZE_BRIGHTNESS_TEMPERATURE)
    tb_params.rename(columns={'cluster_label': 'Cluster'}, inplace=True)
    sns.histplot(data=tb_params, x="Tb", hue='Cluster', palette='colorblind', kde=True, log_scale=True)
    plt.xlabel('Brightness temperature (K)'), plt.ylabel('Number of bursts')
    # plt.title('')
    plt.savefig(f'{config.PLOTS_DIR}/brightness_temp_hist.{const.Plotting.FORMAT}', dpi=300)
