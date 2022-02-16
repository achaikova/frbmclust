from astropy.coordinates import SkyCoord
from astropy.time import Time
import matplotlib.pyplot as plt
import constants as const
import config
import numpy as np
import pandas as pd

"""
dm - Detection DM (pc cm−3)
ra - Right ascension (J2000)
dec - Declination (J2000)
mjd_400 - Time of arrival in UTC at CHIME location (topocentric) with reference to 400.19 MHz for the specific sub-burst (MJD)
"""


class CelestialCoord:
    def __init__(self, frb_names: list, clusters: list, catalog: pd.DataFrame, save: bool = False):
        self.catalog = catalog[(~catalog['repeater_name'].duplicated()) | (catalog['repeater_name'] == '-9999')]
        self.ra, self.dec = self.catalog['ra'], self.catalog['dec']
        self.dm = self.catalog['bonsai_dm']
        self.save = save
        self.frb_names = np.array(frb_names)
        self.clusters = np.array(clusters)
        self.save = save

    def get_cluster(self, cluster):
        """
        Get ra, dec and mjd_400 of FRBs
        :param cluster (int)
        """
        cluster_ra = self.catalog.loc[self.catalog['tns_name'].isin(self.frb_names[self.clusters == cluster])]['ra']
        cluster_dec = self.catalog.loc[self.catalog['tns_name'].isin(self.frb_names[self.clusters == cluster])]['dec']
        cluster_mjd_400 = self.catalog.loc[self.catalog['tns_name'].isin(self.frb_names[self.clusters == cluster])][
            'mjd_400']
        return cluster_ra, cluster_dec, cluster_mjd_400

    def plot(self):
        """
        Plot aitoff projection in equatorial coordinates
        """
        plt.figure(figsize=const.Plotting.FIGSIZE_SKYMAP)
        plt.subplot(111, projection="aitoff")
        plt.grid(True)
        for cluster in np.unique(self.clusters):
            ra, dec, _ = self.get_cluster(cluster)
            eq_coor = SkyCoord(ra, dec, frame='icrs', unit='deg')
            plt.plot(eq_coor.ra.wrap_at('180d').radian, eq_coor.dec.wrap_at('180d').radian, 'o', markersize=5,
                     label=f'Cluster {cluster}')
        plt.xticks(ticks=np.radians([-150, -120, -90, -60, -30, 0,
                                     30, 60, 90, 120, 150]),
                   labels=['10h', '8h', '6h', '4h', '2h', '0h',
                           '22h', '20h', '18h', '16h', '14h'])
        plt.legend(loc='lower right')
        plt.xlabel('Right ascension')
        plt.ylabel('Declination')
        plt.tight_layout()
        if self.save:
            plt.savefig(f'{config.PLOTS_DIR}/skycoord.{const.Plotting.FORMAT}')

    def hist(self):
        """
        Plot histogram of ra, dec and time of arrival in UTC
        """
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=const.Plotting.FIGSIZE_SKYMAP)
        ax0, ax1, ax2 = axes.flatten()
        ras, decs, mjd_400s = [], [], []
        cluster_labels = np.unique(self.clusters)
        for cluster in cluster_labels:
            ra, dec, mjd_400 = self.get_cluster(cluster)
            ras.append(ra)
            decs.append(dec)
            mjd_400s.append(Time(mjd_400, format='mjd', scale='utc').mjd)
        ax0.hist(ras, bins='fd', label=cluster_labels)
        ax0.set_title('Ra')
        ax1.hist(decs, bins='fd', label=cluster_labels)
        ax1.set_title('Dec')
        ax2.hist(mjd_400s, bins='fd', label=cluster_labels)
        ax2.set_title('UTC (mjd)')
        plt.legend()
        plt.tight_layout()
        if self.save:
            plt.savefig(f'{config.PLOTS_DIR}/skycoord_hist.{const.Plotting.FORMAT}')
