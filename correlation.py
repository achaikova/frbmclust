import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from scipy.signal import correlate

import constants as const
import config


class FRBcorr:
    def __init__(self, repack: bool = False):
        self.catalog = pd.read_csv(config.CHIME_FRB_CATALOG)
        self.data, self.frb_names = self.unpack(repack)

    def unpack(self, repack):
        """
        Unpack waterfall data
        :param repack (bool):
            if True unpack from repacked data (for testing)
        """
        err_frb_names, data = [], {}
        chime_frb_names = np.unique(self.catalog['tns_name']).tolist()
        for i, frb in enumerate(chime_frb_names):
            try:
                if repack:
                    with h5py.File(config.REPACKED_WATERFALL / f"{frb}.h5", "r") as datafile:
                        cal_ts = datafile['cal_ts'][:]
                        ts = datafile['ts'][:]
                        data[frb] = {
                            'cal_ts': cal_ts,
                            'ts': ts
                        }
                else:
                    with h5py.File(config.RESOURCES_DIR / f"{frb}_waterfall.h5", "r") as datafile:
                        cal_wfall = datafile['frb']["calibrated_wfall"][:]
                        cal_ts = np.nanmean(cal_wfall, axis=0)
                        ts = np.nansum(datafile['frb']["wfall"][:], axis=0)
                        data[frb] = {
                            'ts': ts,
                            'cal_ts': cal_ts
                        }
            except (FileNotFoundError, KeyError):
                err_frb_names.append(frb)
        for frb in err_frb_names:
            if frb in chime_frb_names:
                chime_frb_names.remove(frb)
        if len(err_frb_names) > 0:
            print(f"No data found for the next {len(err_frb_names)} FRBs: {err_frb_names}")
        return data, chime_frb_names

    def find_best_corr(self, cur_mean, frb_names, ts_type):
        """
        Find a signal which has the highest correlation with the current mean profile
        Return profile shift, according to the cross-correlation result and FRB name
        :param cur_mean (np.ndarray):
        :param frb_names (list):
        :param ts_type (str):
            ts, cal_ts
        :return:
            int, str
        """
        max_corr, shift, next_frb_name = 0, 0, None
        cur_mean_norm = cur_mean / np.sqrt(np.sum(cur_mean ** 2))
        for frb in frb_names:
            profile = self.get_cal_ts(frb, ts_type)
            profile /= np.sqrt(np.sum(profile ** 2))
            cur_corr = correlate(cur_mean_norm, profile, 'full')
            if np.max(cur_corr) > max_corr:
                max_corr = np.max(cur_corr)
                shift = np.arange(-len(profile) + 1, len(cur_mean_norm))[np.argmax(cur_corr)]
                next_frb_name = frb
        return shift, next_frb_name

    def get_cal_ts(self, frb_name, ts_type):
        """
        Get signal profile
        :param frb_name (str):
        :param ts_type (str):
            ts, cal_ts
        :return:
            list
        """
        profile = self.data[frb_name][ts_type].copy()
        # Baseline removal
        # Here we use median instead of mean
        profile -= np.median(profile)
        return profile

    def tolerant_mean(self, profiles):
        """
        Get mean of zero-padded profiles
        :param profiles (list):
        :return:
            np.ndarray
        """
        lens = [len(i) for i in profiles]
        arr = np.ma.empty((np.max(lens), len(profiles)))
        arr.mask = True
        for idx, l in enumerate(profiles):
            arr[:len(l), idx] = l
        return arr.mean(axis=-1)

    def get_cur_mean(self, cur_mean, shift, next_profile):
        """
        Shift signals according to the cross-correlation results
        :param cur_mean (np.ndarray):
        :param shift (int):
        :param next_profile (str):
        :return:
            np.ndarray
        """
        if shift > 0:
            profile_pad = np.pad(next_profile, (shift, 0), mode='constant', constant_values=0)
        else:
            self.profiles = [np.pad(p, (-shift, 0), mode='constant', constant_values=0) for p in self.profiles]
            self.ts_pad = [np.pad(t, (-shift, 0), mode='constant', constant_values=0) for t in self.ts_pad]
            cur_mean = np.pad(cur_mean, (-shift, 0), mode='constant', constant_values=0)
            profile_pad = next_profile.copy()

        self.ts_pad.append(profile_pad.copy())
        profile_pad /= np.nansum(profile_pad ** 2)
        self.profiles.append(profile_pad.copy())

        max_len = max(np.max([len(i) for i in self.ts_pad]), len(cur_mean))
        self.profiles = [np.pad(p, (0, max_len - len(p)), mode='constant', constant_values=0) for p in self.profiles]
        self.ts_pad = [np.pad(t, (0, max_len - len(t)), mode='constant', constant_values=0) for t in self.ts_pad]

        new_mean = self.tolerant_mean(self.profiles)
        return new_mean

    def get_initial_mean(self, corr):
        """
        Get tne name of a signal which has the highest cross-correlation in the initial cross-correlation matrix
        :param corr (list):
        :return:
            str
        """
        indices = np.array([frb in self.frb_names for frb in corr.columns]).nonzero()[0].tolist()
        corr_subset = corr.loc[indices, self.frb_names]
        max_per_row = corr_subset.max()
        return self.frb_names[max_per_row.argmax()]

    def get_ordered_frb_names(self, corr, ts_type):
        """
        Return list of FRB names and profiles according to their cross-correlation with the mean profile (descending)
        :param corr (list):
        :param ts_type (str):
            ts, cal_ts
        :return:
            list, list
        """
        tmp_frb_names = self.frb_names.copy()
        self.profiles, self.ts_pad = [], []

        first_frb_name = self.get_initial_mean(corr)
        cur_mean = self.get_cal_ts(first_frb_name, ts_type)
        cur_mean -= np.median(cur_mean)

        self.ts_pad.append(cur_mean.copy())
        cur_mean /= (np.nansum(cur_mean ** 2))
        self.profiles.append(cur_mean)
        tmp_frb_names.remove(first_frb_name)
        self.ordered_frb_names = [first_frb_name]

        for _ in range(len(self.frb_names) - 1):
            shift, next_frb = self.find_best_corr(cur_mean, tmp_frb_names, ts_type)
            self.ordered_frb_names.append(next_frb)
            cur_mean = self.get_cur_mean(cur_mean, shift,
                                         self.get_cal_ts(next_frb, ts_type))
            tmp_frb_names.remove(next_frb)
        return self.ordered_frb_names, self.ts_pad

    @staticmethod
    def get_titles(clusters):
        titles = []
        for c in range(len(clusters)):
            if c == 0:
                titles.append(f'{c + 1}-st cluster')
            elif c == 1:
                titles.append(f'{c + 1}-nd cluster')
            elif c == 2:
                titles.append(f'{c + 1}-rd cluster')
            else:
                titles.append(f'{c + 1}-th cluster')
        return titles

    def plot_cluster_mean_compare(self, clusters):
        """
        Compare mean profiles of clusters on one plot
        :param clusters (list):
        """
        x = np.arange(const.Signal.WINDOW_WIDTH * 2) * const.Signal.SAMPLING
        titles = self.get_titles(clusters)
        plt.figure(figsize=const.Plotting.FIGSIZE_PROFILES)
        for cluster, title in zip(clusters, titles):
            mean_profile = np.mean(np.array(self.profiles)[cluster], axis=0) * np.median(
                [np.sum(frb ** 2) for frb in np.array(self.ts_pad)[cluster]])
            peak_index = np.argmax(mean_profile)
            plt.plot(x, mean_profile[peak_index - const.Signal.WINDOW_WIDTH:peak_index + const.Signal.WINDOW_WIDTH],
                     label=title)
            plt.xticks(np.linspace(0, (const.Signal.WINDOW_WIDTH * 2) * const.Signal.SAMPLING, 7))
            plt.ylabel('Normalized flux (Jy)'), plt.xlabel('Time (ms)')
            # plt.ylim(const.Plotting.YLIM_PROFILES)
            plt.title('Averaged profiles')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{config.PLOTS_DIR}/cluster_mean.{const.Plotting.FORMAT}', dpi=300)

    def plot_cluster_mean(self, clusters):
        """
        Plot cluster profiles and mean cluster profile
        :param clusters (list):
        """
        x = np.arange(const.Signal.WINDOW_WIDTH * 2) * const.Signal.SAMPLING
        titles = self.get_titles(clusters)
        for cluster, title in zip(clusters, titles):
            plt.figure(figsize=const.Plotting.FIGSIZE_PROFILES)
            for i in cluster:
                ts_pad = self.ts_pad[i]
                peak_index = np.argmax(ts_pad)
                plt.plot(x, ts_pad[peak_index - const.Signal.WINDOW_WIDTH:peak_index + const.Signal.WINDOW_WIDTH],
                         alpha=0.3)
            mean_profile = np.mean(np.array(self.profiles)[cluster], axis=0) * np.median(
                [np.sum(frb ** 2) for frb in np.array(self.ts_pad)[cluster]])
            peak_index = np.argmax(mean_profile)
            plt.plot(x, mean_profile[peak_index - const.Signal.WINDOW_WIDTH:peak_index + const.Signal.WINDOW_WIDTH],
                     label='Averaged profile',
                     color='black')
            plt.ylabel('Normalized flux (Jy)'), plt.xlabel('Time (ms)')
            plt.ylim(const.Plotting.YLIM_PROFILES)
            plt.yscale('symlog')
            plt.xticks(np.linspace(0, (const.Signal.WINDOW_WIDTH * 2) * const.Signal.SAMPLING, 7))
            plt.legend()
            plt.title(title)
            plt.savefig(f'{config.PLOTS_DIR}/{title}_profiles.{const.Plotting.FORMAT}', dpi=300)

    def calc_corr(self, profiles):
        """
        Calculate cross-correlation matrix for given profiles
        :param profiles (list):
        :return:
            numpy.ndarray
        """
        n = len(profiles)
        corr = np.ones(shape=(n, n))
        for i in range(0, n):
            for j in range(i):
                l = profiles[i] / np.sqrt(np.sum(profiles[i] ** 2))
                r = profiles[j] / np.sqrt(np.sum(profiles[j] ** 2))
                cur_corr = correlate(l, r, 'full')
                vmax = np.max(cur_corr)
                corr[i][j] = corr[j][i] = vmax
        return corr
