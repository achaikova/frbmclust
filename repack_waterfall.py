import os
import pandas as pd
import config
import numpy as np
import h5py
from vos import Client


def repack():
    """
    Repack calibrated waterfall for input files to be of smaller size
    Obtained profile is mean of calibrated waterfall over each frequency channel
    """
    catalog = pd.read_csv(config.CHIME_FRB_CATALOG)
    frbs = np.unique(catalog['tns_name'])
    if not os.path.exists(config.REPACKED_WATERFALL):
        os.makedirs(config.REPACKED_WATERFALL)
    for frb in frbs:
        filename = f'{frb}_waterfall.h5'
        data = {}
        if not os.path.exists(config.RESOURCES_DIR / filename):
            try:
                Client().copy('/'.join((config.WATERFALL_STORAGE, filename)), config.RESOURCES_DIR / filename)
            except Exception as e:
                print(f"Could not download file {filename}")
                print(e)
                if os.path.exists(config.RESOURCES_DIR / filename):
                    os.remove(config.RESOURCES_DIR / filename)
                continue
        with h5py.File(config.RESOURCES_DIR / filename, "r") as datafile:
            cal_wfall = datafile['frb']["calibrated_wfall"][:]
            cal_ts = np.nanmean(cal_wfall, axis=0)
            ts = np.nansum(datafile['frb']["wfall"][:], axis=0)
            data[frb] = {
                'ts': ts,
                'cal_ts': cal_ts,
            }
        with h5py.File(config.REPACKED_WATERFALL / f"{frb}.h5", "w") as datafile:
            datafile.create_dataset('cal_ts', data=data[frb]['cal_ts'])
            datafile.create_dataset('ts', data=data[frb]['ts'])
        os.remove(config.RESOURCES_DIR / filename)


if __name__ == '__main__':
    repack()
