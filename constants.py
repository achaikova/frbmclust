TS_TYPE = 'cal_ts'

class Signal:
    WINDOW_WIDTH = 80  # samples
    SAMPLING = 0.983  # ms


class BrightnessTemperature:
    # arXiv:2112.12301 [astro-ph.HE]

    H0 = 67.7  # Hubble constant (km s^{-1} Mpc^{-1})
    WM = 0.31  # Omega(radiation)
    WV = 0.69  # Omega curvaturve = 1-Omega(total)
    TB_CONST = 1.1 * (10 ** 35)


class Plotting:
    FIGSIZE_PROFILES = (8, 6)
    FIGSIZE_CLUSTERING = (12, 7)
    FIGSIZE_SKYMAP = (8, 6)
    FIGSIZE_BRIGHTNESS_TEMPERATURE = (8, 6)
    YLIM_PROFILES = [-1, 30]
    FORMAT = "pdf"
