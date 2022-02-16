# frbmclust



## Getting Started


Clusterize FRB profiles using hierarchical clustering, plot corresponding parameters distributions. 
Input files (waterfall data, CHIME catalog) are stored in the `resources/` directory, output files are stored in the `plots/` directory.

Data used in the clustering belongs to the [CHIME/FRB collaboration](https://www.chime-frb.ca/), [CHIME/FRB Catalog 1 Paper](https://arxiv.org/abs/2106.04352).

## Prerequisites

- Python 3.7+

- The vos Python module for downloading waterfall data: [CANFAR storage system](https://github.com/opencadc/vostools/tree/master/vos)  

  - ```pip install vos --upgrade --user```

- Python code related to DM calculations, estimations, and analysis: [FRB](https://github.com/FRBs/FRB)  

  - For FRB dependencies see: [Installing FRB](https://github.com/FRBs/FRB/blob/main/docs/installing.rst)
    ```
    git clone https://github.com/FRBs/FRB.git
    cd FRB ; python setup.py install --user ; cd ..
    ```
    
    ```
    git clone https://github.com/FRBs/ne2001.git
    cd ne2001 ; python setup.py install --user ; cd ..
    ```
- [astropy](https://docs.astropy.org/en/stable/install.html) v4.3+
    ```
    pip install --upgrade pip 
    pip install astropy
    ```
    
- Installation/upgrade of side packages, which might be required for the version compatibility
  - ```pip install --user --upgrade lxml pandas ipython scipy matplotlib seaborn h5py```
- [cosmocalc](https://cxc.harvard.edu/contrib/cosmocalc/)

	We download `cosmocalc` v0.1.2 and refactor it to be able to run under Python 3 (`2to3` software is used and should be installed via package manager):
    ```
    curl https://files.pythonhosted.org/packages/ca/61/df95b65cd3cf3abdffcbeea6c77a6d8964a8fb625a88eccd9c73d2bfdbf0/cosmocalc-0.1.2.tar.gz | tar -x
    cd cosmocalc-0.1.2 ; 2to3 -w cosmocalc.py ; python setup.py install --user ; cd ..
    ```

## Configuration
For the fast reproduction of the main results one can run `python frb_profiles_clustering.py --use-preprocessed --use-precalculated-z`.

Instead of preprocessed signal data, waterfall data can also be used, which is located in CANFAR storage and which can be downloaded with the `-d` flag.
For the data to be used, it shoud be in the `resources/` folder and have a following name: `{frb_name}_waterfall.h5`

To obtain preprocessed signals, run `python repack_waterfall.py`. It would download waterfall data from CANFAR storage one at a time and save only calibrated profiles – mean of calibrated waterfall data over each frequency channel.

When calling `python frb_profiles_clustering.py` several flags are available:
- `-d, --download`: download waterfall data from CANFAR storage
- `-ef, --emission_freq`: set emission frequency (MHz) used in brightness temperature calculations. Default value is 600 MHz
- `--use-preprocessed`: use preprocessed signal profiles
- `--use-precalculated-z`: use precalculated red-shift values 
- `--use-latex`: enable LaTeX support for plots
