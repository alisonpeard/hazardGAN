"""Fit GEV to distribution, assuming stationary block maxima."""

import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF as ecdf
import tensorflow as tf
from hazardGAN import utils, fig_utils


plt.rcParams["font.family"] = "monospace"


def translate_indices(i, dims=(18, 22)):
    indices = np.arange(0, dims[0] * dims[1], 1)
    x = np.argwhere(indices.reshape(dims[0], dims[1]) == i)
    return tuple(*map(tuple, x))


def translate_indices_r(i, j, dims=(18, 22)):
    indices = np.arange(0, dims[0] * dims[1], 1)
    x = indices.reshape(dims[0], dims[1])[i, j]
    return x


ntrain = 2000
conditions = "all"
dim_dict = {'wind_data': 'total', 'wave_data': 'hmax', 'precip_data': 'tp'}
hist_kwargs = {'color': 'lightgrey', 'alpha': .6, 'edgecolor': 'k', 'density': True}

if __name__ == '__main__':
    for variable in ['precip_data', 'wind_data', 'wave_data']:
        print("Processing {}".format(variable))
        dim = dim_dict[variable]
        datadir = f"/Users/alison/Documents/DPhil/multivariate/{variable}"

        # load data
        data, cyclone_flag = utils.load_data(datadir, dim=dim)
        train = data[:ntrain, ...]
        test = data[ntrain:, ...]

        # test first
        marginals, params = utils.probability_integral_transform(test, evt_type='bm')
        np.save(os.path.join(datadir, 'test', 'bm', 'marginals.npy'), marginals)
        np.save(os.path.join(datadir, 'test', 'bm', 'images.npy'), test)
        np.save(os.path.join(datadir, 'test', 'bm', 'params.npy'), params)
        
        # train set
        marginals, params = utils.probability_integral_transform(train, evt_type='bm')
        np.save(os.path.join(datadir, 'train', 'bm', 'marginals.npy'), marginals)
        np.save(os.path.join(datadir, 'train', 'bm', 'images.npy'), train)
        np.save(os.path.join(datadir, 'train', 'bm', 'params.npy'), params)