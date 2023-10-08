"""Helper functions for running evtGAN in TensorFlow."""
import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
from scipy.stats import ecdf, genpareto, genextreme
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

SEED = 42

def diff(x, d=1):
    """Difference a (time series) array."""
    return x[d:] - x[:-d]

def translate_indices(i, dims=(18, 22)):
    indices = np.arange(0, dims[0] * dims[1], 1)
    x = np.argwhere(indices.reshape(dims[0], dims[1]) == i)
    return tuple(*map(tuple, x))


def translate_indices_r(i, j, dims=(18, 22)):
    indices = np.arange(0, dims[0] * dims[1], 1)
    x = indices.reshape(dims[0], dims[1])[i, j]
    return x


def unpad(tensor, paddings=tf.constant([[0,0], [1,1], [1,1], [0,0]])):
    """Mine: remove Tensor paddings"""
    tensor = tf.convert_to_tensor(tensor)  # incase its a np.array
    unpaddings = [slice(pad.numpy()[0], -pad.numpy()[1]) if sum(pad.numpy()>0)  else slice(None, None) for pad in paddings]
    return tensor[unpaddings]


def sliding_windows(x, size, jump=1):
    n, *_ = x.shape
    window_indices = sliding_window_indices(size, n, jump)
    return x[window_indices, ...]


def sliding_window_indices(size, n, jump=1):
    windows = []
    i = 0
    for i in range(0, n - size, jump):
        windows.append(np.arange(i, i + size, 1))
    return np.array(windows)


def frechet_transform(uniform):
    """Apply to Tensor transformed to uniform using ecdf."""
    return - 1 / tf.math.log(uniform)


def gumbel_transform(uniform):
    return -np.log(-np.log(uniform))


def inverse_gumbel_transform(data):
    uniform = -tf.math.exp(-tf.math.exp(data))
    return uniform


def probability_integral_transform(dataset, evt_type='pot', prior=None, thresholds=None, fit_tail=False, decluster=None):
    if evt_type == 'pot':
        return probability_integral_transform_pot(dataset, prior=prior, thresholds=thresholds, fit_tail=fit_tail, decluster=decluster)
    elif evt_type == 'bm':
        # NOTE: this takes a lot longer since you're fitting over ALL the data not just the excesses
        return probability_integral_transform_bm(dataset)
    else:
        raise ValueError("Invalid type specified '{}'. Must be one of ['pot', 'bm']".format(evt_type))


def probability_integral_transform_pot(dataset, prior=None, thresholds=None, fit_tail=False, decluster=None):
    """Transform data to uniform distribution using ecdf."""
    n, h, w, c = dataset.shape
    assert c == 1, "single channel only"
    dataset = dataset[..., 0].reshape(n, h * w)

    if fit_tail is True:
        assert thresholds is not None, "Thresholds must be supplied if fitting tail."
        thresholds = thresholds.reshape(h * w)

    uniform, parameters = semiparametric_cdf(dataset, prior, thresholds, fit_tail=fit_tail, decluster=decluster)
    uniform = uniform.reshape(n, h, w, 1)
    parameters = parameters.reshape(h, w, 3)
    return uniform, parameters


def probability_integral_transform_bm(dataset):
    """Transform data to uniform distribution using ecdf."""
    n, h, w, c = dataset.shape
    assert c == 1, "single channel only"
    dataset = dataset[..., 0].reshape(n, h * w)

    uniform, _ = semiparametric_cdf(dataset)  # fully parametric by default (maybe change that)
    parameters = gev_cdf(dataset)

    uniform = uniform.reshape(n, h, w, 1)
    parameters = parameters.reshape(h, w, 3)
    return uniform, parameters


def semiparametric_cdf(dataset, prior=None, thresh=None, fit_tail=False, decluster=None):
    assert dataset.ndim == 2, "Requires 2 dimensions"
    x = dataset.copy()
    n, J = np.shape(x)

    if not hasattr(thresh, '__len__'):
        thresh = [thresh] * J
    else:
        assert len(thresh) == J, "Thresholds vector must have same length as data."
    
    shape = np.empty(J)
    loc = np.empty(J)
    scale = np.empty(J)
    for j in range(J):
        x[:, j], shape[j], loc[j], scale[j] = semiparametric_marginal_cdf(x[:, j], prior=prior, fit_tail=fit_tail, thresh=thresh[j], decluster=decluster)
    parameters = np.stack([shape, loc, scale], axis=-1)
    return x, parameters


def gev_cdf(dataset):
    assert dataset.ndim == 2, "Requires 2 dimensions"
    x = dataset.copy()
    n, J = np.shape(x)    
    shape = np.empty(J)
    loc = np.empty(J)
    scale = np.empty(J)

    for j in (pbar :=tqdm(range(J))):
        pbar.set_description("Fitting GEV to marginals.")
        shape[j], loc[j], scale[j] = genextreme.fit(x[:, j], method="MLE")
    parameters = np.stack([shape, loc, scale], axis=-1)
    return parameters


def semiparametric_marginal_cdf(x, prior=None, fit_tail=False, thresh=None, decluster=None):
    """§Heffernan & Tawn (2004). 
    
    Note shape parameter is opposite sign to Heffernan & Tawn (2004).
    Thresh here is a value, not a percentage."""
    
    if (x.max() - x.min()) == 0.:
        return np.array([0.] * len(x)), 0, 0, 0
    
    x = x.astype(np.float64)  # otherwise f_thresh gets rounded down below f_thresh
    f = ecdf(x).astype(np.float64)

    if fit_tail:
        assert thresh is not None, "Threshold must be supplied if fitting tail."
        x_tail = x[x > thresh]
        f_tail = f[x > thresh]

        if type(decluster) == int:  # if declustering is being used
            indices = decluster_array(x, thresh, decluster)
            x_fitting = x[indices]
        elif decluster is None:
            x_fitting = x_tail
        else:
            raise ValueError("Invalid declustering type. Must be one of [None, int]")

        shape, loc, scale = genpareto.fit(x_fitting, floc=thresh, method="MLE")
        f_thresh = np.interp(thresh, sorted(x), sorted(f))
        f_tail = 1 - (1 - f_thresh) * (np.maximum(0, (1 + shape * (x_tail - thresh) / scale)) ** (-1 / shape))  # second set of parenthesis important
        assert min(f_tail) >= f_thresh, "Error in upper tail calculation."
        f[x > thresh] = f_tail
        f *= 1 - 1e-6
    else:
        shape, loc, scale = 0, 0, 0
        f *= 1 - 1e-6
    return f, shape, loc, scale


def rank(x):
    if x.std() == 0:
        ranked = np.array([len(x) / 2 + 0.5] * len(x))
    else:
        temp = x.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(x))
        ranked = ranks + 1
    return ranked


def ecdf(x):
    return rank(x) / (len(x) + 1)


def decluster_array(x:np.array, thresh:float, r:int):
    if (r is None) or (r==0):
        warnings.warn(f'decluster_array only returns indices of exceedences for r={r}.')
        return np.where(x > thresh)[0]
    exceedences = x > thresh
    clusters = identify_clusters(exceedences, r)
    cluster_maxima = []
    for cluster in clusters:
        argmax = cluster[np.argmax(x[cluster])]
        cluster_maxima.append(argmax)   
    return cluster_maxima


def identify_clusters(x:np.array, r:int):
    clusters = []
    cluster_no = 0
    clusters.append([])
    false_counts = 0
    for i, val in enumerate(x):
        if false_counts == r:
            clusters.append([])
            cluster_no += 1
            false_counts = 0
        if val:
            clusters[cluster_no].append(i)
        else:
            false_counts += 1
    clusters = [cluster for cluster in clusters if len(cluster) > 0]
    return clusters


def inv_probability_integral_transform(marginals, x=None, y=None, params=None, evt_type='pot', thresh=None):
    if evt_type == 'pot':
        return inv_probability_integral_transform_pot(marginals, x, y, params, thresh)
    elif evt_type == "bm":
        return inv_probability_integral_transform_bm(marginals, params)
    else:
        raise ValueError("Unknown evt_type '{}'".format(evt_type))


def inv_probability_integral_transform_bm(marginals, params):
    """Transform uniform marginals to original distributions, using quantile function of GEV."""
    assert marginals.ndim == 4, "Function takes rank 4 arrays"
    n, h, w, c = marginals.shape
    marginals = marginals.reshape(n, h * w, c)
    assert params.shape == (h, w, 3, c), "Marginals and parameters have different dimensions."
    params = params.reshape(h * w, 3, c)
    
    quantiles = []
    for channel in range(c):
        q = np.array([genextreme.ppf(marginals[:, j, channel], *params[j, ..., channel]) for j in range(h * w)]).T
        quantiles.append(q)
    
    quantiles = np.stack(quantiles, axis=-1)
    quantiles = quantiles.reshape(n, h, w, c)
    return quantiles


def inv_probability_integral_transform_pot(marginals, x, y, params=None, thresh=None):
    """Transform uniform marginals to original distributions, by inverse-interpolating ecdf."""
    assert marginals.ndim == 4, "Function takes rank 4 arrays"
    n, h, w, c = marginals.shape

    assert x.shape[1:] == (h, w, c), f"Marginals and x have different dimensions: {x.shape[1:]} != {h, w, c}."
    assert y.shape[1:] == (h, w, c), f"Marginals and y have different dimensions: {y.shape[1:]} != {h, w, c}."
    assert x.shape[0] == tf.shape(y)[0], f"x and y have different dimensions: {x.shape[0]} != {y.shape[0]}."
    
    marginals = marginals.reshape(n, h * w, c)
    x = x.reshape(len(x), h * w, c)
    y = y.reshape(len(y), h * w, c)

    if params is not None:
        assert params.shape == (h, w, 3, c), "Marginals and parameters have different dimensions."
        params = params.reshape(h * w, 3, c)
    
    quantiles = []
    for channel in range(c):
        if params is None:
            q = np.array([empirical_quantile(marginals[:, j, channel], x[:, j, channel], y[:, j, channel]) for j in range(h * w)]).T
        else:
            if hasattr(thresh, "__len__"):
                thresh = thresh.reshape(h * w, c)
            else:
                thresh = [thresh] * (h * w, c)
            q = np.array([empirical_quantile(marginals[:, j, channel], x[:, j, channel], y[:, j, channel], params[j, ..., channel], thresh[j, channel]) for j in range(h * w)]).T
        quantiles.append(q)
    
    quantiles = np.stack(quantiles, axis=-1)
    quantiles = quantiles.reshape(n, h, w, c)
    return quantiles


def empirical_quantile(marginals, x, y, params=None, thresh=None):
    """(Semi)empirical quantile/percent/point function.
    
    x [was] a vector of interpolated quantiles of data (usually 100,000)
    Now x and y are data that original marginals were calculated from, where x
    is data and y corresponding densities."""
    n = len(x)
    x = sorted(x)

    if (marginals.max() - marginals.min()) == 0.:
        return np.array([-999] * len(marginals)) # no data proxy

    if marginals.max() >= 1:
        warnings.warn("Some marginals >= 1.")
        marginals *= 1 - 1e-6
    
    quantiles = np.interp(marginals, sorted(y), sorted(x))
    if params is not None:
        f_thresh = np.interp(thresh, sorted(x), sorted(y))
        
        marginals_tail = marginals[marginals > f_thresh]
        quantiles_tail = upper_ppf(marginals_tail, thresh, f_thresh, params)
        quantiles[marginals > f_thresh] = quantiles_tail
    
    return quantiles


def upper_ppf(marginals, u_x, thresh, params):
    """Inverse of (1.3) H&T for $\ksi\leq 0$ and upper tail."""
    shape, scale = params[0], params[2]
    x = u_x + (scale / shape) * (1 - ((1 - marginals) / (1 - thresh))**shape)
    return x


def get_extremal_correlations(marginals, sample_inds):
    coeffs = get_extremal_coeffs(marginals, sample_inds)
    coors = {indices: 2 - coef for indices, coef in coeffs.items()}
    return coors


def get_extremal_coeffs(marginals, sample_inds):
    data = tf.cast(marginals, dtype=tf.float32)
    n, h, w = tf.shape(data)[:3]
    data = tf.reshape(data, [n, h * w])
    data = tf.gather(data, sample_inds, axis=1)
    frechet = inv_frechet(data)
    ecs = {}
    for i in range(len(sample_inds)):
        for j in range(i):
            ecs[sample_inds[i], sample_inds[j]] = raw_extremal_coefficient(frechet[:, i], frechet[:, j]).numpy()
    return ecs


def exp(uniform):
    exp_distributed = -tf.math.log(1 - uniform)
    return exp_distributed


def inv_frechet(uniform):
    exp_distributed = -tf.math.log(uniform)
    return exp_distributed


def raw_extremal_coefficient(frechet_x, frechet_y):
    """Where x and y have been transformed to their Fréchet marginal distributions.

    ..[1] Max-stable process and spatial extremes, Smith (1990) §4
    """
    n = tf.shape(frechet_x)[0]
    assert n == tf.shape(frechet_y)[0]
    n = tf.cast(n, dtype=tf.float32)
    minima = tf.reduce_sum(tf.math.minimum(frechet_x, frechet_y))
    if tf.greater(tf.reduce_sum(minima), 0):
        theta = n / minima
    else:
        tf.print("Warning: all zeros in minima array.")
        theta = 2
    return theta

def get_extremal_corrs_nd(marginals, sample_inds):
    """Calculate extremal coefficients across D-dimensional uniform data."""
    _, _, _, d = marginals.shape
    coefs = get_extremal_coeffs_nd(marginals, sample_inds)
    return {key: d - val for key, val in coefs.items()}


def get_extremal_coeffs_nd(marginals, sample_inds):
    """Calculate extremal coefficients across D-dimensional uniform data."""
    n, h, w, d = marginals.shape
    data = marginals.reshape(n, h * w, d)
    data = data[:, sample_inds, :]
    frechet = inv_frechet(data)
    ecs = {}
    for i in range(len(sample_inds)):
        ecs[sample_inds[i]] = raw_extremal_coeff_nd(frechet[:, i, :])
    return ecs


def raw_extremal_coeff_nd(frechets):
    n, d = frechets.shape
    minima = np.min(frechets, axis=1) # minimum for each row
    minima = np.sum(minima)
    if minima > 0:
        theta = n / minima
    else:
        print("Warning: all zeros in minima array.")
        theta = d
    return theta


def gaussian_blur(img, kernel_size=11, sigma=5):
    """See: https://gist.github.com/blzq/c87d42f45a8c5a53f5b393e27b1f5319"""
    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
                                  padding='SAME', data_format='NHWC')


def interpolate_thresholds(thresholds, x, y):
    n, h, w, c = x.shape
    
    thresholds = thresholds.reshape(h * w, c)
    x = x.reshape(n, h * w, c)
    y = y.reshape(n, h * w, c)
    f_thresholds = np.empty([h * w, c])
    
    for i in range(c):
        for j in range(h * w):
            f_thresholds[j, i] = np.interp(thresholds[j, i], x[:, j, i], y[:, j, i])
                           
    f_thresholds = f_thresholds.reshape(h, w, c)
    return f_thresholds



########################################################################################################
def load_data(datadir, imsize=(18, 22), block_size='daily', conditions='all', dim=None):
    """Load wind image data to correct size."""
    assert conditions in ['cyclone', 'all'], "Invalid conditions."
    df = pd.read_csv(os.path.join(datadir, block_size, f"{dim}_{block_size}max.csv"), index_col=[0])
    if conditions == "cyclone":
        df = df[df['cyclone_flag']]
    cyclone_flag = df['cyclone_flag'].values
    df = df.drop(columns=['time', 'cyclone_flag'])
    values = df.values.astype(float)
    ngrids = int(np.sqrt(values.shape[1]))
    values = values.reshape(values.shape[0], ngrids, ngrids)
    values = np.flip(values, axis=[1])
    values = values[..., np.newaxis]
    data = tf.image.resize(values, (imsize[0], imsize[1]))
    return data.numpy(), cyclone_flag


def load_training_data(datadir, train_size=200, datas=['wind_data', 'wave_data', 'precip_data'],
                       evt_type='pot', block_size='daily',
                       paddings=tf.constant([[0,0], [1,1], [1,1], [0,0]]),
                       shuffle=True):
    marginals = []
    images = []
    params = []
    for data in datas:
        marginals.append(np.load(os.path.join(datadir, data, block_size, 'train', evt_type, 'marginals.npy'))[..., 0])
        params.append(np.load(os.path.join(datadir, data, block_size, 'train', evt_type, 'params.npy')))
        images.append(np.load(os.path.join(datadir, data, block_size, 'train', evt_type, 'images.npy'))[..., 0])

    marginals = np.stack(marginals, axis=-1)
    params = np.stack(params, axis=-1)
    images = np.stack(images, axis=-1)
    marginals = tf.pad(marginals, paddings)

    # train/valid split
    if shuffle:
        np.random.seed(2)
        train_inds = np.random.choice(np.arange(0, marginals.shape[0], 1), size=train_size, replace=False)
    else:
        train_inds = np.arange(0, train_size, 1)
    
    marginals_train = np.take(marginals, train_inds, axis=0)
    marginals_test = np.delete(marginals, train_inds, axis=0)
    images = np.take(images, train_inds, axis=0)

    if evt_type == "pot":
        thresholds = []
        for data in datas:
            thresholds.append(np.load(os.path.join(datadir, data, block_size, 'train', evt_type, 'thresholds.npy')))
        thresholds = np.stack(thresholds, axis=-1)
    else:
        thresholds = None

    return marginals_train, marginals_test, params, images, thresholds


def load_test_data(datadir, datas=['wind_data', 'wave_data', 'precip_data'],
                   evt_type='pot', paddings=tf.constant([[0,0], [1,1], [1,1], [0,0]])):
    marginals = []
    images = []
    params = []

    for data in datas:
        marginals.append(np.load(os.path.join(datadir, data, 'test', evt_type, 'marginals.npy'))[..., 0])
        params.append(np.load(os.path.join(datadir, data, 'train', evt_type, 'params.npy'))) # params from training set
        images.append(np.load(os.path.join(datadir, data, 'test', evt_type, 'images.npy'))[..., 0])

    marginals = np.stack(marginals, axis=-1)
    params = np.stack(params, axis=-1)
    images = np.stack(images, axis=-1)

    # paddings
    marginals = tf.pad(marginals, paddings)
    return marginals, images, params