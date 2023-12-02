"""
Note, requires config to create new model too.
>>> new_gan = DCGAN.DCGAN(config)
>>> new_gan.generator.load_weights(os.path.join(wd, 'saved_models', f'{finish_time}_generator_weights'))
>>> new_gan.discriminator.load_weights(os.path.join(wd, 'saved_models', f'{finish_time}_discriminator_weights'))
"""

import os
import yaml
import numpy as np
from datetime import datetime
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import wandb
from wandb.keras import WandbCallback

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from hazardGAN import ChiScore, CrossEntropy, tDCGAN, utils, fig_utils, compile_dcgan

global rundir
global evt_type

plot_kwargs = {'bbox_inches': 'tight', 'dpi': 300}

# some static variables
cwd = os.getcwd() # scripts directory
wd = os.path.join(cwd, "..") # hazardGAN directory
datadir = os.path.join(wd, "..") # keep data folder in parent directory 
datas = ['wind_data', 'wave_data', 'precip_data']
imdir = os.path.join(wd, 'figures', 'temp')
paddings = tf.constant([[0,0], [1,1], [1,1], [0,0]])
evt_type = "bm"


def log_image_to_wandb(fig, name:str, dir:str):
    impath = os.path.join(dir, f"{name}.png")
    fig.savefig(impath, **plot_kwargs)
    wandb.log({name: wandb.Image(impath)})


def save_config(dir):
    configfile = open(os.path.join(dir, "config-defaults.yaml"), "w")
    configdict = {key: {"value": value} for key, value in wandb.config.as_dict().items()}
    yaml.dump(configdict, configfile)
    configfile.close()


def main(config):
    # load data
    train_marginals, test_marginals, params, images, thresholds = utils.load_training_data(datadir, config.train_size, datas=datas, evt_type=evt_type, paddings=paddings)
    # construct sliding windows
    train_marginals = utils.sliding_windows(train_marginals, 7, 7)
    test_marginals = utils.sliding_windows(test_marginals, 7, 7)

    train = tf.data.Dataset.from_tensor_slices(train_marginals).batch(config.batch_size)
    test = tf.data.Dataset.from_tensor_slices(test_marginals).batch(config.batch_size)

    # callbacks
    # chi_score = ChiScore({'train': next(iter(train)), 'test': next(iter(test))}, frequency=config.chi_frequency)
    cross_entropy = CrossEntropy(next(iter(test)))

    # compile
    with tf.device('/gpu:0'):
        gan = compile_dcgan(config, nchannels=len(datas))
        # gan.generator.load_weights("/Users/alison/Documents/DPhil/multivariate/hazardGAN/saved-models/deft-sweep-7/generator_weights")
        # gan.discriminator.load_weights("/Users/alison/Documents/DPhil/multivariate/hazardGAN/saved-models/deft-sweep-7/discriminator_weights")
        gan.fit(train, epochs=config.nepochs, callbacks=[WandbCallback(), cross_entropy]) #, chi_score

    # reproducibility
    gan.generator.save_weights(os.path.join(rundir, f'generator_weights'))
    gan.discriminator.save_weights(os.path.join(rundir, f'discriminator_weights'))
    save_config(rundir)

    # # generate 1000 images to visualise some results
    # train_marginals = utils.tf_unpad(train_marginals, paddings).numpy()
    # test_marginals = utils.tf_unpad(test_marginals, paddings).numpy()

    # fake_marginals = gan(1000)
    # fake_marginals = utils.tf_unpad(fake_marginals, paddings)
    # fake_marginals = fake_marginals.numpy()

    # fig = fig_utils.plot_generated_marginals(fake_marginals)
    # log_image_to_wandb(fig, f'generated_marginals', imdir)

    # get rough ecs without the tail-fitting
    # fig = fig_utils.compare_ecs_plot(train_marginals, test_marginals, fake_marginals, images, train_marginals, channel=0)
    # log_image_to_wandb(fig, 'correlations_u10', imdir)

    # fig = fig_utils.compare_ecs_plot(train_marginals, test_marginals, fake_marginals, images, train_marginals, channel=0)
    # log_image_to_wandb(fig, 'correlations_v10', imdir)

    # plt.show()


if __name__ == "__main__":
    wandb.init(settings=wandb.Settings(code_dir=".")) # saves snapshot of code as artifact (less useful now)

    rundir = os.path.join(wd, "saved-temporal-models", evt_type, wandb.run.name)
    os.makedirs(rundir)

    tf.keras.utils.set_random_seed(wandb.config['seed']) # sets seeds for base-python, numpy and tf
    tf.config.experimental.enable_op_determinism() # removes stochasticity from individual operations
    main(wandb.config)
