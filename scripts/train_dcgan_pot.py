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

from evtGAN import ChiScore, CrossEntropy, DCGAN, tf_utils, viz_utils, compile_dcgan

global rundir

plot_kwargs = {'bbox_inches': 'tight', 'dpi': 300}

# some static variables
cwd = os.getcwd() # scripts directory
wd = os.path.join(cwd, "..") # cycloneGAN directory
datadir = os.path.join(wd, "..", "multivariate") # keep data folder in parent directory 
imdir = os.path.join(wd, 'figures', 'temp')
paddings = tf.constant([[0,0], [1,1], [1,1], [0,0]])


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
    train_marginals, test_marginals, params, images, thresholds = tf_utils.load_training_data(datadir, config.train_size, paddings=paddings)
    #test_marginals, *_ = tf_utils.load_test_data(datadir)

    train = tf.data.Dataset.from_tensor_slices(train_marginals).batch(config.batch_size)
    test = tf.data.Dataset.from_tensor_slices(test_marginals).batch(config.batch_size)

    # train test callbacks
    chi_score = ChiScore({'train': next(iter(train)), 'test': next(iter(test))}, frequency=config.chi_frequency)
    cross_entropy = CrossEntropy(next(iter(test)))

    # compile
    with tf.device('/gpu:0'):
        gan = compile_dcgan(config, nchannels=3)
        # gan.generator.load_weights("/Users/alison/Documents/DPhil/multivariate/cycloneGAN/saved-models/deft-sweep-7/generator_weights")
        # gan.discriminator.load_weights("/Users/alison/Documents/DPhil/multivariate/cycloneGAN/saved-models/deft-sweep-7/discriminator_weights")
        gan.fit(train, epochs=config.nepochs, callbacks=[WandbCallback(), chi_score, cross_entropy])

    # reproducibility
    gan.generator.save_weights(os.path.join(rundir, f'generator_weights'))
    gan.discriminator.save_weights(os.path.join(rundir, f'discriminator_weights'))
    save_config(rundir)

    # generate 1000 images to visualise some results
    train_marginals = tf_utils.tf_unpad(train_marginals, paddings).numpy()
    test_marginals = tf_utils.tf_unpad(test_marginals, paddings).numpy()

    fake_marginals = gan(1000)
    fake_marginals = tf_utils.tf_unpad(fake_marginals, paddings)
    fake_marginals = fake_marginals.numpy()

    fig = viz_utils.plot_generated_marginals(fake_marginals)
    log_image_to_wandb(fig, f'generated_marginals', imdir)

    # get rough ecs without the tail-fitting
    fig = viz_utils.compare_ecs_plot(train_marginals, test_marginals, fake_marginals, images, train_marginals, channel=0)
    log_image_to_wandb(fig, 'correlations_u10', imdir)

    fig = viz_utils.compare_ecs_plot(train_marginals, test_marginals, fake_marginals, images, train_marginals, channel=0)
    log_image_to_wandb(fig, 'correlations_v10', imdir)

    #plt.show()


if __name__ == "__main__":
    wandb.init(settings=wandb.Settings(code_dir=".")) # saves snapshot of code as artifact (less useful now)

    rundir = os.path.join(wd, "saved-models", wandb.run.name)
    os.makedirs(rundir)

    tf.keras.utils.set_random_seed(wandb.config['seed']) # sets seeds for base-python, numpy and tf
    tf.config.experimental.enable_op_determinism() # removes stochasticity from individual operations
    main(wandb.config)
