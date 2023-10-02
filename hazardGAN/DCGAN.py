"""
WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.
--> should be fine on arc though.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
from tensorflow.nn import sigmoid_cross_entropy_with_logits as cross_entropy


# for plotting
import matplotlib.pyplot as plt
from IPython.display import clear_output

latent_space_distn = tf.random.normal
# tf.random.uniform
# tf.random.normal


def process_adam_from_config(config):
    kwargs = {
        'learning_rate': config.learning_rate,
        'beta_1': config.beta_1,
        'beta_2': config.beta_2,
        'clipnorm': config.clipnorm,
        'global_clipnorm': config.global_clipnorm,
        'use_ema': config.use_ema,
        'ema_momentum': config.ema_momentum,
        'ema_overwrite_frequency': config.ema_overwrite_frequency
    }
    return kwargs


def compile_dcgan(config, loss_fn=cross_entropy, nchannels=2):
    adam_kwargs = process_adam_from_config(config)
    d_optimizer = Adam(**adam_kwargs) # RMSprop(learning_rate=0) #
    g_optimizer = Adam(**adam_kwargs) # RMSprop(learning_rate=0)
    dcgan = DCGAN(config, nchannels=nchannels)
    dcgan.compile(d_optimizer=d_optimizer, g_optimizer=g_optimizer, loss_fn=loss_fn)
    return dcgan


# G(z)
def define_generator(config, nchannels=2):
    """
    >>> generator = define_generator()
    """
    z = tf.keras.Input(shape=(100))

    # First fully connected layer, 1 x 1 x 25600 -> 5 x 5 x 1024
    fc = layers.Dense(config['complexity_0'] * config['g_layers'][0])(z)
    fc = layers.Reshape((5, 5, int(config['complexity_0'] * config['g_layers'][0] / 25)))(fc)
    fc = layers.BatchNormalization(axis=-1)(fc)  # normalise along features layer (1024)
    lrelu0 = layers.LeakyReLU(config.lrelu)(fc)
    drop0 = layers.Dropout(config.dropout)(lrelu0)

    # Deconvolution, 7 x 7 x 512
    conv1 = layers.Conv2DTranspose(config['complexity_1'] * config['g_layers'][1], (3, 3), (1, 1), use_bias=False)(drop0)
    conv1 = layers.BatchNormalization(axis=-1)(conv1)
    lrelu1 = layers.LeakyReLU(config.lrelu)(conv1)
    drop1 = layers.Dropout(config.dropout)(lrelu1)

    # Deconvolution, 9 x 10 x 256
    conv2 = layers.Conv2DTranspose(config['complexity_2'] * config['g_layers'][2], (3, 4), (1, 1), use_bias=False)(drop1)
    conv2 = layers.BatchNormalization(axis=-1)(conv2)
    lrelu2 = layers.LeakyReLU(config.lrelu)(conv2)
    drop2 = layers.Dropout(config.dropout)(lrelu2)

    # Output layer, 20 x 24 x nchannels
    logits = layers.Conv2DTranspose(nchannels, (4, 6), (2, 2))(drop2)

    o = tf.keras.activations.sigmoid(logits)  # not done in original code but doesn't make sense not to

    return tf.keras.Model(z, o, name='generator')


# D(x)
def define_discriminator(config, nchannels=2):
    """
    >>> discriminator = define_discriminator()
    """
    x = tf.keras.Input(shape=(20, 24, nchannels))

    # 1st hidden layer 9x10x64
    conv1 = layers.Conv2D(config['d_layers'][0], (4,5), (2,2), 'valid', kernel_initializer=tf.keras.initializers.GlorotUniform())(x)
    lrelu1 = layers.LeakyReLU(config.lrelu)(conv1)
    drop1 = layers.Dropout(config.dropout)(lrelu1)

    # 2nd hidden layer 7x7x128
    conv1 = layers.Conv2D(config['d_layers'][1], (3,4), (1,1), 'valid', use_bias=False)(drop1)
    conv1 = layers.BatchNormalization(axis=-1)(conv1)
    lrelu2 = layers.LeakyReLU(config.lrelu)(conv1)
    drop2 = layers.Dropout(config.dropout)(lrelu2)

    # 3rd hidden layer 5x5x256
    conv2 = layers.Conv2D(config['d_layers'][2], (3,3), (1,1), 'valid', use_bias=False)(drop2)
    conv2 = layers.BatchNormalization(axis=-1)(conv2)
    lrelu3 = layers.LeakyReLU(config.lrelu)(conv2)
    drop3 = layers.Dropout(config.dropout)(lrelu3)

    # fully connected 1x1
    flat = layers.Reshape((-1, 5 * 5 * config['d_layers'][2]))(drop3)
    logits = layers.Dense(1)(flat)
    logits = layers.Reshape((1,))(logits)
    o = tf.keras.activations.sigmoid(logits)

    return tf.keras.Model(x, [o, logits], name='discriminator')


class DCGAN(keras.Model):
    def __init__(self, config, nchannels=2):
        super().__init__()
        self.discriminator = define_discriminator(config, nchannels)
        self.generator = define_generator(config, nchannels)
        self.latent_dim = config.latent_dims
        self.lambda_ = config.lambda_
        self.config = config
        self.trainable_vars = [*self.generator.trainable_variables, *self.discriminator.trainable_variables]
        self.d_loss_real_tracker = keras.metrics.Mean(name="d_loss_real")
        self.d_loss_fake_tracker = keras.metrics.Mean(name="d_loss_fake")
        self.g_loss_raw_tracker = keras.metrics.Mean(name="g_loss_raw")
        self.g_penalty_tracker = keras.metrics.Mean(name="g_penalty")


    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

        self.d_optimizer.build(self.discriminator.trainable_variables)
        self.g_optimizer.build(self.generator.trainable_variables)


    def call(self, nsamples=5):
        random_latent_vectors = latent_space_distn((nsamples, self.latent_dim))
        return self.generator(random_latent_vectors, training=False)

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        random_latent_vectors = latent_space_distn((batch_size, self.latent_dim))
        fake_data = self.generator(random_latent_vectors, training=False)
        labels_real = tf.ones((batch_size, 1)) * self.config.true_label_smooth
        labels_fake = tf.zeros((batch_size, 1))

        # train discriminator (twice)
        for iteration in range(self.config.training_balance):
            with tf.GradientTape() as tape:
                _, logits_real = self.discriminator(data)
                _, logits_fake = self.discriminator(fake_data)
                d_loss_real = self.loss_fn(labels_real, logits_real)
                d_loss_fake = self.loss_fn(labels_fake, logits_fake)
                d_loss = d_loss_real + d_loss_fake
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # sample random points in the latent space (again)
        random_latent_vectors = tf.random.normal((batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))  # i.e., want to trick discriminator

        # train the generator (don't update disciminator weights this time)
        with tf.GradientTape() as tape:
            generated_data = self.generator(random_latent_vectors)
            _, logits = self.discriminator(generated_data, training=False)
            g_loss_raw = tf.reduce_mean(self.loss_fn(misleading_labels, logits))
            g_penalty = self.lambda_ * get_chi_score(data, generated_data, sample_size=tf.constant(25))
            g_loss = g_loss_raw #+ g_penalty
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # update metrics and return their values
        self.d_loss_real_tracker.update_state(d_loss_real)
        self.d_loss_fake_tracker.update_state(d_loss_fake)
        self.g_loss_raw_tracker.update_state(g_loss_raw)
        self.g_penalty_tracker.update_state(g_penalty)

        return {
            "d_loss_real": self.d_loss_real_tracker.result(),
            "d_loss_fake": self.d_loss_fake_tracker.result(),
            "g_loss_raw": self.g_loss_raw_tracker.result(),
            "g_penalty": self.g_penalty_tracker.result()
        }


class ChiScore(Callback):
    """Custom metric for evtGAN to compare tail dependence (?) coefficients.

    Authors call this mean l2-norm but code is for RMSE (going with code).
    Works on single batch of data for now.
    """
    def __init__(self, validation_data: dict, sample_size=25, frequency=1):
        super().__init__()
        self.validation_data = validation_data
        self.sample_size = sample_size
        self.frequency = frequency
        for name in validation_data.keys():
            setattr(self, f'chi_score_{name}', None)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.frequency == 0:
            for name, data in self.validation_data.items():
                batch_size = tf.shape(data)[0]
                generated_data = self.model(batch_size)
                rmse = get_chi_score(data, generated_data, self.sample_size)
                setattr(self, f'chi_score_{name}', rmse)
                logs[f'chi_score_{name}'] = rmse


class Visualiser(Callback):
    def __init__(self, frequency=1):
        super().__init__()
        self.frequency = frequency
        self.generated_images = []

    def on_epoch_end(self, epoch, logs={}):
        if (epoch % self.frequency == 0) & (epoch > 0):
            clear_output(wait=True)
            fig, axs = plt.subplots(2, 3, figsize=(10, 6))
            generated_data = self.model(3)
            vmin = tf.reduce_min(generated_data)
            vmax = tf.reduce_max(generated_data)
            for i, ax in enumerate(axs[0, :]):
                ax.imshow(generated_data[i, ..., 0].numpy(), cmap="Spectral_r", vmin=vmin, vmax=vmax)
            for i, ax in enumerate(axs[1, :]):
                im = ax.imshow(generated_data[i, ..., 1].numpy(), cmap="Spectral_r", vmin=vmin, vmax=vmax)

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            fig.suptitle(f"Generated images for epoch: {epoch}")
            plt.show()


class CrossEntropy(Callback):
    def __init__(self, validation_data, wandb_run=None):
        super().__init__()
        self.validation_data = validation_data
        self.d_loss_real_test = None
        self.d_loss_fake_test = None
        self.g_loss_raw_test = None
        self.wandb_run = wandb_run

    def on_epoch_end(self, epoch, logs={}):
        batch_size = tf.shape(self.validation_data)[0]
        generated_data = self.model(batch_size)

        labels_real = tf.ones((batch_size, 1))
        labels_fake = tf.zeros((batch_size, 1))

        _, logits_real = self.model.discriminator(self.validation_data, training=False)
        _, logits_fake = self.model.discriminator(generated_data, training=False)

        d_loss_real_test = tf.reduce_mean(cross_entropy(labels_real, logits_real))
        d_loss_fake_test = tf.reduce_mean(cross_entropy(labels_fake, logits_fake))
        g_loss_raw_test = tf.reduce_mean(cross_entropy(labels_real, logits_fake))

        # updates
        self.d_loss_real_test = d_loss_real_test
        self.d_loss_fake_test = d_loss_fake_test
        self.g_loss_raw_test = g_loss_raw_test
        logs['d_loss_real_test'] = d_loss_real_test
        logs['d_loss_fake_test'] = d_loss_fake_test
        logs['g_loss_raw_test'] = g_loss_raw_test


@tf.function
def get_chi_score(data, generated_data, sample_size=tf.constant(25)):
    """Calculates chi-score between real and fake data for given sample size.

    >>> get_chi_score(data, generated_data, self.sample_size)
    """
    
    _, h, w, c = tf.unstack(tf.shape(data))
    sample_inds = tf.random.uniform([sample_size], maxval=(h * w), dtype=tf.dtypes.int32)
    rmses = tf.TensorArray(tf.float32, size=c)
    for i in tf.range(c):
        chi_real = chi(data[..., i], sample_inds)
        chi_fake = chi(generated_data[..., i], sample_inds)
        rmse = tf.math.sqrt(tf.reduce_mean((chi_real - chi_fake)**2))
        rmses = rmses.write(i, rmse)
    rmse = tf.reduce_mean(rmses.stack())
    return rmse


def chi(data, sample_inds):
    """Raw extremal correlation/tail dependence coefficient."""
    n, h, w, *_ = tf.unstack(tf.shape(data))
    sample_size = tf.shape(sample_inds)[0]

    data = tf.reshape(data, [n, h * w])
    data = tf.gather(data, sample_inds, axis=1)
    data = data * tf.constant(.999)  # can't have 1s in exponential quantile function
    exp = inv_unit_frechet(data)

    chis = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    counter = tf.constant(0)
    for i in tf.range(sample_size):
        for j in tf.range(i):
            exp_x = exp[:, i]
            exp_y = exp[:, j]
            chi = chi_ij(exp_x, exp_y)  # doesn't like being compiled in a @tf.function
            chis = chis.write(counter, chi)
            counter = counter + tf.constant(1)
    return chis.stack()


def inv_exp(uniform):
    """Exponential quantile function. TODO: generating NaNs need to fix."""
    tf.debugging.Assert(tf.reduce_all(uniform < 1), ['Cannot perform inverse exponential when tensor contains values exceeding 1.'])
    exp_distributed = -tf.math.log(1 - uniform)
    return exp_distributed


def inv_unit_frechet(uniform):
    """Inverse of unit Fréchet quantile function."""
    tf.debugging.Assert(tf.reduce_all(uniform < 1), ['Cannot perform inverse frechet when tensor contains values exceeding 1.'])
    inv_frechet_distributed = 1 / (-tf.math.log(uniform))
    return inv_frechet_distributed


def chi_ij(exp_x, exp_y):
    """Where x and y have been transformed to exponential distributions (inverted Fréchet).

    ..[1] Max-stable process and spatial extremes, Smith (1990).
    ..[2] Boulaguiem (2022)
    """
    n = tf.shape(exp_x)[0]
    tf.Assert(n == tf.shape(exp_y)[0], ["x and y different sizes."])
    n = tf.cast(n, dtype=tf.float32)
    minima = tf.math.minimum(exp_x, exp_y)
    if tf.equal(tf.reduce_sum(minima), 0):
        chi_ij = tf.constant(2.)  # complete independence case: see [1]]
    else:
        chi_ij = n / tf.reduce_sum(minima)
    return chi_ij


def uniform_transform(data):
    """Transform data to uniform distribution using its ecdf. Univariate only."""
    tf.Assert(tf.rank(data) != 4, ["Function only defined for univariate (single channel) data."])
    n, h, w, *_ = tf.shape(data)
    data = tf.reshape(data, [n, h * w])
    n = tf.cast(tf.shape(data)[1], dtype=tf.float32)
    uniform = tf.map_fn(lambda x: tfd.Empirical(data[:, x]).cdf(data[:, x]) * (n / (n + 1)), tf.range(tf.shape(data)[1]), dtype=tf.float32)
    return uniform
