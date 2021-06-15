import tensorflow as tf
import functools
from tensorflow.keras.losses import BinaryCrossentropy


def make_generator_model():
    Dense = tf.keras.layers.Dense
    LeakyReLU = tf.keras.layers.LeakyReLU
    BatchNormalization = tf.keras.layers.BatchNormalization
    Reshape = tf.keras.layers.Reshape
    Conv2DTranspose = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="same")

    base_num_filters = 64

    generator = tf.keras.Sequential([
        Dense(7*7*256),
        BatchNormalization(),
        LeakyReLU(),

        Reshape((7, 7, 256)),

        Conv2DTranspose(2*base_num_filters, (5, 5), strides=(2, 2)),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(base_num_filters, (5, 5), strides=(2, 2)),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(1, (3, 3), strides=(1, 1),
                        activation=tf.keras.activations.tanh),
    ])

    return generator


def make_discriminator_model():
    Dense = tf.keras.layers.Dense
    Conv2D = functools.partial(tf.keras.layers.Conv2D, padding="same")
    LeakyReLU = tf.keras.layers.LeakyReLU
    Dropout = tf.keras.layers.Dropout
    Flatten = tf.keras.layers.Flatten

    base_num_filters = 64

    discriminator = tf.keras.Sequential([
        Conv2D(base_num_filters, (5, 5), strides=(2, 2)),
        LeakyReLU(),
        Dropout(0.2),

        Conv2D(2*base_num_filters, (5, 5), strides=(2, 2)),
        LeakyReLU(),
        Dropout(0.2),

        Conv2D(4*base_num_filters, (3, 3), strides=(2, 2)),
        LeakyReLU(),
        Dropout(0.2),

        Flatten(),
        Dense(1, activation=tf.keras.activations.sigmoid)
    ])

    return discriminator


binary_crossentropy = BinaryCrossentropy(from_logits=False)


def discriminator_loss(real_pred, fake_pred):
    real_loss = binary_crossentropy(tf.ones_like(real_pred), real_pred)
    fake_loss = binary_crossentropy(tf.zeros_like(fake_pred), fake_pred)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_pred):
    return binary_crossentropy(tf.ones_like(fake_pred), fake_pred)


class GAN_MNIST(tf.keras.Model):
    def __init__(self, input_noise_dim):
        super().__init__()
        self.input_noise_dim = input_noise_dim
        self.generator = make_generator_model()
        self.discriminator = make_discriminator_model()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(self, real_batched_images):
        batch_size = real_batched_images.shape[0]
        input_noise = tf.random.normal((batch_size, self.input_noise_dim))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(input_noise, training=True)
            real_pred = self.discriminator(real_batched_images, training=True)
            fake_pred = self.discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_pred)
            disc_loss = discriminator_loss(real_pred, fake_pred)

        grads_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        grads_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(grads_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(grads_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def generate_images(self, noise_inputs):
        return self.generator(noise_inputs, training=False)
