from GAN_MNIST.gan_mnist import GAN_MNIST
from Settings.settings import *
from Utils.dataset_utils import *


if __name__ == "__main__":
    noise_dim = 100
    gan_mnist = GAN_MNIST(noise_dim)
    gan_mnist.load_weights(CHECKPOINT_PREFIX)
    
    # generate 36 new images.
    latent_inputs = tf.random.normal((36, noise_dim))
    generated_images = np.clip(gan_mnist.generate_images(latent_inputs), 0, 1)
    
    visualize_images(generated_images, cmap="gray")
