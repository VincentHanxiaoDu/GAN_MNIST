from GAN_MNIST.gan_mnist import GAN_MNIST
from Settings.settings import *
from Utils.dataset_utils import *
import numpy as np
from tqdm import tqdm


def train_model(gan_mnist: GAN_MNIST, train_images, batch_size=256, epochs=50):
    train_logger = logging.getLogger("Train")
    train_logger.setLevel(LOG_LEVEL)

    N = train_images.shape[0]
    train_logger.info(f"Training on {N} samples.")
    for epoch in range(epochs):
        train_logger.info(f"Starting epoch {epoch+1}/{epochs}")
        gen_loss_history = []
        disc_loss_history = []
        for index_start in tqdm(range(0, N, batch_size)):
            index_end = min(N+1, index_start + batch_size)
            batched_images = train_images[index_start:index_end]
            gen_loss, disc_loss = gan_mnist.train_step(batched_images)
            gen_loss_history.append(gen_loss)
            disc_loss_history.append(disc_loss_history)
        train_logger.info(
            f"Average generator loss of the epoch: {np.array(gen_loss).mean()}")
        train_logger.info(
            f"Average discriminator loss of the epoch: {np.array(disc_loss).mean()}")
        gan_mnist.save_weights(CHECKPOINT_PREFIX)
    train_logger.info("Model trained successfully!")


if __name__ == "__main__":
    # Load MNIST train images
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    
    # preprocessing
    train_images = tf.expand_dims(
        train_images, axis=-1).numpy().astype(np.float32)/255.

    # visualize_images(train_images[0:36], cmap="gray")

    # Build model
    latent_dim = 100
    gan_mnist = GAN_MNIST(latent_dim)

    # Train
    train_model(gan_mnist, train_images)
