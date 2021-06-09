import os
import requests
import logging
from Settings.settings import *
import numpy as np
import matplotlib.pyplot as plt

train_logger = logging.getLogger("dataset_utils")
train_logger.setLevel(LOG_LEVEL)


def download_file(url: str, file_path: str, buffer_size: int = 100000):
    """Download file at <url> to <file_path>.
    Parameters
    ----------
    url : str
        The url of the file.
    file_path : str
        The local absolute path of the file.
    buffer_size : int, optional
        The buffer size/chunk size in bytes of download,
        by default 100000.

    Returns
    -------
    str
        The absolute file path of the downloaded file.
    """
    train_logger.info(
        f"Start downloading file from url: {url}.")
    train_logger.info(f"Download path: {file_path}.")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=buffer_size):
                f.write(chunk)
    train_logger.info("File downloaded successfully!")
    return file_path


def download_CelebA(dir: str = DATASETS_DIR, filename: str = "CelebA.h5"):
    """Download the CelebA dataset.

    Parameters
    ----------
    dir : str, optional
        The absolute directory path to store the local file, by default
        <DATASETS_DIR>.
    filename : str, optional
        The file name of the dataset file, by default "CelebA.h5"

    Returns
    -------
    str
        The absolute file path of the dataset file.
    """
    # training dataset:
    # https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1
    celebA_url = "https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1"
    return download_file(celebA_url, os.path.join(dir, filename))


def generate_training_testing_indices(data_size: int,
                                      training_ratio: float = 0.8):
    """Randomly generated indices of training data and testing data.

    Parameters
    ----------
    data_size : int
        The dataset size.
    training_ratio : float, optional
        The ratio of training data size over all data size, must between 0
        and 1, by default 0.8.

    Returns
    -------
    tuple
        A tuple contains training indices and testing indices.
    """
    training_indices = generate_random_training_indices(data_size,
                                                        training_ratio)
    testing_indices = generate_testing_indices(data_size, training_indices)
    return training_indices, testing_indices


def generate_random_training_indices(data_size: int,
                                     training_ratio: float = 0.8):
    """Randomly generated sorted indices of the training data.

    Parameters
    ----------
    data_size : int
        The dataset size.
    training_ratio : float
        The ratio of training data size over all data size, must between 0
        and 1, by default 0.8.

    Returns
    -------
    list
        A 1-d list contains all training indices.
    """
    assert 0 < training_ratio < 1
    return np.sort(np.random.choice(data_size,
                                    int(data_size*training_ratio),
                                    replace=False)).tolist()


def generate_testing_indices(data_size: int, training_indices: list):
    """Generate sorted testing indices according to <training_indices>.

    Parameters
    ----------
    data_size : int
        the size of the entire dataset.
    training_indices : list
        A list contains all the training indices.

    Returns
    -------
    list
        A 1-d list contains all testing indices.
    """
    testing_indices = np.ones(data_size, dtype=bool)
    testing_indices[training_indices] = False
    return np.sort(np.where(testing_indices == True)[0]).tolist()


def split_dataset(x: np.ndarray, y: np.ndarray, training_indices: list):
    """Split the dataset into training/testing datasets according to the
    indices of the training indices <training_indices>.

    Parameters
    ----------
    x : ndarray
        Input data, with shape (dataset_size, ...).
    y : ndarray
        Output/label data, with shape (dataset_size, ...).
    train_idx : list
        A list contains all indices of training instances.

    Returns
    -------
    tuple
        A tuple of 2 tuples that contain (training_x, training_y) and
        (testing_x, testing_y), respectively.
    """
    assert x.shape[0] == y.shape[0]
    testing_indices = np.ones(x.shape[0], dtype=bool)
    testing_indices[training_indices] = False
    return (x[training_indices], y[training_indices]), \
        (x[testing_indices], y[testing_indices])


def visualize_images(images: np.ndarray, labels: np.ndarray = None,
                     *args, **kwargs):
    """Visualize 36 random samples from <images> in dataset.

    Parameters
    ----------
    images : np.ndarray
        Images to display.
    labels : np.ndarray
        Corresponding labels of images, 1 dimensional.
    """

    assert images.shape[0] >= 36

    if labels:
        labels = labels.astype(np.uint8)
    random_indices = np.random.choice(images.shape[0], 36, replace=False)
    plt.figure(figsize=(10, 10))
    for i in range(36):
        # plt.subplot index starts with 1
        plt.subplot(6, 6, i+1)
        # clear ticks
        plt.xticks([])
        plt.yticks([])
        # remove grids
        plt.grid(False)

        index = random_indices[i]
        plt.imshow(images[index], *args, **kwargs)
        if labels:
            plt.xlabel(np.squeeze(labels[index], axis=-1))
    plt.show()
