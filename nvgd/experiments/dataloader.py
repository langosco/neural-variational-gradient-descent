from itertools import cycle
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
import numpy as onp

from nvgd.experiments import config as cfg



print("Loading data...")
# Load MNIST
def load_data(dataset="mnist"):
    """
    dataset will be passed to tfds
    """
    data, info = tfds.load(name=dataset,
                           batch_size=-1,
                           data_dir=cfg.data_dir,
                           with_info=True)
    data = tfds.as_numpy(data)
    train_data, test_data = data['train'], data['test']

    # Full train and test set
    train_images, train_labels = train_data['image'], train_data['label']
    test_images, test_labels = test_data['image'], test_data['label']

    def normalize(images):
        """normalize according to training set statistics"""
        data_mean = onp.mean(train_images)
        data_stddev = onp.std(train_images)
        return (images - data_mean) / data_stddev

    # Normalize
    train_images = normalize(train_images)
    test_images = normalize(test_images)

    # Split off the validation set
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.1, random_state=0)

    train_data_size = len(train_images)

    return (train_images, train_labels), (val_images, val_labels), \
        (test_images, test_labels), train_data_size, train_images


def _make_batches(images, labels, batch_size, cyclic=True):
    """Returns an iterator through tuples (image_batch, label_batch).
    if cyclic, then the iterator cycles back after exhausting the batches"""
    num_batches = len(images) // batch_size
    split_idx = onp.arange(1, num_batches+1)*batch_size
    batches = zip(*[onp.split(data, split_idx, axis=0) for data in (images, labels)])
    return cycle(batches) if cyclic else list(batches)


# batches as array so we can jax.lax.map over them
def convert_to_array(batches):
    """
    args:
        batches: list of tuples (images, labels),
            where images, labels are batches.
    """
    batches = list(zip(*batches[:-1]))  # [image_batches, label_batches]
    return [onp.asarray(bs) for bs in batches]


class Data():
    def __init__(self, dataset="mnist", batch_size=cfg.batch_size):
        train, val, test, self.train_data_size, self.train_images = load_data(dataset)
        self.train_batches = _make_batches(*train, batch_size)
        self.val_batches = _make_batches(*val, batch_size, cyclic=False)
        self.test_batches = _make_batches(*test, batch_size, cyclic=False)

        self.test_batches_arr = convert_to_array(self.test_batches)
        self.val_batches_arr = convert_to_array(self.val_batches)

        self.steps_per_epoch = self.train_data_size // cfg.batch_size


data = Data(cfg.dataset)
# mnist = Data("mnist")
# cifar10 = Data("cifar10")
