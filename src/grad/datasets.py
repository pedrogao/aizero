import os
import numpy as np
from grad.utils import get_file, cache_dir


class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), \
                self.target_transform(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass


# =============================================================================
# Toy datasets
# =============================================================================
def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int)

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix] = j
    # Shuffle
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t


class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)


# =============================================================================
# Big datasets
# =============================================================================
class ImageNet(Dataset):

    def __init__(self):
        NotImplemented

    @staticmethod
    def labels():
        url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
        path = get_file(url)
        with open(path, 'r') as f:
            labels = eval(f.read())
        return labels


# =============================================================================
# Sequential datasets: SinCurve, Shapekspare
# =============================================================================
class SinCurve(Dataset):

    def prepare(self):
        num_data = 1000
        dtype = np.float64

        x = np.linspace(0, 2 * np.pi, num_data)
        noise_range = (-0.05, 0.05)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)
        if self.train:
            y = np.sin(x) + noise
        else:
            y = np.cos(x)
        y = y.astype(dtype)
        self.data = y[:-1][:, np.newaxis]
        self.label = y[1:][:, np.newaxis]


class Shakespear(Dataset):

    def prepare(self):
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        file_name = 'shakespear.txt'
        path = get_file(url, file_name)
        with open(path, 'r') as f:
            data = f.read()
        chars = list(data)

        char_to_id = {}
        id_to_char = {}
        for word in data:
            if word not in char_to_id:
                new_id = len(char_to_id)
                char_to_id[word] = new_id
                id_to_char[new_id] = word

        indices = np.array([char_to_id[c] for c in chars])
        self.data = indices[:-1]
        self.label = indices[1:]
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char


# =============================================================================
# Utils
# =============================================================================
def load_cache_npz(filename, train=False):
    filename = filename[filename.rfind('/') + 1:]
    prefix = '.train.npz' if train else '.test.npz'
    filepath = os.path.join(cache_dir, filename + prefix)
    if not os.path.exists(filepath):
        return None, None

    loaded = np.load(filepath)
    return loaded['data'], loaded['label']


def save_cache_npz(data, label, filename, train=False):
    filename = filename[filename.rfind('/') + 1:]
    prefix = '.train.npz' if train else '.test.npz'
    filepath = os.path.join(cache_dir, filename + prefix)

    if os.path.exists(filepath):
        return

    print("Saving: " + filename + prefix)
    try:
        np.savez_compressed(filepath, data=data, label=label)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise
    print(" Done")
    return filepath
