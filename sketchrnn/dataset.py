import math
import numpy as np
import tensorflow as tf


def cleanup(d, limit=1000):
    return d.astype(np.float32).clip(-limit, limit)


def calc_scale_factor(data):
    return np.std(np.concatenate([d[:, :2] for d in data]))


def normalize(data, scale_factor):
    data = data.copy()
    data[:, :2] /= scale_factor
    return data


def random_scale(data, eps):
    res = data.copy()
    res[:, :2] *= np.random.uniform(1 - eps, 1 + eps, [2])
    return res


def augment(data, prob):
    res = data.copy()
    n = res.shape[0]
    mask = np.ones([n], dtype=np.bool)
    rand = np.random.uniform(size=[n]) < prob
    count, last = 0, 0
    for i, d in enumerate(res):
        if d[2] == 0:
            if count > 2 and rand[i]:
                mask[i] = False
                res[last, :2] += d[:2]
            else:
                last = i
                count += 1
        else:
            last = i
            count = 0
    return res[mask]


def pad(data, seq_len):
    l = len(data)
    res = np.pad(data, [(1, seq_len - l), (0, 2)], "constant")
    res[0, :] = [0, 0, 1, 0, 0]
    res[1 : l + 1, 3] = data[:, 2]
    res[1 : l + 1, 2] = 1 - res[1 : l + 1, 3]
    res[l + 1 :, 4] = 1
    return res


def preprocess(d, seq_len, eps, prob):
    d = random_scale(d, eps)
    d = augment(d, prob)
    return pad(d, seq_len)


def data_gen(data, scale_factor):
    source = [normalize(d, scale_factor) for d in data]

    def gen(batch_size, seq_len, eps=0.15, prob=0.1):
        n = len(source)
        num_batches = math.ceil(n / batch_size)
        for b in range(num_batches):
            indices = np.random.permutation(range(n))[:batch_size]
            yield [preprocess(source[i], seq_len, eps, prob) for i in indices]

    return gen


def split(x):
    enc_in = target = x[:, 1:]
    dec_in = x[:, :-1]
    return (enc_in, dec_in), target


def make_train_dataset(
    data, seq_len, batch_size, scale_factor, random_scale_factor=0.1, augment_prob=0.1
):
    train_dataset = tf.data.Dataset.from_generator(
        data_gen(data, scale_factor),
        output_types=tf.float32,
        output_shapes=(None, seq_len + 1, 5),
        args=[batch_size, seq_len, random_scale_factor, augment_prob],
    )
    train_dataset = train_dataset.map(
        split, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return train_dataset


def make_val_dataset(data, seq_len, batch_size, scale_factor):
    val_dataset = tf.data.Dataset.from_tensor_slices(
        [pad(normalize(d, scale_factor), seq_len) for d in data]
    )
    val_dataset = val_dataset.batch(batch_size).map(split)
    return val_dataset
