import sys
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow import name_scope as glab


class Loading:
    """
    Visualisation of the loading process in the terminal. Gives information
    of the state of the process and trivially approximates remaining time left.
    """

    def __init__(self, nepoch, label=""):
        """
        Member of a loading process.
        :param nepoch: Total number of calculations stages
        :param label: name of calculation process
        """
        self.nepoch = nepoch
        self.lastepoch = 0
        self.elapsed = time.time()
        if label != "":
            print("\n" + label + "\n" + "#" * len(label))
        else:
            print()
        print("Loading %d Epochs" % self.nepoch)

    def loading(self, counter):
        """
        Give object information of its state. In the initialisation the
        total amount of processes is given.
        :param counter: state of calculation process.
        :return:
        """
        if self.lastepoch != 79 * counter // self.nepoch:
            lap = time.time() - self.elapsed
            self.elapsed = time.time()
            self.lastepoch = 79 * counter // self.nepoch
            sys.stdout.write("\r" +
                             "Approx. time remaining: %.2f minutes" % (
                                 lap * (79 - self.lastepoch) / 60))
            sys.stdout.flush()

    def treshold(self):
        """Total number of calculation stages"""
        return self.nepoch

    def range(self):
        """Iterable of calculation for e.g. FOR loop etc."""
        return range(self.nepoch)

    def in_progress(self, counter):
        """Checks whether calculation finished"""
        return counter < self.nepoch


class Summary:
    """
    Class for easy generation of tensorboard summaries.
    """
    RUNS = 1

    @classmethod
    def calc_runs(cls):
        cls.RUNS = 1
        while os.path.exists("./graphs/%d" % cls.RUNS):
            cls.RUNS += 1

    @classmethod
    def folder(cls):
        return "./graphs/%d" % cls.RUNS


def im2col(a, f, s):
    """
    A filter is a submatrix of the matrix 'a', containing a specified (by
    height and width of the filter) neighborhood of a matrix element
    m_{i,j} of matrix 'a'. Meaning: f_{k,l}=(i+k, j+l), respecting the
    matrix dimensions. In this scenario we go through every line of the
    matrix a with stepsize s. After finishing one line, we do the same
    procedure for the next line respecting the stepsize starting at
    (i,j)=(0,0).
    Assuming our matrix fits the filter and stepsize.
    :param a: matrix
    :param f: filtersize (filter height, filter width)
    :param s: stride
    :return: im2col matrix
    """
    # dimensions of Matrix
    m, n = a.shape
    # number of possible filters that fit in a row/column
    row_extend = (m - f[0] + 1) / s
    col_extend = (n - f[1] + 1) / s

    # start indices of a filter per row (left up)
    start_idx = np.arange(row_extend)[:, None] * n + np.arange(col_extend) * s

    # remaining indices for each filter
    off_idx = np.arange(f[0])[:, None] * n + np.arange(f[1])

    # get the indices we want to have when flatten the matrix
    return np.take(a, (off_idx.ravel()[:, None] + start_idx.ravel()).
                   astype(np.int64))


def im2col2(a, f, s):
    """
    Stride through matrix a with filter of size f and stride s. Get
    corresponding im2col matrix.

    :param a: matrix
    :param f: filtersize
    :param s: stride
    :return: im2col matrix
    """

    # dimensions of Matrix
    shape = np.array(a.shape)
    # number of possible filters that fit in a row/column
    extend = (shape - f + 1) / s

    # start indices of a filter per row (left up)

    start_idx = np.arange(extend[0])[:, None] * shape[1] * s[0] + \
                np.arange(extend[1]) * s[1]

    # remaining indices for each filter
    off_idx = np.arange(f[0])[:, None] * shape[1] + np.arange(f[1])

    # get the indices we want to have when flatten the matrix
    return np.take(a, (off_idx.ravel()[:, None] + start_idx.ravel()).
                   astype(np.int64))


def gen_layer(name, input, shape, dev_w=0.5, dev_b=0.5,
              activation=tf.nn.sigmoid, res=None):
    """
    Generate tensorflow layer.
    :param name: name for graph visualization
    :param input: input of the layer
    :param shape: Size of input and output vector
    :param dev_w: std deviation of initializing of the weights
    :param dev_b: std deviation of initializing of the bias
    :param activation: tensorflow activation function. None for no activation
    :param res: residual term from past layers
    :return: weights, bias and node of the output for computation graph
    """
    with tf.name_scope(name) as scope:
        W = tf.Variable(
            tf.random_normal([shape[0], shape[1]],
                             stddev=dev_w, dtype=tf.float32), name="W")
        b = tf.Variable(
            tf.random_normal([1, shape[1]],
                             stddev=dev_b, dtype=tf.float32), name="b")

        out = tf.matmul(input, W) + b
        if activation is None:
            pass
        else:
            out = activation(out)

        # residual
        if res is not None:
            out += res

            # tf.summary.histogram("weights", W)
            # tf.summary.histogram("bias", b)

    return W, b, out


def layerloop(name, input, shapes):
    """
    Create multiple layers that are connected to each other
    :param name: name for graph visualization
    :param input: input of the first layer
    :param shapes: input and output shape of all layers
    :return: node of the output for computation graph
    """
    if len(shapes) == 0:
        return input
    else:
        Wi, bi, out = gen_layer(name, input, shapes[0])
        return layerloop(name, out, shapes[1:])


def gen_myst_layer(input, shape, std_dev=0.5):
    """
    Create hierarchical (mystery) layer that filters the outcome of the
    output of the previous layer
    :param input: input to the layer that has to be filtere
    :param shape:
    :param std_dev:
    :return: Output node for calculation graph and filter
    """
    half = shape[0] // 2
    with glab("h_hidden") as scope:
        inp = input
        shapes = [(shape[0], shape[0]),
                  (shape[0], half),
                  (half, shape[0]),
                  (shape[0], shape[0])]
        with glab("filter") as scope:
            input = layerloop("mytery", input, shapes)
        # TODO: Residual connection
            filter = input
            input = tf.multiply(inp, input, name="conv")
        Wh, bh, out = gen_layer("hidden", input, (shape[0], shape[1]))

        return out, filter


def generate_ensemble(pi, mu, sigma, amount=1):
    """
    Generate ensemble from gaussian mixtures.

    :param pi: weight of the mixtures
    :param mu: mean of the mixtures
    :param sigma: deviation of the mixtures
    :param amount: amount of random variables
    :return: Random variable from mixture model
    """
    # standard uniform random variable
    uniform = np.random.rand(amount)[:, None]
    # standard normal random variable
    normal = np.random.randn(amount)[:, None]

    # add up probabilities till 1
    pdf = np.cumsum(pi)
    # find random weight
    idx = np.argmax(uniform <= pdf, axis=1)
    # choose corresponding parameters
    mu = np.choose(idx, mu)[:, None]
    std = np.choose(idx, sigma)[:, None]

    # convert std gaussian to corresponding mixture
    uniform = mu + normal * std

    return uniform


class Preprocessing():
    """
    Encoding and decoding for 2D numpy array of data of the form (samples,
    dimension). i.e. zerocenter and normalize the data.
    """

    def __init__(self, data):
        min = np.min(data, axis=0)
        max = np.max(data, axis=0)

        self.spread = (max - min) / 2
        self.diff = np.max(data/self.spread, axis=0) - 1

        print("spread:\t\t", self.spread)
        print("diff:\t\t", self.diff)

    def encode(self, data):
        data /= self.spread
        data -= self.diff
        return data

    def decode(self, data):
        data += self.diff
        data *= self.spread
        return data
