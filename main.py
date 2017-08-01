import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import name_scope as name_scp

from Neurocat import Loading
from Neurocat import im2col2
from Neurocat import glab

from Neurocat import gen_layer, layerloop, gen_myst_layer

from Neurocat import Preprocessing

# horizont for time embedding
pasHor = 200
futHor = 1
infHor = futHor
stride = 1

# num of training epochs
train_epoch = futHor * 2500

# batch
batch = 1

# noise factor for training
noise_fac = 0.1

# number of mixtures
mixtures = 3

hidden = 4
out_units = mixtures * futHor * 3

# generate linear descending layer chain
layer = [pasHor] \
        + [(pasHor + i * (out_units - pasHor) // hidden)
           for i in range(hidden + 1)] \
        + [out_units]
print("Layer:\t\t", layer)

# length of the closing MLP
mlp = 3

# standard deviation for weight initialisation
std_dev = 0.5

# num of samples that should be generated
gen_epoch = futHor * 1000

# choose the training data
data = np.float32(np.loadtxt("./examples/sinus.txt")[None, :])

#assert len(data.shape == 2) and data.shape[0] == 1 and data.shape[1] > 0
print("data:\t\t", data.shape)

# preprocess the data
pp = Preprocessing(data.T)
data = pp.encode(data.T).T

# time embedding of the data
fx = np.array([1, pasHor])
fy = np.array([1, futHor])
s = np.array([1, stride])

print("filter:\t\t", fx, fy)
print("stride:\t\t", s)

x_data = im2col2(data[:, :-futHor], fx, s).T
y_data = im2col2(data[:, pasHor:], fy, s).T

print("x data:\t\t", x_data.shape)
print("y data:\t\t", y_data.shape)

# amount of training data
train_len = x_data.shape[0]

x = tf.placeholder(dtype=tf.float32, shape=[None, pasHor], name="input")
y = tf.placeholder(dtype=tf.float32, shape=[None, futHor], name="label")

with name_scp("filter"):
    with glab("mystery") as scope:
        shapes = [(layer[0], layer[0])] * 3
        myst = layerloop("mytery", x, shapes)
    out = tf.multiply(x, myst, name="conv")

# input layer
Wi, bi, out = gen_layer("input", out, (layer[0], layer[1]))

# TODO residual
# hierarchical convs
with name_scp("Hidden"):
    hierarchie = []
    for i in range(1, hidden + 1):
        out, hier = gen_myst_layer(out, (layer[i], layer[i + 1]))
        hierarchie += [hier]

with name_scp("MLP"):
    shapes = []
    for i in range(mlp):
        shapes += [(layer[-1], layer[-1])]
    out = layerloop("hidden", out, shapes)
    print("MLP:\t\t", shapes)

# output layer
Wo, bo, out = gen_layer("output", out, (layer[-1], layer[-1]), activation=None)

with name_scp("mixture_model"):
    pi, sigma, mu = tf.split(out, 3, axis=1, name="mixture")

new_shape = (-1, mixtures, futHor)

with name_scp("Pi"):
    pi = tf.reshape(pi, shape=new_shape, name="fit_horizont")
    max_pi = tf.reduce_max(pi, 1, keep_dims=True)
    pi = tf.subtract(pi, max_pi)
    pi = tf.exp(pi)
    norm = tf.reduce_sum(pi, 1, keep_dims=True)
    norm = tf.reciprocal(norm)
    pi = tf.multiply(norm, pi)

with name_scp("Sigma"):
    sigma = tf.reshape(sigma, shape=new_shape, name="fit_horizont")
    sigma = tf.exp(sigma, name="exp")

with name_scp("Mu"):
    mu = tf.reshape(mu, shape=new_shape, name="fit_horizont")

# normalisation factor for gaussian, not needed.
norm_fac = 1. / np.sqrt(2. * np.pi)
# make it feedable for tensorflow
gauss_norm = tf.constant(np.float32(norm_fac), name="Gaussnormalizer")
# don't forget to normalize over mixtures
mix_norm = tf.constant(np.float32(1 / (mixtures * futHor)), name="Mixturenorm")

# calculate the loss, see Bishop 1994
with name_scp("loss"):
    label = tf.reshape(y, shape=(-1, 1, futHor), name="horizont")
    with name_scp("normal"):
        normal = tf.subtract(label, mu)
        normal = tf.multiply(normal, tf.reciprocal(sigma))
        normal = -tf.square(normal)
        normal = tf.multiply(normal, tf.reciprocal(tf.constant(2.)))
        normal = tf.multiply(tf.exp(normal), tf.reciprocal(sigma))
        normal = tf.multiply(normal, gauss_norm)
    with name_scp("cond_average"):
        lossfunc = tf.multiply(normal, pi)
        lossfunc = tf.multiply(lossfunc, mix_norm)
        lossfunc = tf.reduce_sum(lossfunc, 2, keep_dims=True)
        lossfunc = tf.reduce_sum(lossfunc, 1)
    lossfunc = -tf.log(tf.multiply(lossfunc, mix_norm))
    lossfunc = tf.reduce_mean(lossfunc)

with name_scp("trainer"):
    train_op = tf.train.AdamOptimizer().minimize(lossfunc)

with name_scp("inference"):
    # find random index by the probability of pi
    with name_scp("random_pick"):
        samples = tf.multinomial(tf.log(tf.transpose(pi[0])), 1)

    # choose coresponding parameter for mixture
    with name_scp("Pi"):
        _pi = pi[0, :][tf.cast(samples[0][0], tf.int32)]
    with name_scp("Mu"):
        _mu = mu[0, :][tf.cast(samples[0][0], tf.int32)]
    with name_scp("Sigma"):
        _sig = sigma[0, :][tf.cast(samples[0][0], tf.int32)]

        # alternatively you can just choose the argmax of pi (below numpy code)
        # find argmax of pi such that you can choose optimal mu
        # argmax = np.argmax(_pi, axis=1)[0]

# train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    # toolkit for visual loading
    loader = Loading(train_epoch, 'Training the Network')
    # lossfunction
    loss = np.zeros(loader.treshold())
    for i in loader.range():
        # batching because input to large
        for b in range(batch):
            batch_len = train_len // batch
            noise = np.random.normal(size=(batch_len, futHor)) * noise_fac
            # training
            sess.run(train_op, feed_dict={
                x: x_data[
                   b * batch_len:(b + 1) * batch_len, :],
                y: np.add(
                    y_data[b * batch_len:(b + 1) * batch_len, :],
                    noise
                )
            })
        final_x = x_data[(batch - 1) * batch_len:batch * batch_len, :]
        final_y = y_data[(batch - 1) * batch_len:batch * batch_len, :]
        # loss
        loss[i], NaN_check = sess.run([lossfunc, mu], feed_dict={
            x: final_x,
            y: final_y
        })

        # Don't waste Time when gradients explode
        if np.any(np.isnan(NaN_check)):  exit(7)

        # update visual loading
        loader.loading(i)

    # generate input
    x_gen = x_data[0][None, :]
    # generate prediction
    prediction = np.empty(shape=(1, 0))
    # get confidence and variance of the prediction
    confidence = np.empty(shape=(1, 0))
    variance = np.empty(shape=(1, 0))
    # mystery
    mystery = np.empty(shape=(layer[0], 0))
    hierarch_viz = [np.empty(shape=(layer[i + 1], 0))
                    for i in range(len(hierarchie))]

    loader = Loading(gen_epoch + pasHor, 'Generating Autonomous Signal')
    while loader.in_progress(prediction.shape[1]):
        __pi, __sig, __mu, _myst, _hier = sess.run(
            [_pi, _sig, _mu, myst, hierarchie],
            feed_dict={x: x_gen})

        if np.any(np.isnan(__mu)):
            exit(7)
            break

        # append new information
        x_gen = np.concatenate((x_gen, __mu[None, :infHor]), axis=1)
        prediction = np.concatenate((prediction, __mu[None, :infHor]), axis=1)
        confidence = np.concatenate((confidence, __pi[None, :infHor]), axis=1)
        variance = np.concatenate((variance, __sig[None, :infHor]), axis=1)
        mystery = np.concatenate((mystery, _myst.T), axis=1)

        for i in range(len(hierarchie)):
            hierarch_viz[i] = np.concatenate((hierarch_viz[i], _hier[i].T),
                                             axis=1)

        # delete old input that is not needed anymore
        x_gen = np.delete(x_gen, list(range(infHor)), axis=1)

        loader.loading(prediction.shape[1])
writer.close()

# prepare data, optional, you can create new with size of y_aut
data = data[:, pasHor:]
# little trick that subtracts numpy arrays of possibly different length
error = np.subtract(data[:, :prediction.shape[1]],
                    prediction[:, :data.shape[1]])

# plot
fig = plt.figure(1)
ax1 = fig.add_subplot(221)
ax1.plot(data.T, 'g-')
ax1.plot(confidence.T, 'r--')
ax1.plot(variance.T, 'y--')
ax1.plot(prediction.T, 'b-')
ax1.set_title("Autonomous")
ax1.grid(True)

ax2 = fig.add_subplot(224)
ax2.plot(loss, 'r-')
ax2.set_title("errorrate MDN")
ax2.grid(True)

ax3 = fig.add_subplot(223, sharex=ax1)
ax3.plot(error.T, 'r-')
ax3.set_title("Difference")
ax3.grid(True)

ax4 = fig.add_subplot(222)
ax4.imshow(mystery, aspect='auto', interpolation='nearest')
ax4.set_title("Hierarchie")
ax4.grid(True)

plt.suptitle('mixtures: %d, '
             'layer: %d, '
             'training: %d, '
             'past horizont: %d, '
             'future horizont: %d, '
             'inference horizont: %d, '
             'noise %.2f' %
             (mixtures, len(layer), train_epoch, pasHor,
              futHor, infHor, noise_fac))
plt.show()

np.savetxt("auto.np", pp.decode(prediction.T))

# visualize layerwise hierarchie
fig, axs = plt.subplots(1, len(hierarch_viz))
axs = axs.ravel()
for i in range(len(axs)):
    axs[i].imshow(hierarch_viz[i], aspect='auto', interpolation='nearest')
    axs[i].set_title("Hierarchie")
    axs[i].grid(True)
plt.show()
