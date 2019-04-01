import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
# ops.reset_default_graph()
x_range = 8
hidden_size = 4
class_num = 1
feature_dim = 1
batch_size = 8
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


def plot_3(session, data, sample_range=x_range, num_points=10000, num_bins=100):
    xs = np.linspace(-sample_range, sample_range, num_points)
    bins = np.linspace(-sample_range, sample_range, num_bins)

    # decision boundary
    db = session.run(D1, {x: np.reshape(xs, (-1, 1))})

    # data distribution
    d = data.sample(num_points)
    pd, _ = np.histogram(d, bins=bins, density=True)

    # generated samples
    g = session.run(G, {z: np.reshape(xs, (-1, 1))})
    pg, _ = np.histogram(g, bins=bins, density=True)

    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))
    f, ax = plt.subplots(1)
    ax.plot(db_x, db, label='decision boundary')
    ax.set_ylim(0, 1)
    plt.plot(p_x, pd, label='real data')
    plt.plot(p_x, pg, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()


def log(x):
    '''
    Sometimes discriminiator outputs can reach values close to
    (or even slightly less than) zero due to numerical rounding.
    This just makes sure that we exclude those values so that we don't
    end up with NaNs during optimisation.
    '''
    return tf.log(tf.maximum(x, 1e-5))


def linear(input, output_dim, scope=None, stddev=1.0):
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input, w) + b


def generator(input, h_dim=hidden_size):
    with tf.variable_scope('G'):
        h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
        h1 = linear(h0, 1, 'g1')
    return h1


def discriminator(input, h_dim=hidden_size, reuse_=False):
    with tf.variable_scope('D', reuse=reuse_):
        h0 = tf.nn.relu(linear(input, h_dim * 2, 'd0'))
        h1 = tf.nn.relu(linear(h0, h_dim * 2, 'd1'))
        h2 = tf.nn.relu(linear(h1, h_dim * 2, scope='d2'))
        h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
    return h3


def optimizer(loss, var_list):
    # print(var_list)
    learning_rate = 0.001
    step = tf.Variable(0, trainable=False)
    optimizer_ = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=step,
        var_list=var_list
    )
    return optimizer_


class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GeneratorDistribution(object):
    def __init__(self, range = x_range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01


z = tf.placeholder(dtype=tf.float32, shape=[None, feature_dim])
# x = tf.placeholder(dtype=tf.float32, shape=[None, feature_dim])
# 为什么这样写是错的！！！！！
G = generator(z)
x = tf.placeholder(dtype=tf.float32, shape=[None, feature_dim])
D2 = discriminator(G)

D1 = discriminator(x, reuse_=True)
d_loss = tf.reduce_mean(-log(D1)-log(1-D2))
d_acc = (tf.reduce_mean(D1)+(1-tf.reduce_mean(D2)))/2
g_loss = tf.reduce_mean(-log(D2))
vars = tf.trainable_variables()
d_params = [v for v in vars if v.name.startswith('D/')]
g_params = [v for v in vars if v.name.startswith('G/')]

# pre_opt = tf.train.AdamOptimizer(0.001).minimize(pre_loss)
d_opt = optimizer(d_loss, d_params)
g_opt = optimizer(g_loss, g_params)

data = DataDistribution()
gen = GeneratorDistribution()


with tf.Session() as session:
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    for i in range(5000+1):
        xd = data.sample(batch_size)
        zd = gen.sample(batch_size)
        d_l, _ = session.run([d_loss, d_opt], {
            x: np.reshape(xd, (batch_size, 1)),
            z: np.reshape(zd, (batch_size, 1))
        })
        # update generator
        zd = gen.sample(batch_size)
        g_l, _ = session.run([g_loss, g_opt], {
            z: np.reshape(zd, (batch_size, 1))
        })
        if (i+1) % 100 == 0:
            session.run([G, D1, D2], {
                x: np.reshape(xd, (batch_size, 1)),
                z: np.reshape(zd, (batch_size, 1))
            })
            print("Generation {} d_acc:{}  g_loss:{}".format(i, d_l, g_l))
    plot_3(session, data)







