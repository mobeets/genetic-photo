import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def get_model(batch_size, original_dim, intermediate_dims, optimizer):
    x = Input(batch_shape=(batch_size, original_dim), name='x')
    hprev = x
    for i,intermediate_dim in enumerate(intermediate_dims):
        h = Dense(intermediate_dim, activation='relu', name='h{}'.format(i))(hprev)
        hprev = h
    yhat = Dense(1, activation='sigmoid', name='yhat')(h)

    mdl = Model(x, yhat)
    mdl.compile(optimizer=optimizer, loss='mean_squared_error')
    return mdl

def maximize_model_output(model, X0, nsteps, stepsize=1e-2):
    """
    src: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
    """
    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

    input_img = model.input
    loss = model.output
    grads = K.gradients(loss, input_img)[0]
    grads = normalize(grads)
    iterate = K.function([input_img], [loss, grads])
    Xh = X0.copy()
    # run gradient ascent on model to maximize f(Xh)
    for i in range(nsteps):
        loss_value, grads_value = iterate([Xh])
        Xh += grads_value * stepsize
    return Xh

def get_test_data(N, K, batch_size):
    # generate example data
    x1 = np.linspace(-1, 1, N)
    x = np.meshgrid(*[x1 for i in range(K)])
    X = np.vstack([xi.flatten() for xi in x]).T
    n = batch_size*int(X.shape[0]/batch_size)
    return X[:n]

def plot_3d_scatter(X, y, outfile, do_mesh=False, n_reshape=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if do_mesh:
        N = int(np.sqrt(len(y)))
        X1 = X[:,0].reshape(N, N)
        X2 = X[:,1].reshape(N, N)
        Y = y.reshape(N, N)
        ax.plot_surface(X1, X2, Y, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
    else:
        ax.scatter(X[:,0], X[:,1], y, 'k.', s=1)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    ax.set_zlim(0, 1)
    if outfile is None:
        plt.show()
    else:
        fig.savefig(outfile)
        plt.close(fig)

def plot_y_vs_yh(y, yh, outfile):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(y, yh, marker='.', c='k', s=1)
    ax.plot([0, 0], [1, 1], 'k-')
    plt.xlim(0, 1.)
    plt.ylim(0, 1.)
    if outfile is None:
        plt.show()
    else:
        fig.savefig(outfile)
        plt.close(fig)

def update_population(X, y, K, mdl, num_epochs, batch_size, nsteps, stepsize, tol=1e-4):
    # update fitness function estaimte
    history = mdl.fit(X, y,
        shuffle=True,
        epochs=num_epochs,
        verbose=0,
        batch_size=batch_size)

    # update X by maximizing fitness function estimate
    Xh = maximize_model_output(mdl, X, nsteps, stepsize)

    # keep in bounds
    Xh[Xh < -1.] = -1.
    Xh[Xh > 1.] = 1.

    # new population is Xh we chose; but if it's already in X, choose random mutation
    Xhs = init_population(Xh.shape[0], K) # random mutations
    for i, x in enumerate(Xh):
        ix1 = (np.square(X - x).sum(axis=1) < tol) # already in X
        ix2 = (np.square(Xhs - x).sum(axis=1) < tol) # already in Xhs
        if not (ix1 | ix2).any():
            # overwrite random mutation
            Xhs[i] = x
        # n.b., if x doesn't increase actual evaluation, keep parent X[i]
    return Xhs

def init_population(N, K):
    return 2*(np.random.rand(N, K) - 0.5)

def update_history(X, y, x_max, y_max, init=False):
    if y.max() > y_max:
        y_max_ind = np.argmax(y)
        x_max = X[y_max_ind]
        y_max = y[y_max_ind]
    name = 'True' if init else 'Best estimate of'
    x_max_str = '(' + ', '.join(['{:0.3f}'.format(x) for x in x_max]) + ')'
    print(("{} maximum is at {}: {:0.3f}").format(name, x_max_str, y_max))
    return x_max, y_max

def _eval_fcn(X, K):
    beta = [1, 5, 3, -2, 4]
    offset = [0, 1, 0, -1, 3]
    beta2 = [7, 0, 3, -1, 0]
    offset2 = [0, 0, 1, -1, 0]
    y = np.prod(np.cos(beta[:K]*X + offset[:K]), axis=1)
    y += np.sin(beta2[:K]*X + offset2[:K]).sum(axis=1)
    y = y*multivariate_normal.pdf(X, mean=np.zeros(K))
    y = y - y.min() + 0.1 # keep y above 0
    return y

def make_eval_fcn(K):
    assert(K <= 5)
    return lambda X: _eval_fcn(X, K=K)

def fit(K=2, num_gens=4, batch_size=100, dims=[16, 32], plot_every=1, num_epochs=1000, nsteps=1, stepsize=1e-1, outfile='logs/nn/{}.png', optimizer='adam'):
    eval_fcn = make_eval_fcn(K)

    # show true function
    X0 = get_test_data(N=np.power(batch_size*batch_size, 1./K), K=K, batch_size=batch_size)
    y = eval_fcn(X0)
    update_history(X0, y, 0., -np.inf, init=True)
    plot_3d_scatter(X0, y, outfile.format('_true'), do_mesh=K==2)

    # init population
    mdl = get_model(batch_size, K, dims, optimizer)
    # X = init_population(N=batch_size, K=K)
    X = 0.5*np.random.rand(batch_size, K) + 0.4
    y = eval_fcn(X)
    x_max, y_max = update_history(X, y, 0., -np.inf)

    # evolve
    print("Updating fitness model given {} cells...".format(len(y)))
    for i in range(num_gens):
        X = update_population(X, y, K, mdl, num_epochs, batch_size, nsteps, stepsize)
        y = eval_fcn(X)
        x_max, y_max = update_history(X, y, x_max, y_max)
        if i % plot_every == 0:
            yh = mdl.predict(X, batch_size=batch_size)
            yh_all = mdl.predict(X0, batch_size=batch_size)
            plot_3d_scatter(X, yh, outfile.format('prediction_cur_e' + str(i)))
            plot_3d_scatter(X0, yh_all, outfile.format('prediction_tot_e' + str(i)), do_mesh=K==2)
            plot_y_vs_yh(y, yh, outfile.format('compare_e' + str(i)))

    # show fits
    yh = mdl.predict(X, batch_size=batch_size)
    yh_all = mdl.predict(X0, batch_size=batch_size)
    plot_3d_scatter(X, yh, outfile.format('_prediction_cur'))
    plot_3d_scatter(X0, yh_all, outfile.format('_prediction_tot'), do_mesh=K==2)
    plot_y_vs_yh(y, yh, outfile.format('_compare'))

if __name__ == '__main__':
    fit()
