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
        Xh -= grads_value * stepsize
    return Xh

def eval_fcn(X):
    y = np.cos(X[:,0])*np.cos(5*X[:,1] + 1) + np.sin(7*X[:,0])
    y = y*multivariate_normal.pdf(X, mean=[0,0]) + 0.35
    return y
    
def get_test_data(batch_size=100, K=2):
    # generate example data
    assert(K==2)
    x1 = np.linspace(-1, 1, batch_size)
    xv, yv = np.meshgrid(x1, x1)
    X = np.vstack([xv.flatten(), yv.flatten()]).T

    return X, eval_fcn(X)

def plot_3d_scatter(X, y, outfile, do_mesh=False):
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

def init_population(N=2000, K=2):
    return 2*(np.random.rand(N, K) - 0.5)

def update_population(X, y, mdl, num_epochs, batch_size, nsteps, stepsize):
    history = mdl.fit(X, y,
        shuffle=True,
        epochs=num_epochs,
        verbose=0,
        batch_size=batch_size)
    # update X
    Xh = maximize_model_output(mdl, X, nsteps, stepsize)
    return Xh

def fit(num_gens=4, batch_size=100, dims=[16, 32], update_every=1, num_epochs=1000, nsteps=1, stepsize=1e-1, outfile='logs/nn/{}.png', optimizer='adam'):
    K = 2

    # show true function
    X0, y = get_test_data(batch_size=batch_size, K=K)
    plot_3d_scatter(X0, y, outfile.format('_true'), do_mesh=True)

    # evolve
    mdl = get_model(batch_size, K, dims, optimizer)
    X = init_population(N=batch_size, K=K)
    y = eval_fcn(X)
    print("Updating fitness model given {} cells...".format(len(y)))
    for i in range(num_gens):
        X = update_population(X, y, mdl, num_epochs, batch_size, nsteps, stepsize)
        if i % update_every == 0:
            yh = mdl.predict(X, batch_size=X.shape[0])
            yh_all = mdl.predict(X0, batch_size=X.shape[0])
            plot_3d_scatter(X, yh, outfile.format('prediction_cur_e' + str(i)))
            plot_3d_scatter(X0, yh_all, outfile.format('prediction_tot_e' + str(i)), do_mesh=True)
            plot_y_vs_yh(y, yh, outfile.format('compare_e' + str(i)))

    # show fits
    yh = mdl.predict(X, batch_size=X.shape[0])
    yh_all = mdl.predict(X0, batch_size=X.shape[0])
    plot_3d_scatter(X, yh, outfile.format('_prediction_cur'))
    plot_3d_scatter(X0, yh_all, outfile.format('_prediction_tot'), do_mesh=True)
    plot_y_vs_yh(y, yh, outfile.format('_compare'))

if __name__ == '__main__':
    fit()
