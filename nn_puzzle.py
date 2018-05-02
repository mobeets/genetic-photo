import os.path
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# from img import load_img, write_img
from PIL import Image

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
        # todo: if x doesn't increase actual evaluation, keep parent X[i]
    return Xhs

def init_population(N, K):
    return 2*(np.random.rand(N, K) - 0.5)

def update_history(X, y, x_max, y_max, do_print=True, do_plot=False, init=False, mdl=None, batch_size=None, outfile=None, ind=None, render_fcn=None):

    if y.max() > y_max:
        y_max_ind = np.argmax(y)
        x_max = X[y_max_ind]
        y_max = y[y_max_ind]
    if do_print:
        name = 'True' if init else 'Best estimate of'
        x_max_str = '(' + ', '.join(['{:0.3f}'.format(x) for x in x_max]) + ')'
        print(("{} maximum is at {}: {:0.3f}").format(name, x_max_str, y_max))
    if do_plot:
        # compare predicted with actual fitness
        yh = mdl.predict(X, batch_size=batch_size)
        if ind is None:
            outf = outfile.format('compare_e')
        else:
            outf = outfile.format('compare_e' + str(ind))
        plot_y_vs_yh(y, yh, outf)

        # render image and save
        img = render_fcn(x_max)
        if ind is None:
            outf = outfile.format('img_e')
        else:
            outf = outfile.format('img_e' + str(ind))
        img.save(outf)
    return x_max, y_max

def img_fitness(img_predicted, img_target):
    res = 1.0*np.square(img_predicted - img_target).sum()
    # tot = np.prod(img_target.shape).sum() # maximum possible
    tot = 1.0*np.square(img_target).sum() # maximum possible
    return 1.0 - res/tot

def render_img(x, img_size, imgs):
    img = Image.new('RGB', img_size)
    pts = (x+1)/2. # pts in [0,1]
    order = np.argsort(pts[-4:]) # last 4 dims defines ordering
    pts = pts[:-4]
    pts = pts.reshape(-1, 2)
    for (x,y), pimg in zip(pts[order], [imgs[i] for i in order]):
        cx = int(x*img.size[0])
        cy = int(y*img.size[1])
        img.paste(pimg, (cx,cy)) # cx,cy are the upper left corner
    return img

def _eval_fcn(X, img_target, imgs):
    ys = np.zeros(X.shape[0])
    for i, x in enumerate(X):
        img = render_img(x, img_target.shape[:2], imgs)
        ys[i] = img_fitness(np.array(img), img_target) # should be in [0,1]
        # todo: consider squashing so it's actually in [0.2, 0.8]
        #   since with sigmoid activation we likely can't fit the bounds as well
    return ys

def load_img(infile, as_grey=False, keep_alpha=False):
    fin = open(infile) # open the file
    img = Image.open(fin)
    assert(not as_grey)
    assert(not keep_alpha)
    return img

def make_eval_and_render_fcn(targetdir):
    # load target image
    infile = os.path.join(targetdir, 'target.png')
    target = load_img(infile, as_grey=False, keep_alpha=False)

    # load puzzle pieces
    nms = ['NW', 'NE', 'SW', 'SE']
    imgfiles = [os.path.join(targetdir, '{}.png'.format(nm)) for nm in nms]
    imgs = [load_img(imgfile, as_grey=False, keep_alpha=False) for imgfile in imgfiles]

    target_img = np.array(target)[:,:,:3]
    evalf = lambda X: _eval_fcn(X, target_img, imgs)
    renderf = lambda X: render_img(X, target_img.shape[:2], imgs)
    return evalf, renderf

def fit(K=12, num_gens=100, batch_size=100, dims=[16, 32], plot_every=5,
    num_epochs=100, nsteps=1, stepsize=1e-1,
    targetdir='images/trump_puzzle', outfile='logs/nn_puzzle/{}.png',
    optimizer='adam'):
    eval_fcn, render_fcn = make_eval_and_render_fcn(targetdir)

    # init population
    mdl = get_model(batch_size, K, dims, optimizer)
    X = init_population(N=batch_size, K=K)
    y = eval_fcn(X)
    x_max, y_max = update_history(X, y, 0., -np.inf, do_plot=True, mdl=mdl, batch_size=batch_size, outfile=outfile, ind=0, render_fcn=render_fcn)

    # evolve
    print("Updating fitness model given {} cells...".format(len(y)))
    for i in range(num_gens):
        X = update_population(X, y, K, mdl, num_epochs, batch_size, nsteps, stepsize)
        y = eval_fcn(X)
        x_max, y_max = update_history(X, y, x_max, y_max, do_print=True, do_plot=i % plot_every == 0, mdl=mdl, batch_size=batch_size, outfile=outfile, ind=i, render_fcn=render_fcn)

    # show final results
    x_max, y_max = update_history(X, y, x_max, y_max, do_print=True, do_plot=True, mdl=mdl, batch_size=batch_size, outfile=outfile, ind=None, render_fcn=render_fcn)

if __name__ == '__main__':
    fit()
