import glob
import os.path
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.misc import imread
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
    Xh = X0
    # run gradient ascent on model to maximize f(Xh)
    for i in range(nsteps):
        loss_value, grads_value = iterate([Xh])
        Xh += grads_value * stepsize
    return Xh

def plot_y_vs_yh(y, yh, outfile):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(y, yh, marker='.', c='k', s=1)
    # ax.plot([0, 0], [1, 1], 'k-')
    plt.xlim(0, 1.)
    plt.ylim(0, 1.)
    if outfile is None:
        plt.show()
    else:
        fig.savefig(outfile)
        plt.close(fig)

def get_test_data(N, K, batch_size):
    # generate example data
    x1 = np.linspace(-1, 1, N)
    x = np.meshgrid(*[x1 for i in range(K)])
    X = np.vstack([xi.flatten() for xi in x]).T
    n = batch_size*int(X.shape[0]/batch_size)
    return X[:n]

def init_population(N, K):
    return 2*(np.random.rand(N, K) - 0.5)

def mutate_population(X, mutation_rate):
    Xprime = init_population(X.shape[0], X.shape[1])
    ix = np.random.rand(X.shape[0], X.shape[1]) < mutation_rate
    X[ix] = Xprime[ix]
    X[X < -1.] = -1.
    X[X > 1.] = 1.
    return X

def update_population(X, Y, K, mdl, eval_fcn, num_epochs, batch_size,
    nsteps, stepsize, tol=1e-4, mutation_rate=0.2, pctToMutate=50, pctToPreserve=10):
    # update fitness function estimate
    history = mdl.fit(X, Y,
        shuffle=True,
        epochs=num_epochs,
        verbose=0,
        batch_size=batch_size)

    # update X by maximizing fitness function estimate
    Xh = maximize_model_output(mdl, X.copy(), nsteps, stepsize)

    # keep in bounds
    Xh[Xh < -1.] = -1.
    Xh[Xh > 1.] = 1.

    # new population is Xh above; but if it's already in X, choose random mutation
    Xmuts = mutate_population(X.copy(), mutation_rate)
    Xnew = np.nan*Xmuts
    for i, x_new in enumerate(Xh):
        ix1 = (np.square(X - x_new).sum(axis=1) < tol) # already in X
        ix2 = (np.square(Xmuts - x_new).sum(axis=1) < tol) # already in Xmuts
        if (ix1 | ix2).any():
            # overwrite random mutation
            x_new = Xmuts[i]
        Xnew[i] = x_new
    Ynew = eval_fcn(Xnew)

    # count number improved
    n_improved = (Ynew > Y).sum()
    pct_improved = 100.*n_improved/len(Ynew)
    print("        Improved {} of {} ({:0.1f}% success)".format(n_improved, len(Ynew), pct_improved))
    ixTopPart = Ynew > np.percentile(Ynew, 100-pctToPreserve)
    n_improved = (Ynew[ixTopPart] > Y[ixTopPart]).sum()
    pct_improved = 100.*n_improved/ixTopPart.sum()
    print("        Best {}% improved {} of {} ({:0.1f}% success)".format(pctToPreserve, n_improved, ixTopPart.sum(), pct_improved))

    # if best parent still best, keep it
    if Y.max() > Ynew.max():
        x_best_ind = np.argmax(Y)
        x_worst_ind = np.argmin(Ynew)
        Xnew[x_worst_ind] = X[x_best_ind]
        Ynew[x_worst_ind] = Y[x_best_ind]

    # swap out lower half with best parents
    inds = np.arange(len(Ynew))
    ixRejects = Ynew < np.percentile(Ynew, pctToPreserve)
    rejectInds = inds[ixRejects]
    ixBests = Y > np.percentile(Y, 100-pctToPreserve)
    betterInds = inds[ixBests]
    nDiff = ixBests.sum() - ixRejects.sum()
    if nDiff > 0:
        betterInds = betterInds[:-nDiff]
    elif nDiff < 0:
        rejectInds = rejectInds[:nDiff]
    Xnew[rejectInds] = X[betterInds]
    Ynew[rejectInds] = Y[betterInds]

    # mutate lower half with mutants of better half
    inds = np.arange(len(Ynew))
    ixRejects = Ynew < np.percentile(Ynew, pctToMutate)
    rejectInds = inds[ixRejects]
    ixBests = Ynew > np.percentile(Ynew, 100-pctToMutate)
    betterInds = inds[ixBests]
    nDiff = ixBests.sum() - ixRejects.sum()
    if nDiff > 0:
        betterInds = betterInds[:-nDiff]
    elif nDiff < 0:
        rejectInds = rejectInds[:nDiff]
    Xnew[rejectInds] = Xmuts[betterInds]
    Ynew[rejectInds] = eval_fcn(Xnew[rejectInds])

    n_improved = (Ynew[rejectInds] > Ynew[ixBests].min()).sum()
    pct_improved = 100.*n_improved/len(rejectInds)
    print("        Mutants: {} of {} reached top {}% ({:0.1f}% success)".format(n_improved, len(rejectInds), pct_improved, 100-pctToMutate))
    # print(ns)
    return Xnew, Ynew

def update_history(X, y, x_max, y_max, do_print=True, do_plot=False, init=False, mdl=None, batch_size=None, outfile=None, ind=None, render_fcn=None):

    if y.max() > y_max:
        y_max_ind = np.argmax(y)
        x_max = X[y_max_ind]
        y_max = y[y_max_ind]
    else:
        y_max_ind = 'prev'
    if do_print:
        name = 'True' if init else 'Iter #{}'.format(ind)
        # x_max_str = '(' + ', '.join(['{:0.3f}'.format(x) for x in x_max]) + ')'
        # print(("{} max @ {}: {:0.4f}").format(name, x_max_str, y_max))
        print(("{} max @ {}: {:0.4f}").format(name, y_max_ind, y_max))
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

def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.
    src: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)
    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background

def render_img(x, img_size, imgs):
    # img = Image.new('RGB', img_size)
    img = Image.new('RGBA', img_size)

    pts = (x+1)/2. # pts in [0,1]
    nimgs = len(imgs)
    # order = np.argsort(pts[-nimgs:]) # last 4 dims defines ordering
    # pts = pts[:-nimgs]
    order = np.arange(nimgs)
    pts = pts.reshape(-1, 2)
    for (x,y), pimg in zip(pts[order], [imgs[i] for i in order]):
        cx = int(x*img.size[0])
        cy = int(y*img.size[1])
        # cx,cy are the upper left corner
        img.paste(pimg, (cx,cy), pimg) # third arg is "mask" to use alpha
    img = pure_pil_alpha_to_color_v2(img) # RGBA to RGB
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
    mode = 'L' if as_grey else ('RGBA' if keep_alpha else 'RGB')
    img = imread(infile, mode=mode)
    return Image.fromarray(img)

def make_eval_and_render_fcn(targetdir, transparency=0.8, outfile=None):
    # load target image
    infile = os.path.join(targetdir, 'target.png')
    target = load_img(infile, as_grey=False, keep_alpha=False)
    if transparency < 1.0:        
        a_channel = Image.new('L', target.size, int(255.*transparency))
        target.putalpha(a_channel)
        target = pure_pil_alpha_to_color_v2(target)

    # load puzzle pieces
    # nms = ['NE', 'SW', 'SE', 'NW']
    # imgfiles = [os.path.join(targetdir, '{}.png'.format(nm)) for nm in nms]
    imgfiles = glob.glob(os.path.join(targetdir, '*.jpg'))
    imgs = [load_img(imgfile, as_grey=False, keep_alpha=False) for imgfile in imgfiles]
    # add alpha channel
    for i in range(len(imgs)):
        # 'L' 8-bit pixels, black and white
        a_channel = Image.new('L', imgs[i].size, int(255.*transparency))
        imgs[i].putalpha(a_channel)

    target_img = np.array(target)[:,:,:3]

    renderf = lambda X: render_img(X, target_img.shape[:2], imgs)
    
    print("Warning: Trying to make best target image...")
    n_per_row = int(np.sqrt(len(imgs)))
    x = np.linspace(-1.0, 2.0/n_per_row, n_per_row)
    aa,bb = np.meshgrid(x, x)
    x_best = np.array(zip(aa.flatten(), bb.flatten())).flatten()
    target = renderf(x_best)
    target.save(outfile.format('_target'))
    target_img = np.array(target)[:,:,:3]

    renderf = lambda X: render_img(X, target_img.shape[:2], imgs)
    evalf = lambda X: _eval_fcn(X, target_img, imgs)
    y_best = evalf(x_best[None,:])[0]
    print("Best possible fitness: {}".format(y_best))
    return evalf, renderf

def fit(K=16*2, num_gens=500, batch_size=1000, print_every=1,
    plot_every=10, num_epochs=100, nsteps=1, stepsize=1e-2,
    targetdir='images/trump_puzzle_hard', outfile='logs/nn_puzzle_hard/{}.png',
    optimizer='adam', transparency=0.8):
    eval_fcn, render_fcn = make_eval_and_render_fcn(targetdir, transparency=transparency, outfile=outfile)

    # init population
    # dims = [16, 32]
    dims = [K*2, K*4]
    mdl = get_model(batch_size, K, dims, optimizer)
    X = init_population(N=batch_size, K=K)
    y = eval_fcn(X)
    x_max, y_max = update_history(X, y, 0., -np.inf, do_plot=True, mdl=mdl, batch_size=batch_size, outfile=outfile, ind=0, render_fcn=render_fcn)

    # evolve
    print("Updating fitness model given {} cells...".format(len(y)))
    for i in range(num_gens):
        X, y = update_population(X, y, K, mdl, eval_fcn, num_epochs, batch_size, nsteps, stepsize)
        x_max, y_max = update_history(X, y, x_max, y_max, do_print=i % print_every == 0, do_plot=i % plot_every == 0, mdl=mdl, batch_size=batch_size, outfile=outfile, ind=i, render_fcn=render_fcn)

    # show final results
    x_max, y_max = update_history(X, y, x_max, y_max, do_print=True, do_plot=True, mdl=mdl, batch_size=batch_size, outfile=outfile, ind='final', render_fcn=render_fcn)

if __name__ == '__main__':
    fit()
