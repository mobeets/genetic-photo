SEED = 666
from numpy.random import seed
seed(SEED)
from tensorflow import set_random_seed
set_random_seed(SEED)

import glob
import os.path
import argparse
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
from keras import initializers
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.misc import imread
import pandas as pd
from PIL import Image

def get_model(batch_size, original_dim, intermediate_dims, optimizer):
    x = Input(batch_shape=(batch_size, original_dim), name='x')
    hprev = x
    for i,intermediate_dim in enumerate(intermediate_dims):
        h = Dense(intermediate_dim, activation='relu',
            kernel_initializer=initializers.random_normal(seed=SEED),
            bias_initializer=initializers.random_normal(seed=SEED),
            name='h{}'.format(i))(hprev)
        hprev = h
    yhat = Dense(1, activation='sigmoid',
        kernel_initializer=initializers.random_normal(seed=SEED),
        bias_initializer=initializers.random_normal(seed=SEED),
        name='yhat')(h)

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
    plt.xlabel('actual fitness')
    plt.ylabel('predicted fitness')
    if outfile is None:
        plt.show()
    else:
        fig.savefig(outfile)
        plt.close(fig)

def plot_fitness(y, y_medians, outfile):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if len(y_medians) > 0:
        ax.plot(np.arange(len(y_medians)), y_medians, 'k--')
    ax.plot(np.arange(len(y)), y, 'k')
    ax.scatter(np.arange(len(y)), y, marker='.', c='k', s=3)
    # ax.plot([0, 0], [1, 1], 'k-')
    # plt.xlim(0, 1.)
    plt.ylim(0, 1.)
    plt.xlabel('index')
    plt.ylabel('fitness')
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

def mutate_population(X, mutation_rate, crossover_rate):
    """
    randomly mutate and crossover members of population
    """
    # crossover
    Xprime = X[np.random.permutation(X.shape[0])]
    ix = np.random.rand(X.shape[0], X.shape[1]) < crossover_rate
    X[ix] = Xprime[ix]

    # mutate
    Xprime = init_population(X.shape[0], X.shape[1])
    ix = np.random.rand(X.shape[0], X.shape[1]) < mutation_rate
    X[ix] = Xprime[ix]
    
    X[X < -1.] = -1.
    X[X > 1.] = 1.
    return X

def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.

    This is a fast approximation of re-initializing the weights of a model.

    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).

    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    src: https://github.com/keras-team/keras/issues/341
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)

class ModelHolder(object):
    def __init__(self, model, batch_size, num_epochs, iters_to_update):
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.iters_to_update = iters_to_update
        self.X = []
        self.Y = []
        self.iters = 0

    def update(self, Xc, Yc, nsteps, stepsize):
        self.iters += 1
        if len(self.X) == 0:
            self.X = Xc
            self.Y = Yc
        else:
            self.X = np.vstack([self.X, Xc])
            self.Y = np.hstack([self.Y, Yc])
        if self.iters % self.iters_to_update > 0:
            return Xc, False
        X, ix = np.unique(self.X, axis=0, return_index=True)
        Y = self.Y[ix]

        # use last N to match with batch_size
        ind = self.batch_size*(len(X)/self.batch_size)
        history = self.model.fit(X[-ind:], Y[-ind:],
            shuffle=False,
            epochs=self.num_epochs,
            verbose=0,
            batch_size=self.batch_size)

        # reset
        self.X = []
        self.Y = []

        # mutate by maximizing approximated fitness function
        Xh = self.improve(Xc, nsteps, stepsize)
        return Xh, True

    def improve(self, X, nsteps, stepsize):
        return maximize_model_output(self.model, X.copy(), nsteps, stepsize)

def update_population(X, Y, H, M, eval_fcn, num_epochs, batch_size,
    nsteps, stepsize, mutation_rate, crossover_rate, pctToMutate=50, pctToPreserve=10, tol=1e-4):

    # update fitness function estimate
    if nsteps > 0:
        Xh, improved = M.update(X, Y, nsteps, stepsize)
    else:
        improved = False

    # update X by maximizing fitness function estimate
    # Xh = maximize_model_output(mdl, X.copy(), nsteps, stepsize)
    if not improved:
        Xh = mutate_population(X.copy(), mutation_rate, crossover_rate)
        H.update('used_NN', False)
    else:
        H.update('used_NN', True)

    # keep in bounds
    Xh[Xh < -1.] = -1.
    Xh[Xh > 1.] = 1.
    Xnew = Xh
    Ynew = eval_fcn(Xnew)

    # count number improved
    n_improved = (Ynew > Y).sum()
    pct_improved = 100.*n_improved/len(Ynew)
    H.update('improved', [n_improved, len(Ynew), pct_improved])
    ixTopPart = Y >= np.percentile(Y, 100-pctToPreserve)
    n_improved = (Ynew[ixTopPart] > Y[ixTopPart]).sum()
    pct_improved = 100.*n_improved/ixTopPart.sum()
    H.update('best-improved', [pctToPreserve, n_improved, ixTopPart.sum(), pct_improved])

    # replace lower pct with best parents
    inds = np.arange(len(Ynew))
    ixRejects = Ynew <= np.percentile(Ynew, pctToPreserve)
    rejectInds = inds[ixRejects]
    ixBests = Y >= np.percentile(Y, 100-pctToPreserve)
    betterInds = inds[ixBests]
    nDiff = ixBests.sum() - ixRejects.sum()
    if nDiff > 0:
        betterInds = betterInds[:-nDiff]
    elif nDiff < 0:
        rejectInds = rejectInds[:nDiff]
    Xnew[rejectInds] = X[betterInds]
    Ynew[rejectInds] = Y[betterInds]

    return Xnew, Ynew

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
    # based on minimum image patch, adjust range to be [0.0, pctSquish]
    # so that nothing in imgs can be plotted outside the frame
    pctSquish = 1.0 - 1.0*np.min([im.size for im in imgs])/np.max(img.size)
    pts = pctSquish*pts # pts in [0, pctSquish]

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

def _eval_fcn(X, img_target, render_fcn):
    ys = np.zeros(X.shape[0])
    for i, x in enumerate(X):
        img = render_fcn(x)
        ys[i] = img_fitness(np.array(img), img_target) # should be in [0,1]
        # todo: consider squashing so it's actually in [0.2, 0.8]
        #   since with sigmoid activation we likely can't fit the bounds as well
    return ys

def load_img(infile, as_grey=False, keep_alpha=False):
    mode = 'L' if as_grey else ('RGBA' if keep_alpha else 'RGB')
    img = imread(infile, mode=mode)
    return Image.fromarray(img)

def plot_interpolated_fitness(x_best, N, renderf, evalf, outfile):
    x_init = init_population(N=2, K=len(x_best))[0]
    ys = []
    for i, alpha in enumerate(np.linspace(0.0, 1.0, N)):
        x_cur = x_init + alpha*(x_best - x_init)
        y_cur = evalf(x_cur[None,:])[0]
        ys.append(y_cur)
        renderf(x_cur).save(outfile.format('interpolate_{}'.format(i)))
    plot_fitness(np.array(ys), [], outfile.format('interpolate_fitness'))

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
    # x = np.linspace(-1.0, 2.0/n_per_row, n_per_row)
    x = np.linspace(-1.0, 1.0, n_per_row)
    aa,bb = np.meshgrid(x, x)
    x_best = np.array(zip(aa.flatten(), bb.flatten())).flatten()
    target = renderf(x_best)
    target.save(outfile.format('_target'))
    target_img = np.array(target)[:,:,:3]

    renderf = lambda X: render_img(X, target_img.shape[:2], imgs)
    evalf = lambda X: _eval_fcn(X, target_img, renderf)
    y_best = evalf(x_best[None,:])[0]

    # plot_interpolated_fitness(x_best, 100, renderf, evalf, outfile)

    print("Best possible fitness: {}".format(y_best))
    return evalf, renderf

class History:
    def __init__(self, print_every, plot_every, outfile, model, batch_size, render_fcn):
        self.print_every = print_every
        self.plot_every = plot_every
        self.outfile = outfile
        self.model = model
        self.batch_size = batch_size
        self.render_fcn = render_fcn
        self.y_max = -np.inf
        self.x_max = None
        self.x_maxes = []
        self.y_maxes = []
        self.y_medians = []
        self.history = {}

    def update(self, key, vals):
        if key not in self.history:
            self.history[key] = []
        self.history[key].append(vals)

    def print_history(self):
        if 'used_NN' in self.history:
            if self.history['used_NN'][-1]:
                print("        Updated via NN...")
        if 'improved' in self.history:
            a,b,c = self.history['improved'][-1]
            print("        Improved {} of {} ({:0.1f}% success)".format(a, b, c))
        if 'best-improved' in self.history:
            a,b,c,d = self.history['best-improved'][-1]
            print("        Best {}% improved {} of {} ({:0.1f}% success)".format(a,b,c,d))

    def dump_history(self, outfile):
        vss = []
        columns = []

        if 'used_NN' in self.history:
            vs = np.array(self.history['used_NN'])[:,None]
            cols = ['used_NN']
            vss.append(vs)
            columns.extend(cols)

        if 'improved' in self.history:
            vs = np.array(self.history['improved'])
            cols = ['n_improved', 'n_total', 'pct_improved']
            vss.append(vs)
            columns.extend(cols)

        if 'best-improved' in self.history:
            vs = np.array(self.history['best-improved'])
            cols = ['top_pct_val_best', 'n_improved_best', 'n_total_best', 'pct_improved_best']
            vss.append(vs)
            columns.extend(cols)

        vss.append(np.array(self.y_maxes)[:,None])
        vss.append(np.array(self.y_medians)[:,None])
        columns.extend(['y_max', 'y_median'])

        vals = np.hstack(vss)
        df = pd.DataFrame(vals, columns=columns)
        df.to_csv(outfile)

    def update_history(self, X, y, index):
        x_max = self.x_max
        y_max = self.y_max

        if y.max() > y_max:
            y_max_ind = np.argmax(y)
            x_max = X[y_max_ind]
            y_max = y[y_max_ind]
        else:
            y_max_ind = 'prev'

        # save history
        self.x_max = x_max
        self.y_max = y_max
        if type(index) is not str and index >= 0:
            self.y_maxes.append(y_max)
            self.x_maxes.append(x_max)
            self.y_medians.append(np.median(y))

        # print
        if type(index) is str or index % self.print_every == 0:
            name = index if type(index) is str else 'Iter #{}'.format(index)
            print("{} max @ {}: {:0.5f}, median: {:0.5f}".format(name, y_max_ind, y_max, np.median(y)))
            self.print_history()

        # plot
        if type(index) is str or index % self.plot_every == 0:
            # compare predicted with actual fitness
            yh = self.model.predict(X, batch_size=self.batch_size)
            outf = self.outfile.format('compare_e{}'.format(index))
            plot_y_vs_yh(y, yh, outf)

            # render image and save
            img = self.render_fcn(x_max)
            outf = self.outfile.format('img_e{}'.format(index))
            img.save(outf)

            # plot fitness
            outf = self.outfile.format('fitness'.format(index))
            plot_fitness(np.array(self.y_maxes), np.array(self.y_medians), outf)

            # write history
            if type(index) is not str and index > 0:
                outf = self.outfile.format('history').replace('.png', '.csv')
                self.dump_history(outf)

def fit(args):
    outdir = os.path.join(args.outdir, args.run_name)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outfile = os.path.join(outdir, '{}.png')

    # load target image; prepare evaluation and render functions
    eval_fcn, render_fcn = make_eval_and_render_fcn(args.targetdir,
        transparency=args.transparency, outfile=outfile)
    # return

    # init population, NN model, and history
    dims = args.intermediate_dims
    mdl = get_model(args.batch_size, args.K, dims, args.optimizer)
    if args.model_file:
        mdl.load_weights(args.model_file)

    M = ModelHolder(mdl, args.batch_size, args.n_epochs, args.train_every)
    X = init_population(N=args.batch_size, K=args.K)
    y = eval_fcn(X)
    H = History(args.print_every, args.plot_every, outfile, mdl,
        args.batch_size, render_fcn)
    H.update_history(X, y, index='_init')

    # set temperature (for step size)
    temperature = np.sqrt(1./(10+np.arange(args.n_gens)+1))
    temperature = temperature/temperature[0]
    
    # evolve
    print("Evolving {} cells for {} generations...".format(len(y),
        args.n_gens))
    for i in range(args.n_gens):
        # if i == 100:
        #     args.n_gradient_steps = 1
        #     print("Turning on NN...")
        cur_stepsize = args.stepsize*temperature[i]
        cur_mutation_rate = args.mutation_rate*temperature[i]
        cur_crossover_rate = args.crossover_rate*temperature[i]
        print('step={:0.3f}, mut={:0.3f}, cross={:0.3f}'.format(cur_stepsize, cur_mutation_rate, cur_crossover_rate))
        X, y = update_population(X, y, H, M, eval_fcn, args.n_epochs,
            batch_size=args.batch_size, nsteps=args.n_gradient_steps,
            stepsize=cur_stepsize, mutation_rate=cur_mutation_rate, 
            crossover_rate=cur_crossover_rate)
        H.update_history(X, y, index=i)

    # show final results
    H.update_history(X, y, index='_final')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str, help='tag for current run')
    parser.add_argument('-K', '--K', type=int, default=16*2)
    parser.add_argument('--n_gens', type=int, default=1000)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_gradient_steps', type=int, default=1)
    parser.add_argument('--stepsize', type=float, default=1e-2)
    parser.add_argument('--intermediate_dims', type=int,
                                nargs='+', default=[64, 128])
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--train_every', type=int, default=10)
    parser.add_argument('--plot_every', type=int, default=10)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--crossover_rate', type=float, default=0.25)
    parser.add_argument('--mutation_rate', type=float, default=0.05)
    parser.add_argument('--transparency', type=float, default=0.8)
    parser.add_argument('--outdir', type=str, default='logs')
    parser.add_argument('--model_file', type=str,
        default='logs/model.h5')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--targetdir', type=str,
        default='images/trump_puzzle_hard/')
    args = parser.parse_args()
    fit(args)
