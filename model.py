import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense

def get_model(batch_size, original_dim, intermediate_dims, optimizer):
    x = Input(batch_shape=(batch_size, original_dim), name='x')
    hprev = x
    for i,intermediate_dim in enumerate(intermediate_dims):
        h = Dense(intermediate_dim, activation='relu', name='h{}'.format(i))(hprev)
        hprev = h
    yhat = Dense(1, activation='sigmoid', name='yhat')(h)

    mdl = Model(x, yhat)
    mdl.compile(optimizer=optimizer, loss='mse')
    return mdl

def maximize_model_output(model, X0, nsteps, stepsize=1.):
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
    Xh = X0[None,:]
    # run gradient ascent on model to maximize f(Xh)
    for i in range(nsteps):
        loss_value, grads_value = iterate([Xh])
        Xh -= grads_value * stepsize
    return Xh

def update_model_and_population(P, img_target, args):
    """
    to do: only include part of seqs that's actually being optimized over
    also, figure out why we don't improve...is Xh just the same as X0?
        Or is it just noise?
    """
    # selection
    old_cells = sorted(P.cells, key=lambda cell: cell.fitness, reverse=True)
    ncells = len(old_cells)
    nparents = int(np.ceil(args.pct_reproduce*ncells))
    parents = old_cells[:nparents]

    # update fitness model given (X,y)
    seqf = lambda seq: (seq - P.lbs)/(P.ubs - P.lbs) # normalize
    genes_to_row = lambda genes: np.hstack([seqf(g.seq) for g in genes])

    X = np.vstack([genes_to_row(cell.genes) for cell in P.cells])
    y = np.array([cell.fitness for cell in P.cells])
    print("Updating fitness model given {} cells...".format(len(y)))
    history = P.model.fit(X, y,
        shuffle=True,
        epochs=10,
        verbose=1,
        batch_size=args.num_cells)
    P.model_b1.set_weights(P.model.get_weights())
    # n.b. for some reason it's predicting 1 for everything...
    1/0

    # maximize y w.r.t. X, starting from X0
    # print("Gradient ascent to maximize {} cells...".format(ncells-nparents))
    nseq = len(P.cells[0].genes[0].seq)    
    X_to_seqs = lambda X: np.reshape(X, (-1, nseq))
    
    X0 = genes_to_row(P.cells[0].genes)
    Xh = maximize_model_output(P.model_b1, X0, nsteps=20)
    seqs = X_to_seqs(Xh)
    cell = P.cell_from_seqs(seqs, img_target, is_normalized=False)
    cells = [cell]
    for i in range(ncells-nparents-1):
        cells.append(P.cell_from_parent(cell, img_target, args))

    # Xhs = []
    # for cell in old_cells[:(ncells-nparents)]:
    #     X0 = genes_to_row(cell.genes)
    #     Xh = maximize_model_output(P.model_b1, X0, nsteps=5)
    #     Xhs.append(Xh)
    #     seqs = X_to_seqs(Xh)
    #     cell = P.cell_from_seqs(seqs, img_target)
    #     cells.append(cell)
    # vals = P.model_b1.predict(np.vstack(Xhs), batch_size=1)
    # print("fitness predictions: min:{:0.3f}, med:{:0.3f}, max:{:0.3f}".format(vals.min(), np.median(vals), vals.max()))

    # update cells (keeping parents)
    P.cells = cells
    P.cells = np.hstack([P.cells, parents]) # keep parents
    P.evaluate(img_target)
    return P
