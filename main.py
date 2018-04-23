import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.draw import (line, polygon, circle,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve)
from itertools import product
from model import get_model, update_model_and_population

class PolygonGene(object):
    def __init__(self, seq):
        self.seq = seq

    def render(self, img):
        (x1, x2, x3, y1, y2, y3, color) = self.seq

        poly = np.array((
            (x1, y1),
            (x2, y2),
            (x3, y3),
            (x1, y1),
        ))
        rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
        img[rr, cc] = color
        return img

class CircleGene(object):
    def __init__(self, seq):
        self.seq = seq

    def render_clr(self, img):
        """
        filled circles with color
        """
        (xpos, ypos, radius, R, G, B) = self.seq
        if radius == 0:
            return img

        rr, cc = circle(xpos, ypos, radius, img.shape)
        img[rr, cc, :] = (R, G, B)
        return img

    def render_circ(self, img):
        """
        filled circles
        """
        (xpos, ypos, radius, color) = self.seq
        if radius == 0:
            return img

        rr, cc = circle(xpos, ypos, radius, img.shape)
        img[rr, cc] = color
        return img

    def render(self, img):
        """
        filled squares
        """
        (xpos, ypos, radius, color) = self.seq
        if radius == 0:
            return img

        xs = np.arange(int(xpos - radius), int(np.ceil(xpos + radius)))
        ys = np.arange(int(ypos - radius), int(np.ceil(ypos + radius)))
        xs = xs[(xs >= 0) & (xs < img.shape[0])]
        ys = ys[(ys >= 0) & (ys < img.shape[1])]
        rr = [x for x,y in product(xs, ys)]
        cc = [y for x,y in product(xs, ys)]

        img[rr, cc] = color
        return img

    def render_sq(self, img):
        """
        boundary of squares
        """
        (xpos, ypos, radius, color) = self.seq
        if radius == 0:
            return img

        x1 = int(xpos - radius)
        x2 = int(np.ceil(xpos + radius))
        y1 = int(ypos - radius)
        y2 = int(np.ceil(ypos + radius))
        
        xs = np.arange(x1, x2)
        ys = np.arange(y1, y2)

        top = np.vstack([xs, np.repeat(y1, len(xs))])
        bot = np.vstack([xs, np.repeat(y2, len(xs))])
        lef = np.vstack([np.repeat(x1, len(ys)), ys])
        rig = np.vstack([np.repeat(x2, len(ys)), ys])
        pts = np.hstack([top, bot, lef, rig])
        rr = pts[0,:]
        cc = pts[1,:]
        ix = (rr < 0) | (cc < 0) | (rr >= img.shape[0]) | (cc >= img.shape[1])
        rr = pts[0,~ix]
        cc = pts[1,~ix]

        # rr = [x for x,y in product(xs, ys)]
        # cc = [y for x,y in product(xs, ys)]

        img[rr, cc] = color
        return img

class Cell(object):
    def __init__(self, img_size):
        self.img_size = img_size
        self.genes = []
        self.fitness = 0.0

    def to_img(self):
        img = np.ones(self.img_size, dtype=np.double)
        for gene in self.genes:
            img = gene.render(img)
        return img

    def evaluate(self, img_target):
        self.fitness = fitness(self, img_target)

    def write(self, outfile=None, fig_size=(3,3)):
        """
        img_size = number of pixels
        fig_size = size when making image
        """
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=fig_size)
        img_cell = self.to_img()
        ax.imshow(1-img_cell, cmap='Greys', vmin=0, vmax=1) # invert colors
        # ax.imshow(img_cell, vmin=0, vmax=255)
        ax.axis('off')
        if outfile is not None:
            fig.savefig(outfile)
            plt.close(fig)

class Population(object):
    def __init__(self, geneclass, lbs, ubs, args, img_target):
        self.geneclass = geneclass
        self.img_size = img_target.shape
        self.lbs = lbs
        self.ubs = ubs
        self.cells = self.init_population(args, img_target)
        self.evaluate(img_target)

    def init_population(self, args, img_target):
        num_cells = args.num_cells
        cells = []
        for i in range(num_cells):
            cell = Cell(self.img_size)
            cell = self.init_genes(cell, args.num_genes, img_target)
            cells.append(cell)
        return cells

    def evaluate(self, img_target):
        [cell.evaluate(img_target) for cell in self.cells]

    def mutate_seq(self, old_seq, args):
        if np.random.rand() > args.prob_gene_mutate:
            return old_seq

        seq = (old_seq - self.lbs)/(self.ubs - self.lbs) # normalize

        delta = np.sqrt(args.noise_var)*np.random.randn(len(self.lbs),1)

        ps = np.random.rand(len(self.lbs))
        # while (ps < args.prob_feature_mutate).sum() == 0:
        #     # make sure we mutate something
        #     ps = np.random.rand(len(self.lbs))
        delta[0] = 0.0
        delta[1] = 0.0
        delta[2] = 0.0
        delta[ps > args.prob_feature_mutate] = 0.0 # don't mutate
        seq = seq + delta.T
        seq = np.maximum(0, np.minimum(1, seq)) # keep in range

        seq = self.lbs + seq*(self.ubs - self.lbs)
        # print('{} to {}'.format(old_seq, seq))
        return seq[0]

    def cell_from_seqs(self, seqs, img_target, is_normalized=True):
        cell = Cell(img_target.shape)
        if not is_normalized:
            seqs = self.lbs + seqs*(self.ubs - self.lbs)
        cell.genes = [self.geneclass(seq) for seq in seqs]
        return cell

    def cell_from_parent(self, parent, img_target, args):
        """
        mutate parent
        """
        cell = Cell(img_target.shape)
        cell.genes = [self.geneclass(self.mutate_seq(g.seq, args)) for g in parent.genes]
        return cell

    def cell_from_parents(self, p1, p2, img_target, args):
        """
        crossover parent genes, then mutate
        """
        cell = Cell(img_target.shape)
        ngenes = len(p1.genes)
        crossover_ind = int(ngenes*np.random.rand())
        cell.genes = p1.genes[:crossover_ind]
        cell.genes.extend(p2.genes[crossover_ind:])
        assert(len(cell.genes) == ngenes)
        cell.genes = [self.geneclass(self.mutate_seq(g.seq, args)) for g in cell.genes]
        if np.random.rand() < args.prob_shuffle:
            np.random.shuffle(cell.genes)
        return cell

    def random_seq(self, img_target, n_circles, i):
        seq = np.random.rand(len(self.lbs),1)
        circs_per_row = np.sqrt(n_circles)
        cur_row = (np.floor(i/circs_per_row) / circs_per_row)
        cur_col = (i % circs_per_row) / circs_per_row
        seq[0] = cur_row
        seq[1] = cur_col
        seq[2] = 0.26 # initial radius
        seq = self.lbs + seq.T*(self.ubs - self.lbs)
        sd = 0.1
        # seq[0][3] = img_target.mean()#-(sd/2) + sd*np.random.rand()
        # seq[0][4] = img_target.mean()#-(sd/2) + sd*np.random.rand()
        # seq[0][5] = img_target.mean()#-(sd/2) + sd*np.random.rand()
        return seq[0]

    def init_genes(self, cell, n_circles, img_target):
        cell.genes = [self.geneclass(self.random_seq(img_target, n_circles, i)) for i in range(n_circles)]
        return cell

def update_population(P, img_target, args):
    # selection
    old_cells = sorted(P.cells, key=lambda cell: cell.fitness, reverse=True)
    ncells = len(old_cells)
    nparents = int(np.ceil(args.pct_reproduce*ncells))
    parents = old_cells[:nparents]
    
    # self-breeding with mutation
    cells = []
    if args.self_breed:
        for parent in parents:
            for i in range(args.num_children):
                cells.append(P.cell_from_parent(parent, img_target, args))
    else:
        # crossover with mutation
        pinds = (nparents*np.random.rand(ncells-nparents,2)).astype(int)
        for i1, i2 in pinds:
            cells.append(P.cell_from_parents(parents[i1], parents[i2], img_target, args))

    P.cells = cells[:ncells-nparents]
    # P.cells = np.random.choice(cells, ncells-nparents)
    P.cells = np.hstack([P.cells, parents]) # keep parents
    P.evaluate(img_target)
    return P

def fitness(cell, img_target):
    res = np.square(cell.to_img() - img_target).sum()
    # tot = np.prod(img_target.shape).sum() # maximum possible
    tot = np.square(img_target).sum() # maximum possible
    return 1.0 - res/tot
    # return -res

def get_best_cell(cells, img_target, best_cell=None):
    cur_best_cell = max(cells, key=lambda cell: cell.fitness)
    if best_cell is None or cur_best_cell.fitness > best_cell.fitness:
        return cur_best_cell
    else:
        return best_cell

def train(args):
    # load target img
    img_target = imread(args.target_file, as_grey=args.as_grey)
    print(img_target.shape, img_target.min(), img_target.max())
    # if len(img_target.shape) > 3:
    #     # ignore alpha channel, if present
    #     img_target = img_target[:,:,:3]

    # prepare to write logs
    if not os.path.exists(os.path.join(args.outdir, args.run_name)):
        os.mkdir(os.path.join(args.outdir, args.run_name))

    # init random circle population (b&w)
    img_size = img_target.shape
    lbs = np.array([0.0, 0.0, 0.0, 0.0])
    ubs = np.array([img_size[0], img_size[1], np.min(img_size)/30., 1.0])
    # ubs = np.array([img_size[0], img_size[1], 1.0, 1.0])
    P = Population(CircleGene, lbs, ubs, args, img_target)

    # create model to estimate fitness given individual
    if args.model_fitness:
        original_dim = len(lbs)*args.num_genes
        P.model = get_model(args.num_cells, original_dim, [int(original_dim/2), int(original_dim/4)], 'adam')
        P.model_b1 = get_model(1, original_dim, [int(original_dim/2), int(original_dim/4)], 'adam')

    # init random circle population (color)
    # img_size = img_target.shape
    # lbs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # ubs = np.array([img_size[0], img_size[1], np.min(img_size[:2])/10., 255.0, 255.0, 255.0])
    # P = Population(CircleGene, lbs, ubs, args, img_target)

    # init random polygon population
    # img_size = img_target.shape
    # lbs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # ubs = np.array([img_size[0], img_size[0], img_size[0], img_size[1], img_size[1], img_size[1], 1.0])
    # P = Population(PolygonGene, lbs, ubs, args, img_target)

    best_cell = get_best_cell(P.cells, img_target)
    for i in range(args.num_epochs):
        # new generation
        if args.model_fitness:
            P = update_model_and_population(P, img_target, args)
        else:
            P = update_population(P, img_target, args)

        # find best cell and save its image
        if i % 10 == 0:
            print('Epoch {}'.format(i+1))
            best_cell = get_best_cell(P.cells, img_target, best_cell)
            # print(sorted([c.fitness for c in P.cells]))
            fs = np.array([c.fitness for c in P.cells])
            print('{} cells with best fitness {:0.3f}, median fitness {:0.3f}, and minimum fitness {:0.3f}'.format(len(P.cells), best_cell.fitness, np.median(fs), fs.min()))
            outfile = os.path.join(args.outdir, args.run_name, '{}.png'.format(i))
            best_cell.write(outfile)

if __name__ == '__main__':
    # source deactivate
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
        help='tag for current run')
    parser.add_argument('--num_epochs', type=int, default=20000)
    parser.add_argument('--num_cells', type=int, default=50)
    parser.add_argument('--num_children', type=int, default=5)
    parser.add_argument('--num_genes', type=int, default=2105)
    parser.add_argument('--pct_reproduce', type=float, default=0.2)
    parser.add_argument('--prob_gene_mutate', type=float, default=0.5)
    parser.add_argument('--prob_feature_mutate', type=float, default=0.5)
    parser.add_argument('--prob_shuffle', type=float, default=0.0)
    parser.add_argument('--noise_var', type=float, default=0.001) # 0.0001
    # parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument("--self_breed", action="store_true")
    parser.add_argument("--as_grey", action="store_true")
    parser.add_argument("--model_fitness", action="store_true")
    parser.add_argument('--outdir', type=str, default='logs')
    parser.add_argument('--target_file', type=str, default='images/trump.png')
    args = parser.parse_args()
    train(args)
