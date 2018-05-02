import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
# from skimage.draw import circle, circle_perimeter
from itertools import product
from model import Cell
from PIL import Image, ImageDraw

def load_img(target_file, as_grey=False, keep_alpha=False):
    img = imread(target_file, as_grey=as_grey)
    if not keep_alpha and len(img.shape) > 3:
        # ignore alpha channel, if present
        img = img[:,:,:3]
    return img

def write_img(img, outfile=None, fig_size=(3,3)):
    """
    img_size = number of pixels
    fig_size = size when making image
    """
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=fig_size)
    # ax.imshow(1-img, cmap='Greys', vmin=0, vmax=1) # invert colors
    ax.imshow(img, vmin=0, vmax=255)
    ax.axis('off')
    if outfile is not None:
        fig.savefig(outfile)
        plt.close(fig)

def fitness(img_predicted, img_target):
    res = np.square(img_predicted - img_target).sum()
    # tot = np.prod(img_target.shape).sum() # maximum possible
    tot = np.square(img_target).sum() # maximum possible
    return 1.0 - res/tot

def render_square(seq, img):
    """
    filled squares
    """
    (xpos, ypos, radius, color) = seq
    if radius == 0:
        return img

    xs = np.arange(np.round(xpos), np.round(xpos + radius))
    ys = np.arange(np.round(ypos), np.round(ypos + radius))
    xs = xs[(xs >= 0) & (xs < img.shape[0])].astype(int)
    ys = ys[(ys >= 0) & (ys < img.shape[1])].astype(int)
    rr = [x for x,y in product(xs, ys)]
    cc = [y for x,y in product(xs, ys)]
    if len(rr) == 0:
        raise Exception("Empty square. Radius must be too small.")
    img[rr, cc] = color
    return img

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

class TriangleCell(Cell):
    def __init__(self, img_target, ngenes, genes=None):
        self.img_target = img_target
        self.img_size = img_target.shape
        self.lbs, self.ubs = self.set_bounds()
        self.nparams = len(self.lbs)
        Cell.__init__(self, ngenes, genes)

    def set_bounds(self):
        lbs = np.zeros(10)
        ubs = np.array([self.img_size[0], self.img_size[1], self.img_size[0], self.img_size[1], self.img_size[0], self.img_size[1], 255., 255., 255., 255.])
        return lbs, ubs

    def render_triangle(self, seq):
        x1,y1,x2,y2,x3,y3,R,G,B,A = seq

        img = Image.new('RGBA', self.img_size[:2]) # Use RGBA
        draw = ImageDraw.Draw(img)
        draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=(int(R),int(G),int(B),int(A)))
        return img

    def render(self):
        imgs = []
        seqs = np.reshape(self.genes, (-1, len(self.lbs)))
        Img = Image.new('RGBA', self.img_size[:2])
        for seq in seqs:
            cseq = self.lbs + seq*(self.ubs - self.lbs)
            img = self.render_triangle(cseq)
            Img = Image.alpha_composite(Img, img)
        return pure_pil_alpha_to_color_v2(Img)

    def evaluate(self):
        self.fitness = fitness(self.render(), self.img_target)

class SquareCell(Cell):
    def __init__(self, img_target, ngenes, genes=None):
        self.img_target = img_target
        self.img_size = img_target.shape
        self.lbs, self.ubs = self.set_bounds()
        self.nparams = len(self.lbs)
        Cell.__init__(self, ngenes, genes)

    def set_bounds(self):
        lbs = np.array([0.0, 0.0, 0.0, 0.0])
        ubs = np.array([self.img_size[0], self.img_size[1], np.min(self.img_size)/30., 1.0])
        return lbs, ubs

    def render(self):
        img = np.ones(self.img_size, dtype=np.double)
        seqs = np.reshape(self.genes, (-1, 4))
        for seq in seqs:
            cseq = self.lbs + seq*(self.ubs - self.lbs)
            img = render_square(cseq, img)
        return img

    def evaluate(self):
        self.fitness = fitness(self.render(), self.img_target)

class SquareCellFixedPosAndRadius(SquareCell):
    def set_bounds(self):
        lbs = np.array([0.0])
        ubs = np.array([1.0])
        return lbs, ubs

    def render(self):
        """
        seq only contains radius and color
        so assume that each gene refers to squares in a grid spanning the img
        """
        self.genes[self.genes < self.lbs] = self.lbs
        self.genes[self.genes > self.ubs] = self.ubs
        return (self.lbs + self.genes*(self.ubs - self.lbs)).reshape(self.img_size)

class SquareCellFixedPos(SquareCell):
    def set_bounds(self):
        lbs = np.array([0.0, 0.0])
        ubs = np.array([np.min(self.img_size)/10., 1.0])
        return lbs, ubs

    def render(self):
        """
        seq only contains radius and color
        so assume that each gene refers to squares in a grid spanning the img
        """
        img = np.ones(self.img_size, dtype=np.double)
        seqs = np.reshape(self.genes, (-1, self.nparams))

        n_circles = seqs.shape[0]
        circs_per_row = 1.0*np.sqrt(n_circles)

        for i, seq in enumerate(seqs):
            xpos = np.floor(i/circs_per_row) / circs_per_row
            ypos = (i % circs_per_row) / circs_per_row
            xpos = xpos*self.img_size[0]
            ypos = ypos*self.img_size[1]

            cseq = self.lbs + seq*(self.ubs - self.lbs)
            cseq = (xpos, ypos) + tuple(cseq)
            img = render_square(cseq, img)
        return img

class SquareCellFixedRad(SquareCell):
    def set_bounds(self):
        lbs = np.array([0.0, 0.0, 0.0])
        ubs = np.array([self.img_size[0], self.img_size[1], 1.0])
        return lbs, ubs

    def render(self):
        rad = 2.
        img = np.ones(self.img_size, dtype=np.double)
        seqs = np.reshape(self.genes, (-1, 3))
        for seq in seqs:
            cseq = self.lbs + seq*(self.ubs - self.lbs)
            cseq = cseq[:2].tolist() + [rad] + [cseq[-1]]
            img = render_square(cseq, img)
        return img

def cell_maker_fcn(cellcls, args, img_target):
    return lambda genes=None: cellcls(img_target, args.num_genes, genes)
