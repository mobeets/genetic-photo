import os.path
import argparse
import numpy as np
from skimage.io import imread
from model import Population
from img import load_img, write_img, cell_maker_fcn, SquareCellFixedPos, SquareCellFixedPosAndRadius, SquareCellFixedRad, TriangleCell
from methods import update_population_NM, update_population_DE

def get_update_fcn(args):
    if args.use_nm:
        return update_population_NM
    elif args.use_de:
        return update_population_DE
    else:
        raise Exception("No evolution method was specified.")

def train(P, updatefcn, args):
    best_cell = max(P.cells, key=lambda cell: cell.fitness)
    for i in range(args.num_epochs):
        # make new generation
        P, is_done = updatefcn(P, args)

        # log population fitness info
        if i % args.log_every == 0:
            print('Epoch {}'.format(i+1))
            fs = np.array([c.fitness for c in P.cells])
            print('{} cells with best fitness {:0.3f}, median fitness {:0.3f}, and minimum fitness {:0.3f}'.format(len(P.cells), fs.max(), np.median(fs), fs.min()))
        # save image using best cell
        if i % args.save_every == 0:
            best_cell = max(P.cells, key=lambda cell: cell.fitness)
            outfile = os.path.join(args.outdir, args.run_name, '{}.png'.format(i))
            write_img(best_cell.render(), outfile)

        if is_done:
            break

    best_cell = max(P.cells, key=lambda cell: cell.fitness)
    return best_cell

def main(args):
    # load target img and write (for reference)
    img_target = load_img(args.target_file, as_grey=args.as_grey, keep_alpha=False)
    outfile = os.path.join(args.outdir, args.run_name, '_input.png')
    write_img(img_target, outfile)

    # prepare to write logs
    if not os.path.exists(os.path.join(args.outdir, args.run_name)):
        os.mkdir(os.path.join(args.outdir, args.run_name))

    # create population
    cellfcn = cell_maker_fcn(TriangleCell, args, img_target)
    # cellfcn = cell_maker_fcn(SquareCellFixedPosAndRadius, args, img_target)
    # cellfcn = cell_maker_fcn(SquareCellFixedPos, args, img_target)
    # cellfcn = cell_maker_fcn(SquareCellFixedRad, args, img_target)
    updatefcn = get_update_fcn(args)
    print("Initialized. Creating population...")
    P = Population(args, cellfcn)
    print("Training...")
    return train(P, updatefcn, args)

if __name__ == '__main__':
    # source deactivate
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str, help='tag for current run')
    parser.add_argument('--num_epochs', type=int, default=20000)
    parser.add_argument('--num_cells', type=int, default=5000) # 50
    parser.add_argument('--num_genes', type=int, default=500)

    # Nelder-Mead:
    parser.add_argument("--use_nm", action="store_true",
        help="use Nelder-Mead algorithm")
    parser.add_argument('--nm_alpha', type=float, default=0.9)
    parser.add_argument('--nm_gamma', type=float, default=2.0)
    parser.add_argument('--nm_rho', type=float, default=0.7)
    parser.add_argument('--nm_sigma', type=float, default=0.5)

    # Differential Evolution:
    parser.add_argument("--use_de", action="store_true",
        help="Use Differential Evolution")
    parser.add_argument('--de_crossover_rate', type=float, default=0.5)
    parser.add_argument('--de_F', type=float, default=0.8)

    parser.add_argument('--target_file', type=str, default='images/trump.png')
    parser.add_argument("--as_grey", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument('--outdir', type=str, default='logs')
    args = parser.parse_args()
    main(args)
