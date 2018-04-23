import random
import numpy as np

def update_population_DE(P, args):
	"""
	Differential Evolution
	src: "Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces"
	"""
	cells = P.cells
	ncells = len(cells)

	pinds = (ncells*np.random.rand(ncells,3)).astype(int)
	nfeatures = P.cells[0].ngenes
	probs = np.random.rand(ncells, nfeatures)
	new_cells = []
	n_new_cells = 0
	for i,cell in enumerate(cells):
		# pick 3 random other cells to form mutated vector
		if ncells > 3000:
			a,b,c = np.random.permutation(ncells)[:3]
		else:
			a,b,c = random.sample(range(ncells), 3)
		vecA = cells[a].genes
		vecB = cells[b].genes
		vecC = cells[c].genes
		vec_mutated = vecA + args.de_F*(vecB - vecC)

		# crossover cell's vec with vec_mutated to form new cell
		vec = cell.genes
		ix = (probs[i,:] < args.de_crossover_rate)
		if ix.sum() == 0:
			ix[int(nfeatures*np.random.rand())] = True
		vec[ix] = vec_mutated[ix]
		new_cell = P.new_cell(vec)

		# evaluate, and keep only if we improved
		if new_cell.fitness > cell.fitness:
			new_cells.append(new_cell)
			n_new_cells += 1
		else:
			# might need to mutate here?
			new_cells.append(cell)
	print("{} of {} new cells".format(n_new_cells, ncells))
	if n_new_cells == 0:
		raise Exception("Done.")
	P.cells = new_cells
	return P

def cell_from_vector_math(A, B, scale, P):
	"""
	computes A + scale*(B - A)
		where A and B are Cell() and scale is a scalar
		P is Population()
	"""
	Avec = A.genes
	Bvec = B.genes
	vec = Avec + scale*(Bvec - Avec)
	return P.new_cell(vec)

def cell_to_centroid_cell(cells, P):
	"""
	creates cell that is the centroid of all current cells
	"""
	vecs = [cell.genes for cell in cells]
	vec = np.mean(np.vstack(vecs), axis=0)
	return P.new_cell(vec)

def update_population_NM(P, args):
	"""
	Nelder-Mead algorithm
	src: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

	alpha = 1
	gamma = 2
	rho = 0.5
	sigma = 0.5
	"""
	cells = sorted(P.cells, key=lambda cell: cell.fitness, reverse=True)
	worst_cell = cells[-1]
	best_cell = cells[0]

	# compute centroid
	centroid_cell = cell_to_centroid_cell(cells[:-1], P)

	# reflect
	reflected_cell = cell_from_vector_math(centroid_cell, worst_cell, -args.nm_alpha, P)
	if reflected_cell.fitness > cells[-2].fitness and reflected_cell.fitness < best_cell.fitness:
		P.cells = cells[:-1] + [reflected_cell]
		if args.verbose:
			print("1 - Reflected")
		return P

	# expansion
	if reflected_cell.fitness > best_cell.fitness:
		expanded_cell = cell_from_vector_math(centroid_cell, reflected_cell, args.nm_gamma, P)
		if expanded_cell.fitness > reflected_cell.fitness:
			P.cells = cells[:-1] + [expanded_cell]
			if args.verbose:
				print("2 - Expanded")
			return P
		else:
			P.cells = cells[:-1] + [reflected_cell]
			if args.verbose:
				print("2 - Expanded but reflected")
			return P

	# contraction
	assert(reflected_cell.fitness < cells[-2].fitness)
	contracted_cell = cell_from_vector_math(centroid_cell, worst_cell, args.nm_rho, P)
	if contracted_cell.fitness > worst_cell.fitness:
		P.cells = cells[:-1] + [contracted_cell]
		if args.verbose:
			print("3 - Contracted")
		return P

	# shrink: keep best, shrink the others
	if args.verbose:
		print("4 - Shrinking")
	new_cells = [best_cell]
	for cur_cell in cells[1:]:
		shrunk_cell = cell_from_vector_math(best_cell, cur_cell, args.nm_sigma, P)
		new_cells.append(shrunk_cell)
	P.cells = new_cells
	return P
