import random
import numpy as np

def update_population_DE(P, args):
	"""
	Differential Evolution
	src: "Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces"

	note: changes domain to be [-0.5, 0.5] rather than [0, 1]
	"""
	cells = P.cells
	ncells = len(cells)

	pinds = (ncells*np.random.rand(ncells,3)).astype(int)
	ps = np.random.rand(ncells)
	nfeatures = P.cells[0].ngenes
	muts = np.random.rand(ncells, nfeatures)
	probs = np.random.rand(ncells, nfeatures)
	new_cells = []
	n_new_cells = 0
	n_oob = 0
	best_cell = max(cells, key=lambda cell: cell.fitness)
	for i,cell in enumerate(cells):
		# pick 3 random other cells to form mutated vector
		# if ncells > 3000:
		# 	a,b,c = np.random.permutation(ncells)[:3]
		# else:
		# 	a,b,c = random.sample(range(ncells), 3)
		# vecA = cells[a].genes
		# vecB = cells[b].genes
		# vecC = cells[c].genes
		# vec_mutated = ps[i]*vecA + (1-ps[i])*vecB # convex combo
		# vec_mutated = vecA + args.de_F*(vecB - vecC)
		# vec_mutated[vec_mutated > 1.0] = 1.0
		# vec_mutated[vec_mutated < 0.0] = 0.0

		mut = (muts[i,:] - 0.5)/10.
		vec_mutated = cell.genes + mut
		vec_mutated[vec_mutated > 1.0] = 1.0
		vec_mutated[vec_mutated < 0.0] = 0.0
		
		# crossover cell's vec with vec_mutated to form new cell
		vec = cell.genes
		ix = (probs[i,:] < args.de_crossover_rate)
		if ix.sum() == 0:
			ix[int(nfeatures*np.random.rand())] = True
		vec[ix] = vec_mutated[ix]
		new_cell = P.new_cell(vec)
		# if ((cell.genes < cell.lbs) | (cell.genes > cell.ubs)).any():
		# 	n_oob += 1

		# evaluate, and keep only if we improved
		if new_cell.fitness > cell.fitness:
			new_cells.append(new_cell)
			n_new_cells += 1
		else:
			# might need to mutate here?
			new_cells.append(cell)
	print("{} of {} new cells".format(n_new_cells, ncells))
	P.cells = new_cells
	is_done = n_new_cells == 0
	return P, is_done

def cell_from_vector_math(A, B, scale, P):
	"""
	computes A + scale*(B - A) = (1 - scale)*A + scale*B
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
	is_done = False

	# compute centroid
	centroid_cell = cell_to_centroid_cell(cells[:-1], P)

	# reflect
	reflected_cell = cell_from_vector_math(centroid_cell, worst_cell, -args.nm_alpha, P)
	if reflected_cell.fitness > cells[-2].fitness and reflected_cell.fitness < best_cell.fitness:
		P.cells = cells[:-1] + [reflected_cell]
		if args.verbose:
			print("1 - Reflected")
		return P, is_done

	# expansion
	if reflected_cell.fitness >= best_cell.fitness:
		expanded_cell = cell_from_vector_math(centroid_cell, reflected_cell, args.nm_gamma, P)
		if expanded_cell.fitness > reflected_cell.fitness:
			P.cells = cells[:-1] + [expanded_cell]
			if args.verbose:
				print("2 - Expanded")
			return P, is_done
		else:
			P.cells = cells[:-1] + [reflected_cell]
			if args.verbose:
				print("2 - Expanded but reflected")
			return P, is_done

	# contraction
	assert(reflected_cell.fitness <= cells[-2].fitness)
	contracted_cell = cell_from_vector_math(centroid_cell, worst_cell, args.nm_rho, P)
	if contracted_cell.fitness > worst_cell.fitness:
		P.cells = cells[:-1] + [contracted_cell]
		if args.verbose:
			print("3 - Contracted")
		return P, is_done

	# shrink: keep best, shrink the others
	if args.verbose:
		print("4 - Shrinking")
	new_cells = [best_cell]
	for cur_cell in cells[1:]:
		shrunk_cell = cell_from_vector_math(best_cell, cur_cell, args.nm_sigma, P)
		new_cells.append(shrunk_cell)
	P.cells = new_cells
	return P, is_done
