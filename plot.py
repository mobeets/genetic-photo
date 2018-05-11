import os.path
import pandas as pd
import matplotlib.pyplot as plt

f1 = 'logs/nn_puzzle_hard_nonn_v2/history.csv'
# f2 = 'logs/nn_puzzle_hard_nn_every2/history.csv'
f2 = 'logs/nn_puzzle_hard_nn_every20/history.csv'

# fs = ['nnp' + str(i) for i in ['',2,3,4]]
fs = ['nnp' + str(i) for i in ['']]
fs += ['nnp_' + str(i) for i in [2,5,20,50]]

fs = ['nnp' + str(i) for i in ['']]
fs += ['nnp_d' + str(i) for i in[1,2,3]]

fs = ['nnp_d1_v1', 'nnp_d1_v2']

fs = ['logs/' + f + '/history.csv' for f in fs]
clrs = ['k','r','g','b','c']

outfile = 'logs/vs1.png'
key = 'y_max'
range_is_0_1 = True
# key = 'n_improved_best'
# range_is_0_1 = False

fig = plt.figure()
ax = fig.add_subplot(111)
for i,f in enumerate(fs):
	df = pd.read_csv(f)
	xs = df.index.values
	ix = df['used_NN'].values == 1.0
	plt.plot(xs, df[key].values, clrs[i]+'.-')
	# plt.plot(xs[ix], df[key].values[ix], 'r.')
	# plt.plot(xs[~ix], df[key].values[~ix], 'k.')

plt.ylim(0, 100.)

if range_is_0_1:
	plt.ylim(0, 1.)
plt.xlabel('index')
plt.ylabel(key)
if outfile is None:
    plt.show()
else:
    fig.savefig(outfile)
    plt.close(fig)
