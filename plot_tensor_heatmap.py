import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def plot_tensor_heatmap(x, d, k, label=False, figsize=(20, 10), textsize=5):
	""" Plot a heatmap of a vector that has the same structure as a truncated  signature, that is a sum of tensors k+1
	tensors of order 1 up to k+1 in R^d.

	Parameters
	----------
	x: array, shape (n_points,)
		Array that is to be plotted. We must have n_points=(d^(k+1)-1)/(d-1)

	d: int
		Dimension of the underlying space.

	k: int
		Truncation order of a signature corresponding to x.

	label: boolean, default=False
		If label=True, the labels of the signature coefficients are plotted on the heatmap.

	figsize: tuple, default=(20,10)
		Size of the figure

	textsize: int
		Size of the text if label=True.

	Returns
	-------
	f, ax: object
		Instance of matplotlib.pyplot.subplots containing the figure.
	"""

	mat_coef = np.zeros((k + 1, d ** k))
	mask = np.zeros((k + 1, d ** k))
	annot = np.full((k + 1, d ** k), "", dtype='U256')
	count = 0
	for j in range(k+1):
		mat_coef[j, :d**j] = x[count:count+d**j]
		mask[j, d**j:] = True
		inner_count = 0
		for annot_label in itertools.product(np.arange(d)+1, repeat=j):
			annot[j, inner_count] = str(annot_label)
			inner_count += 1
		count += d**j
	with sns.axes_style("white"):
		f, ax = plt.subplots(figsize=figsize)
		if label:
			ax = sns.heatmap(
				mat_coef, mask=mask, xticklabels=False, center=0,
				cbar_kws={"orientation": "horizontal"}, annot=annot, fmt='',
				annot_kws={"size": textsize})
		else:
			ax = sns.heatmap(mat_coef, mask=mask, xticklabels=False, center=0, cbar_kws={"orientation": "horizontal"})
	return f, ax





