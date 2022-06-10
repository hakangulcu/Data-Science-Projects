"""
Hakan Gulcu
21702275
GE461 Project 2

For question 3, I will use a GitHub repo (https://github.com/tompollard/sammon) for Sammon Mapping.
I will use sklearn again for t-SNE

Question 3
"""

from sammon import sammon as SammonMapping
from sklearn.manifold import TSNE
import numpy as np
import matplotlib
import matplotlib.pyplot as plot
import scipy.io as loader

"""
Loadmat is a part of scipy.io external library to load .mat files in python.
Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html
"""
digits = loader.loadmat('digits.mat')
labels = digits['labels']
features = digits['digits']

##################### SAMMON ################################

#def sammon(x, n, display = 2, inputdist = 'raw', maxhalves = 20, maxiter = 500, tolfun = 1e-9, init = 'default'):
[x,E] = SammonMapping(features, n=2, maxiter=100)

plot.scatter(x[labels[:,0] == 0, 0], x[labels[:,0] == 0, 1], marker='.', label="0")
plot.scatter(x[labels[:,0] == 1, 0], x[labels[:,0] == 1, 1], marker='.', label="1")
plot.scatter(x[labels[:,0] == 2, 0], x[labels[:,0] == 2, 1], marker='.', label="2")
plot.scatter(x[labels[:,0] == 3, 0], x[labels[:,0] == 3, 1], marker='.', label="3")
plot.scatter(x[labels[:,0] == 4, 0], x[labels[:,0] == 4, 1], marker='.', label="4")
plot.scatter(x[labels[:,0] == 5, 0], x[labels[:,0] == 5, 1], marker='.', label="5")
plot.scatter(x[labels[:,0] == 6, 0], x[labels[:,0] == 6, 1], marker='.', label="6")
plot.scatter(x[labels[:,0] == 7, 0], x[labels[:,0] == 7, 1], marker='.', label="7")
plot.scatter(x[labels[:,0] == 8, 0], x[labels[:,0] == 8, 1], marker='.', label="8")
plot.scatter(x[labels[:,0] == 9, 0], x[labels[:,0] == 9, 1], marker='.', label="9")
plot.title('Sammon Mapping for 100 Iterations')
plot.legend(['0','1','2','3','4','5','6','7','8','9'], loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plot.show()

##################### TSNE ################################

"""
class sklearn.manifold.TSNE(n_components=2, *, perplexity=30.0, early_exaggeration=12.0, learning_rate='warn', n_iter=1000, n_iter_without_progress=300, 
min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')
"""

################ Experiment according to Perplexity ################

tSNE = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate='warn', n_iter=1000, n_iter_without_progress=300, 
min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=0, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')
x = tSNE.fit_transform(features)

plot.scatter(x[labels[:,0] == 0, 0], x[labels[:,0] == 0, 1], marker='.', label="0")
plot.scatter(x[labels[:,0] == 1, 0], x[labels[:,0] == 1, 1], marker='.', label="1")
plot.scatter(x[labels[:,0] == 2, 0], x[labels[:,0] == 2, 1], marker='.', label="2")
plot.scatter(x[labels[:,0] == 3, 0], x[labels[:,0] == 3, 1], marker='.', label="3")
plot.scatter(x[labels[:,0] == 4, 0], x[labels[:,0] == 4, 1], marker='.', label="4")
plot.scatter(x[labels[:,0] == 5, 0], x[labels[:,0] == 5, 1], marker='.', label="5")
plot.scatter(x[labels[:,0] == 6, 0], x[labels[:,0] == 6, 1], marker='.', label="6")
plot.scatter(x[labels[:,0] == 7, 0], x[labels[:,0] == 7, 1], marker='.', label="7")
plot.scatter(x[labels[:,0] == 8, 0], x[labels[:,0] == 8, 1], marker='.', label="8")
plot.scatter(x[labels[:,0] == 9, 0], x[labels[:,0] == 9, 1], marker='.', label="9")
plot.title('t-SNE Mapping for 1000 Iterations, Perplexity = 30')
plot.legend(['0','1','2','3','4','5','6','7','8','9'], loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plot.show()

tSNE = TSNE(n_components=2, perplexity=20.0, early_exaggeration=12.0, learning_rate='warn', n_iter=1000, n_iter_without_progress=300, 
min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=0, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')
x = tSNE.fit_transform(features)

plot.scatter(x[labels[:,0] == 0, 0], x[labels[:,0] == 0, 1], marker='.', label="0")
plot.scatter(x[labels[:,0] == 1, 0], x[labels[:,0] == 1, 1], marker='.', label="1")
plot.scatter(x[labels[:,0] == 2, 0], x[labels[:,0] == 2, 1], marker='.', label="2")
plot.scatter(x[labels[:,0] == 3, 0], x[labels[:,0] == 3, 1], marker='.', label="3")
plot.scatter(x[labels[:,0] == 4, 0], x[labels[:,0] == 4, 1], marker='.', label="4")
plot.scatter(x[labels[:,0] == 5, 0], x[labels[:,0] == 5, 1], marker='.', label="5")
plot.scatter(x[labels[:,0] == 6, 0], x[labels[:,0] == 6, 1], marker='.', label="6")
plot.scatter(x[labels[:,0] == 7, 0], x[labels[:,0] == 7, 1], marker='.', label="7")
plot.scatter(x[labels[:,0] == 8, 0], x[labels[:,0] == 8, 1], marker='.', label="8")
plot.scatter(x[labels[:,0] == 9, 0], x[labels[:,0] == 9, 1], marker='.', label="9")
plot.title('t-SNE Mapping for 1000 Iterations, Perplexity = 20')
plot.legend(['0','1','2','3','4','5','6','7','8','9'], loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plot.show()

tSNE = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate='warn', n_iter=1500, n_iter_without_progress=300, 
min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=0, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')
x = tSNE.fit_transform(features)

plot.scatter(x[labels[:,0] == 0, 0], x[labels[:,0] == 0, 1], marker='.', label="0")
plot.scatter(x[labels[:,0] == 1, 0], x[labels[:,0] == 1, 1], marker='.', label="1")
plot.scatter(x[labels[:,0] == 2, 0], x[labels[:,0] == 2, 1], marker='.', label="2")
plot.scatter(x[labels[:,0] == 3, 0], x[labels[:,0] == 3, 1], marker='.', label="3")
plot.scatter(x[labels[:,0] == 4, 0], x[labels[:,0] == 4, 1], marker='.', label="4")
plot.scatter(x[labels[:,0] == 5, 0], x[labels[:,0] == 5, 1], marker='.', label="5")
plot.scatter(x[labels[:,0] == 6, 0], x[labels[:,0] == 6, 1], marker='.', label="6")
plot.scatter(x[labels[:,0] == 7, 0], x[labels[:,0] == 7, 1], marker='.', label="7")
plot.scatter(x[labels[:,0] == 8, 0], x[labels[:,0] == 8, 1], marker='.', label="8")
plot.scatter(x[labels[:,0] == 9, 0], x[labels[:,0] == 9, 1], marker='.', label="9")
plot.title('t-SNE Mapping for 1500 Iterations, Perplexity = 30')
plot.legend(['0','1','2','3','4','5','6','7','8','9'], loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plot.show()

tSNE = TSNE(n_components=2, perplexity=20.0, early_exaggeration=12.0, learning_rate='warn', n_iter=1500, n_iter_without_progress=300, 
min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=0, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')
x = tSNE.fit_transform(features)

plot.scatter(x[labels[:,0] == 0, 0], x[labels[:,0] == 0, 1], marker='.', label="0")
plot.scatter(x[labels[:,0] == 1, 0], x[labels[:,0] == 1, 1], marker='.', label="1")
plot.scatter(x[labels[:,0] == 2, 0], x[labels[:,0] == 2, 1], marker='.', label="2")
plot.scatter(x[labels[:,0] == 3, 0], x[labels[:,0] == 3, 1], marker='.', label="3")
plot.scatter(x[labels[:,0] == 4, 0], x[labels[:,0] == 4, 1], marker='.', label="4")
plot.scatter(x[labels[:,0] == 5, 0], x[labels[:,0] == 5, 1], marker='.', label="5")
plot.scatter(x[labels[:,0] == 6, 0], x[labels[:,0] == 6, 1], marker='.', label="6")
plot.scatter(x[labels[:,0] == 7, 0], x[labels[:,0] == 7, 1], marker='.', label="7")
plot.scatter(x[labels[:,0] == 8, 0], x[labels[:,0] == 8, 1], marker='.', label="8")
plot.scatter(x[labels[:,0] == 9, 0], x[labels[:,0] == 9, 1], marker='.', label="9")
plot.title('t-SNE Mapping for 1500 Iterations, Perplexity = 20')
plot.legend(['0','1','2','3','4','5','6','7','8','9'], loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plot.show()

tSNE = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate='warn', n_iter=2500, n_iter_without_progress=300, 
min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=0, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')
x = tSNE.fit_transform(features)

plot.scatter(x[labels[:,0] == 0, 0], x[labels[:,0] == 0, 1], marker='.', label="0")
plot.scatter(x[labels[:,0] == 1, 0], x[labels[:,0] == 1, 1], marker='.', label="1")
plot.scatter(x[labels[:,0] == 2, 0], x[labels[:,0] == 2, 1], marker='.', label="2")
plot.scatter(x[labels[:,0] == 3, 0], x[labels[:,0] == 3, 1], marker='.', label="3")
plot.scatter(x[labels[:,0] == 4, 0], x[labels[:,0] == 4, 1], marker='.', label="4")
plot.scatter(x[labels[:,0] == 5, 0], x[labels[:,0] == 5, 1], marker='.', label="5")
plot.scatter(x[labels[:,0] == 6, 0], x[labels[:,0] == 6, 1], marker='.', label="6")
plot.scatter(x[labels[:,0] == 7, 0], x[labels[:,0] == 7, 1], marker='.', label="7")
plot.scatter(x[labels[:,0] == 8, 0], x[labels[:,0] == 8, 1], marker='.', label="8")
plot.scatter(x[labels[:,0] == 9, 0], x[labels[:,0] == 9, 1], marker='.', label="9")
plot.title('t-SNE Mapping for 2500 Iterations, Perplexity = 30')
plot.legend(['0','1','2','3','4','5','6','7','8','9'], loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plot.show()

tSNE = TSNE(n_components=2, perplexity=20.0, early_exaggeration=12.0, learning_rate='warn', n_iter=2500, n_iter_without_progress=300, 
min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=0, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')
x = tSNE.fit_transform(features)

plot.scatter(x[labels[:,0] == 0, 0], x[labels[:,0] == 0, 1], marker='.', label="0")
plot.scatter(x[labels[:,0] == 1, 0], x[labels[:,0] == 1, 1], marker='.', label="1")
plot.scatter(x[labels[:,0] == 2, 0], x[labels[:,0] == 2, 1], marker='.', label="2")
plot.scatter(x[labels[:,0] == 3, 0], x[labels[:,0] == 3, 1], marker='.', label="3")
plot.scatter(x[labels[:,0] == 4, 0], x[labels[:,0] == 4, 1], marker='.', label="4")
plot.scatter(x[labels[:,0] == 5, 0], x[labels[:,0] == 5, 1], marker='.', label="5")
plot.scatter(x[labels[:,0] == 6, 0], x[labels[:,0] == 6, 1], marker='.', label="6")
plot.scatter(x[labels[:,0] == 7, 0], x[labels[:,0] == 7, 1], marker='.', label="7")
plot.scatter(x[labels[:,0] == 8, 0], x[labels[:,0] == 8, 1], marker='.', label="8")
plot.scatter(x[labels[:,0] == 9, 0], x[labels[:,0] == 9, 1], marker='.', label="9")
plot.title('t-SNE Mapping for 2500 Iterations, Perplexity = 20')
plot.legend(['0','1','2','3','4','5','6','7','8','9'], loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plot.show()

tSNE = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate='warn', n_iter=4500, n_iter_without_progress=300, 
min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=0, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')
x = tSNE.fit_transform(features)

plot.scatter(x[labels[:,0] == 0, 0], x[labels[:,0] == 0, 1], marker='.', label="0")
plot.scatter(x[labels[:,0] == 1, 0], x[labels[:,0] == 1, 1], marker='.', label="1")
plot.scatter(x[labels[:,0] == 2, 0], x[labels[:,0] == 2, 1], marker='.', label="2")
plot.scatter(x[labels[:,0] == 3, 0], x[labels[:,0] == 3, 1], marker='.', label="3")
plot.scatter(x[labels[:,0] == 4, 0], x[labels[:,0] == 4, 1], marker='.', label="4")
plot.scatter(x[labels[:,0] == 5, 0], x[labels[:,0] == 5, 1], marker='.', label="5")
plot.scatter(x[labels[:,0] == 6, 0], x[labels[:,0] == 6, 1], marker='.', label="6")
plot.scatter(x[labels[:,0] == 7, 0], x[labels[:,0] == 7, 1], marker='.', label="7")
plot.scatter(x[labels[:,0] == 8, 0], x[labels[:,0] == 8, 1], marker='.', label="8")
plot.scatter(x[labels[:,0] == 9, 0], x[labels[:,0] == 9, 1], marker='.', label="9")
plot.title('t-SNE Mapping for 4500 Iterations, Perplexity = 30')
plot.legend(['0','1','2','3','4','5','6','7','8','9'], loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plot.show()

tSNE = TSNE(n_components=2, perplexity=20.0, early_exaggeration=12.0, learning_rate='warn', n_iter=4500, n_iter_without_progress=300, 
min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=0, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')
x = tSNE.fit_transform(features)

plot.scatter(x[labels[:,0] == 0, 0], x[labels[:,0] == 0, 1], marker='.', label="0")
plot.scatter(x[labels[:,0] == 1, 0], x[labels[:,0] == 1, 1], marker='.', label="1")
plot.scatter(x[labels[:,0] == 2, 0], x[labels[:,0] == 2, 1], marker='.', label="2")
plot.scatter(x[labels[:,0] == 3, 0], x[labels[:,0] == 3, 1], marker='.', label="3")
plot.scatter(x[labels[:,0] == 4, 0], x[labels[:,0] == 4, 1], marker='.', label="4")
plot.scatter(x[labels[:,0] == 5, 0], x[labels[:,0] == 5, 1], marker='.', label="5")
plot.scatter(x[labels[:,0] == 6, 0], x[labels[:,0] == 6, 1], marker='.', label="6")
plot.scatter(x[labels[:,0] == 7, 0], x[labels[:,0] == 7, 1], marker='.', label="7")
plot.scatter(x[labels[:,0] == 8, 0], x[labels[:,0] == 8, 1], marker='.', label="8")
plot.scatter(x[labels[:,0] == 9, 0], x[labels[:,0] == 9, 1], marker='.', label="9")
plot.title('t-SNE Mapping for 4500 Iterations, Perplexity = 20')
plot.legend(['0','1','2','3','4','5','6','7','8','9'], loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plot.show()

