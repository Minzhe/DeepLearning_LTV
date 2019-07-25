###############################################################
###                      utility.py                         ###
###############################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

def plot_corr_train_val(y_train, y_test, pred_train, pred_test, fig_path):
    f, ax = plt.subplots(2, 2, figsize=(16,16))
    plot_corr(y_train, pred_train, ax[0][0], 'Training set')
    plot_corr(y_test, pred_test, ax[0][1], 'Validation set')
    plot_density(y_train, ax[1][0])
    plot_density(y_test, ax[1][1])
    f.savefig(fig_path)

def plot_corr(truth, pred, ax, title=''):
    x = np.reshape(truth, -1)
    y = np.reshape(pred, -1)
    lims = min(min(x), min(y)), max(max(x), max(y))
    r2 = round(r2_score(x, y), 4)
    pearson_r = round(pearsonr(x, y)[0], 4)
    spearman_r = round(spearmanr(x, y)[0], 4)
    ax.scatter(x, y, s=5)
    ax.set_title(title)
    ax.set_xlabel('Truth')
    ax.set_ylabel('Predicted')
    ax.text(lims[0], lims[1], s='r2_score: {}\npearsonr: {}\nspearmanr: {}'.format(str(r2), str(pearson_r), str(spearman_r)), verticalalignment='top')
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)

def plot_density(x, ax):
    sns.distplot(x, ax=ax)