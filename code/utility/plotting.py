###############################################################
###                      utility.py                         ###
###############################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

def plot_corr_train_val(y_train, y_val, y_test, pred_train, pred_val, pred_test, fig_path, lims=None):
    f, ax = plt.subplots(2, 3, figsize=(24,16))
    plot_corr(y_train, pred_train, ax[0][0], 'Training set', lims)
    plot_corr(y_val, pred_val, ax[0][1], 'Validation set', lims)
    plot_corr(y_test, pred_test, ax[0][2], 'Test set', lims)
    plot_density(y_train, ax[1][0])
    plot_density(y_val, ax[1][1])
    plot_density(y_test, ax[1][2])
    f.savefig(fig_path)

def plot_corr(truth, pred, ax, title='', lims=None):
    x = np.reshape(truth, -1)
    y = np.reshape(pred, -1)
    border = min(min(x), min(y)), max(max(x), max(y))
    r2 = round(r2_score(x, y), 4)
    pearson_r = round(pearsonr(x, y)[0], 4)
    spearman_r = round(spearmanr(x, y)[0], 4)
    ax.scatter(x, y, s=5)
    ax.set_title(title)
    if lims is not None:
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    ax.set_xlabel('Truth')
    ax.set_ylabel('Predicted')
    if lims is not None:
        ax.text(lims[0], lims[1], s='r2_score: {}\npearsonr: {}\nspearmanr: {}'.format(str(r2), str(pearson_r), str(spearman_r)), verticalalignment='top')
    else:
        ax.text(border[0], border[1], s='r2_score: {}\npearsonr: {}\nspearmanr: {}'.format(str(r2), str(pearson_r), str(spearman_r)), verticalalignment='top')
    ax.plot(border, border, 'r--', alpha=0.75, zorder=0)

def plot_density(x, ax):
    sns.distplot(x, ax=ax)

def plot_scatter(x, y, fig_path, s=None, c=None, cmap=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.scatter(x, y, s=s, c=c, cmap=cmap)
    fig.colorbar(im, ax=ax)
    fig.savefig(fig_path, transparent=True)