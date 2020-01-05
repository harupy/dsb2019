import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


class JointConfusionMatrix:
    """
    Ref. https://github.com/mwaskom/seaborn/blob/master/seaborn/axisgrid.py#L1551
    """

    def __init__(self, cm, height=6, ratio=5, space=.2,
                 dropna=True, xlim=None, ylim=None, size=None):

        # set up the subplot grid
        f = plt.figure(figsize=(height, height))
        gs = plt.GridSpec(ratio + 1, ratio + 1)

        ax_joint = f.add_subplot(gs[1:, :-1])
        ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
        ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)

        self.fig = f
        self.ax_joint = ax_joint
        self.ax_marg_x = ax_marg_x
        self.ax_marg_y = ax_marg_y
        self.cm = cm

        # turn off tick visibility for the measure axis on the marginal plots
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)

        # utrn off the ticks on the density axis for the marginal plots
        plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
        plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
        plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
        plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
        plt.setp(ax_marg_x.get_yticklabels(), visible=False)
        plt.setp(ax_marg_y.get_xticklabels(), visible=False)

        ax_marg_x.yaxis.grid(False)
        ax_marg_y.xaxis.grid(False)

        if xlim is not None:
            ax_joint.set_xlim(xlim)
        if ylim is not None:
            ax_joint.set_ylim(ylim)

        # make the grid look nice
        sns.utils.despine(f)
        sns.utils.despine(ax=ax_marg_x, left=True)
        sns.utils.despine(ax=ax_marg_y, bottom=True)
        f.tight_layout()
        f.subplots_adjust(hspace=space, wspace=space)

    def make_annotation(self, cm, cm_norm, normalize=True):
        annot = []
        nrows, ncols = cm.shape
        base = '{}\n({:.2f})'
        for ir in range(nrows):
            annot.append([])
            for ic in range(ncols):
                annot[ir].append(base.format(cm[ir, ic], cm_norm[ir, ic]))

        return np.array(annot)

    def plot(self, labels=None, normalize=True):
        labels = [i for i in range(len(self.cm))] if labels is None else labels

        true_dist = self.cm.sum(axis=1)
        pred_dist = self.cm.sum(axis=0)
        pos = np.arange(self.cm.shape[0]) + 0.5

        # normalize
        cm_norm = self.cm / true_dist.reshape(-1, 1)
        annot = self.make_annotation(self.cm, cm_norm)

        FONTSIZE = 20

        # plot confusion matrix as a heatmap
        sns.heatmap(cm_norm, cmap='Blues', vmin=0, vmax=1,
                    annot=annot, fmt='s', annot_kws={'fontsize': FONTSIZE},
                    linewidths=0.2, cbar=False, square=True, ax=self.ax_joint)
        self.ax_joint.set_xlabel('Predicted label', fontsize=FONTSIZE)
        self.ax_joint.set_ylabel('True label', fontsize=FONTSIZE)
        self.ax_joint.set_xticklabels(labels, fontsize=FONTSIZE)
        self.ax_joint.set_yticklabels(labels, fontsize=FONTSIZE)

        props = {'align': 'center'}

        # plot label distribution
        self.ax_marg_x.bar(pos, pred_dist / pred_dist.sum(), **props)
        self.ax_marg_y.barh(pos, true_dist / true_dist.sum(), **props)


def plot_confusion_matrix(cm):
    g = JointConfusionMatrix(cm, height=10)
    g.plot()
    g.fig.tight_layout()
    return g.fig


def plot_feature_importance(feature_names, importance, importance_type, limit=30):
    """
    Plot feature importance and return the figure.
    """
    indices = np.argsort(importance)[-limit:]
    y = np.arange(len(indices))

    fig, ax = plt.subplots()
    ax.barh(y, importance[indices], align='center', height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(feature_names[indices])
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'Feature Importance: {importance_type}')
    fig.tight_layout()
    return fig


def plot_label_share(labels):
    unique, counts = np.unique(labels, return_counts=True)

    counts_norm = counts / counts.sum()
    fig, ax = plt.subplots()
    bar = sns.barplot(unique, counts_norm)
    for idx, p in enumerate(bar.patches):
        bar.annotate('{:.2f}\n({})'.format(counts_norm[idx], counts[idx]),
                     (p.get_x() + p.get_width() / 2, p.get_height() / 2),
                     ha='center', va='center', color='white', fontsize='large')
    ax.set_xlabel('Label')
    ax.set_ylabel('Share')
    ax.set_title('Label Share')
    fig.tight_layout()
    return fig


def plot_eval_history(eval_results):
    fig, ax = plt.subplots()
    for fold_idx, eval_result in enumerate(eval_results):
        for data_name, metrics in eval_result.items():
            for metric_name, values in metrics.items():
                label = f'{data_name}-{metric_name}-{fold_idx}'
                ax.plot(values, label=label, zorder=1)[0]
                # ax.scatter(data['best_iteration'], data['values'][data['best_iteration'] - 1],
                #            s=60, c=[line.get_color()], edgecolors='k', linewidths=1, zorder=2)
    ax.set_xlabel('Iteration')
    ax.set_title('Evaluation History (marker on each line represents the best iteration)')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    return fig
