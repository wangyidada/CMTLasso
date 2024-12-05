import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from matplotlib.ticker import MultipleLocator


def plot_error(alphas, mse, alpha, save_folder):
    mse_mean =np.mean(mse, axis=1)
    mse_std = np.std(mse, axis=1)
    plt.figure()
    plt.errorbar(
        alphas,
        mse_mean,
        yerr=mse_std,
        ms=3,
        fmt='o',
        mfc='r',
        mec='r',
        ecolor='lightblue',
        elinewidth=2,
        capsize=4,
        capthick=1
    )
    plt.semilogx()
    plt.axvline(alpha, color='black', ls='--')
    
    plt.xlabel('alpha')
    plt.ylabel('MSE')
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    plt.savefig(os.path.join(save_folder, 'error_figure.jpg'), dpi=600)
    # plt.show()


def plot_coefs(alphas, coefs, alpha, save_folder):
    if len(coefs.shape) == 3:
        for i in range(np.shape(coefs)[2]):
            plt.figure()
            plt.title('Task'+str(i+1))
            plt.semilogx(alphas, coefs[..., i])
            plt.axvline(alpha, color='black', ls='--')
            plt.xlabel('alpha')
            plt.ylabel('Cofficients')
            plt.savefig(os.path.join(save_folder, 'coefs_figure' + str(i) + '.jpg'), dpi=600)
    elif len(coefs.shape) == 2:
            plt.figure()
            plt.title('Task')
            plt.semilogx(alphas, coefs)
            plt.axvline(alpha, color='black', ls='--')
            plt.xlabel('alpha')
            plt.ylabel('Cofficients')
            plt.savefig(os.path.join(save_folder, 'coefs_figure.jpg'), dpi=600)


def get_pred_values(files,  pred_name='pred', label_name='label'):
    labels, scores=[], []
    for file in files:
        df = pd.read_csv(file)
        target = df[label_name].values
        pred = df[pred_name].values
        score = np.asarray(list(pred), dtype=np.float32)
        label = np.asarray(list(target), dtype=np.int32)
        labels.append(label)
        scores.append(score)
    return labels, scores


def plot_roc(labels, scores, name_list=['Trainging ROC'], save_file=None, dpi=300):
    colors_list=['blue', 'green', 'orange', 'red', 'purple']
    lw = 1.5
    size = 12
    plt.figure(figsize=(8, 6))

    for i, (label, score) in enumerate(zip(labels, scores)):
        FPR, TPR, t = roc_curve(label, score)
        auc_value = auc(FPR, TPR)
        plt.plot(FPR, TPR, color=colors_list[i], lw=lw, label=name_list[i] +' (AUC = %0.3f)' % auc_value)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity', size=size)
    plt.ylabel('Sensitivity', size=size)
    plt.legend(loc="lower right", fontsize=size - 2)
    plt.title('ROC', size=16)
    plt.tick_params(labelsize=size)
    plt.savefig(save_file, dpi=dpi)
    plt.show()
    plt.close()


def Plot_ROCs(files, pred_name='pred0', label_name='label0', save_file=None, name_list=['Trainging ROC', 'Test ROC']):
    labels, scores = get_pred_values(files,  pred_name, label_name)
    plot_roc(labels, scores, name_list=name_list, save_file=save_file)
