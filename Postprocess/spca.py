#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ristretto.pca import SPCA


def orthogonalMetricVar(ndata, inds):
    Xtrqro = np.linalg.qr(ndata)
    Xtrro = Xtrqro[1]
    varO = 1 / ndata.shape[0] * np.diag(Xtrro) ** 2  # Orthogonal variance

    X_man = ndata[:, inds]
    Xtrqrm = np.linalg.qr(X_man)
    Xtrrm = Xtrqrm[1]
    varM = 1 / ndata.shape[0] * np.diag(Xtrrm) ** 2  # Orthogonal variance
    evM = varM / np.sum(varO)

    return np.sum(evM)


def sensitivity(ndata, metLab, nComp, savePath):
    fig, axs = plt.subplots(
        nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8.75, 3.5)
    )
    for i in range(2):
        for j in range(2):
            nTr = ndata.copy()
            if i == 0:
                lamb = 0.1
            else:
                lamb = 0.01
            if j == 1:
                nTr[:, 9] = 0  # Remove SCAI

            cbax = fig.add_axes([1.0, 0.35, 0.01, 0.6])

            spca = SPCA(alpha=lamb, n_components=nComp)
            spca.fit(nTr)

            axs[i, j] = sns.heatmap(
                spca.B_.T ** 2,
                cmap="cubehelix_r",
                ax=axs[i, j],
                vmin=0,
                vmax=1,
                cbar_ax=cbax,
            )
            axs[i, j].set_axis_on()

            if i == 1:
                if j == 0:
                    axs[i, j].set_xlabel("Metric \n Full metric set")
                else:
                    axs[i, j].set_xlabel("Metric \n Missing mean length")
                axs[i, j].set_xticklabels(metLab, rotation="vertical")
            if j == 0:
                axs[i, j].set_ylabel(r"$\lambda = $" + str(lamb) + "\n" r"Sparse PC")
            axs[i, j].set_yticklabels(np.arange(1, nComp + 1))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(savePath + "/sensSPCA.pdf", bbox_inches="tight")
