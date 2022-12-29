import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_covmats(
    list_covmat,
    list_pval=None,
    list_titles=None,
    main_title=None,
    delta_labels=None,
    scales=None,
    mask_diag=False,
):
    N = len(list_covmat)
    N_delta = list_covmat[0].shape[0]
    if mask_diag:
        mask_covmat = np.arange(N_delta)[:, None] >= np.arange(N_delta)
    else:
        mask_covmat = np.arange(N_delta)[:, None] > np.arange(N_delta)

    if delta_labels is None:
        delta_labels = [f"$\\Delta_{{{x}}}$" for x in range(N_delta)]

    fig, axs = plt.subplots(1, N, figsize=(5 * N + 5, 5))
    for i in range(N):
        tmp_mat = list_covmat[i].copy()
        tmp_mat[mask_covmat] = np.nan
        if scales is None:
            scale_max = np.max(np.abs([np.nanmin(tmp_mat), np.nanmax(tmp_mat)]))
        else:
            scale_max = scales[i]
        sns.heatmap(
            tmp_mat.T,
            cmap="vlag",
            vmin=-scale_max,
            vmax=scale_max,
            xticklabels=delta_labels,  # type: ignore
            yticklabels=delta_labels,  # type: ignore
            linewidths=0.5,  # type: ignore
            ax=axs[i],
        )
        if list_titles is not None:
            axs[i].set_title(list_titles[i])

        if list_pval is not None:
            if list_pval[i] is not None:
                sig = list_pval[i]
                if sig.dtype == float:
                    Bonf_alpha = 0.05 / (N_delta * (N_delta - 1) / 2 + N_delta)
                    sig = list_pval[i]
                    for j in range(sig.shape[0]):
                        for z in range(j, sig.shape[0]):
                            if sig[j, z] < Bonf_alpha:
                                _ = axs[i].text(
                                    j + 0.5, z + 0.5, "*", ha="center", va="center"
                                )
                if sig.dtype == bool:
                    for j in range(sig.shape[0]):
                        for z in range(j, sig.shape[0]):
                            if sig[j, z]:
                                _ = axs[i].text(
                                    j + 0.5, z + 0.5, "*", ha="center", va="center"
                                )

    fig.subplots_adjust(bottom=0.2)
    if main_title is not None:
        fig.suptitle(main_title)
    return fig
