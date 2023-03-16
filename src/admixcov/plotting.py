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

#======================================
# plotting of confidence interval data

def plot_ci_line(
    x: np.ndarray | list,
    CI: tuple,
    ax,
    color='black',
    **kwargs
):
    lower = CI[0]
    m =CI[1]
    upper = CI[2]
    yerr = np.array([m - lower, upper - m])
    ax.errorbar(x, m, yerr, color=color, ecolor=color, **kwargs)

def cov_lineplot(times, CIs: list[tuple], ax, colors, time_padding=0, **kwargs):
    k = len(times) - 1
    for i in range(k-1):
        plot_ci_line(np.array(times[i+1:-1]), np.stack(CIs)[:, i, i+1:], ax, color=colors[i], **kwargs)
    ax.hlines(y=0, xmin=0, xmax=times[1] + time_padding, linestyles='dotted', colors='black')
    ax.set_xlim(times[1] + time_padding, -time_padding)
    ax.set_xlabel('time')
    ax.set_ylabel('covariance')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def combine_covmat_CIs(ci_l: np.ndarray, ci_u: np.ndarray):
	N = ci_l[1].shape[0]
	res = tuple([x.copy() for x in ci_l])
	tri_up = np.triu_indices(N, k=1)
	for k in range(3):
		res[k][np.arange(N)[:, None] == np.arange(N)] = np.nan
		res[k][tri_up] = ci_u[k][tri_up]
	return res

def plot_covmat_ci(CI, ax, scale_max=None):
    N_delta = CI[1].shape[0]
    delta_labels = [f"$\\Delta_{{{x}}}$" for x in range(N_delta)]
    tmp_mat = CI[1].copy()
    if scale_max is None:
        scale_max = np.max(np.abs([np.nanmin(tmp_mat), np.nanmax(tmp_mat)]))
    sns.heatmap(
        tmp_mat.T,
        cmap="vlag",
        vmin=-scale_max,
        vmax=scale_max,
        xticklabels=delta_labels,  # type: ignore
        yticklabels=delta_labels,  # type: ignore
        linewidths=0.5,  # type: ignore
        ax=ax,
    )
    sig = (CI[0] * CI[2]) > 0
    for z in range(sig.shape[0]):
        for j in range(sig.shape[0]):
            if (sig[j, z]) & (z != j):
                _ = ax.text(
                    j + 0.5, z + 0.5, "*", ha="center", va="center"
                )
    ax.axline(xy1=(N_delta, N_delta), slope=1, color='black')