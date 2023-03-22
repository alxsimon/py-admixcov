# Set of functions to compute covariances
# and error terms

import numpy as np

# Starting point of the functions here:
# - af array of allele frequencies of shape (T, N_loci)
# - sample_size array of sample sizes of shape (T, [1, N_loci])
# 	which depend on wether there is missing data or not.
# - ref_af array of allele frequencies for each reference pop
# 	of shape (N_r, N_loci).

# let's define some things:
# - Q is a mean admixture array of shape (T, N_r),
# 	T is number of time points and N_r number of reference pops.
# - dQ is the \Delta_q array, shape (T - 1, N_r).
# - A is the array of \alpha for each time interval
# 	of shape (T - 1, N_r).
# - var_drift is the variance of drift for each time interval
# 	of shape (T - 1,).


def get_pseudohap_sampling_bias(
    af: np.ndarray,
    sample_size: np.ndarray,
):
    hh = af * (1 - af)
    # in case of NaNs (sample_size 0 or 1)
    tmp = (sample_size - 1).astype(float)
    tmp[tmp <= 0] = np.nan
    sample_correction = 1 / tmp
    # array shape adjustment
    if sample_correction.ndim != hh.ndim:
        if sample_correction.ndim == 1:
            sample_correction = sample_correction[:, np.newaxis]
    bias = np.nanmean(hh * sample_correction, axis=1)
    return bias


def kth_diag_indices(a, k=0):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def create_bias_correction_matrix(b):
    size = len(b) - 1
    M = np.zeros((size, size))
    M[kth_diag_indices(M, k=0)] += b[:-1] + b[1:]
    M[kth_diag_indices(M, k=1)] -= b[1:-1]
    M[kth_diag_indices(M, k=-1)] -= b[1:-1]
    return M


def get_bias_matrix(af, sample_size):
    b = get_pseudohap_sampling_bias(af, sample_size)
    M = create_bias_correction_matrix(b)
    return M


def get_ref_bias_matrix(ref_af, sample_size, dQ):
    b = get_pseudohap_sampling_bias(ref_af, sample_size)
    M = np.zeros((dQ.shape[0], dQ.shape[0]))
    for i in range(M.shape[0]):
        for j in range(M.shape[0]):
            M[i, j] = np.sum(dQ[i] * dQ[j] * b)
    return M


def get_covariance_matrix(
    af,
    bias=False,
    sample_size=None,
):
    # determine if af has NAs
    covmat = np.ma.cov(np.ma.masked_invalid(np.diff(af, axis=0))).data
    # determine bias
    if bias:
        covmat -= get_bias_matrix(af, sample_size)
    return covmat


def get_admix_covariance_matrix(
    Q,
    ref_af,
    bias=False,
    ref_sample_size=None,
):
    admix_af = Q @ ref_af
    admix_cov = np.ma.cov(np.ma.masked_invalid(np.diff(admix_af, axis=0))).data
    # determine bias
    if bias:
        dQ = np.diff(Q, axis=0)
        admix_cov -= get_ref_bias_matrix(ref_af, ref_sample_size, dQ)
        # subtract so that it's an addition when subtracting it from raw covmat
    return admix_cov


# ==================================================
# Drift errors

# alpha_mask = np.array([
# 	[0, 0, 1],
# 	[0, 1, 0],
# 	[0, 1, 0],
# 	[0, 1, 0],
# 	[1, 0, 0],
# 	[0, 1, 0],
# ], dtype=bool) # says which alpha is different from zero


def q2a_simple(Q, alpha_mask):
    # alpha_mask boolean array
    if (alpha_mask.sum(axis=1) > 1).any():
        raise ValueError("Only one alpha per time can be different from 0.")
    dQ = np.diff(Q, axis=0)
    dQ[~alpha_mask] = 0  # shortcut
    denom = 1 - Q[:-1]
    denom[~alpha_mask] = 99  # avoid 0 when Q = 1
    A = dQ / denom
    return A


def solve_for_variances(diag_V, A):
    # diag_V: diagonal of admixture and sampling bias corrected covariance matrix
    # A: vector or array of alphas of shape (T, N_pop - 1)
    # returns the $\Delta d_t$ vector
    m = diag_V.size
    B = diag_V[:, np.newaxis]
    C = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1):
            if j == i:
                C[i, j] = 1
            elif j == (i - 1):
                C[i, j] = (A[i].sum()) ** 2
            else:
                prod_l = 1
                for l in range(j + 1, i):
                    prod_l *= (1 - A[l].sum()) ** 2
                C[i, j] += (A[i].sum()) ** 2 * prod_l
    X = np.linalg.solve(C, B)
    return X[:, 0]


def indic(x):
    if x > 0:
        return 1
    else:
        return 0


def get_drift_err(var_drift, i, j, A):
    # var_drift and A are arrays
    if i == j:
        return 0
    elif i > j:
        tmp = j
        j = i
        i = tmp
    if j > (len(var_drift) - 1):
        raise Exception("j too large")

    res = 0
    for k in range(i + 1):
        m_i = 1
        m_j = 1
        for l in range(k + 1, i):
            m_i *= 1 - A[l].sum()
        for l in range(k + 1, j):
            m_j *= 1 - A[l].sum()
        res += (
            var_drift[k]
            * (-A[i].sum()) ** indic(i - k)
            * (m_i) ** indic(i - k - 1)
            * (-A[j].sum())
            * (m_j) ** indic(j - k - 1)
        )
    return res


def get_drift_err_matrix(var_drift, A):
    size = len(var_drift)
    res = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            res[i, j] = get_drift_err(var_drift, i, j, A)
    return res


# ================================================================
# Summaries

def get_tot_var(af: np.ndarray, sample_size):
    b = get_pseudohap_sampling_bias(af, sample_size)
    tot_var = np.nanvar(af[-1] - af[0]) - b[-1] - b[0]
    return tot_var


def get_matrix_sum(covmat: np.ndarray, include_diag=False, abs=False):
    if abs:
        covmat = np.abs(covmat)
    if include_diag:
        S = np.nansum(covmat)
    else:
        S = np.nansum(covmat - np.diag(np.diag(covmat)))
    return S


def get_summary(covmat, af, sample_size, include_diag=False, abs=False):
    num = get_matrix_sum(covmat, include_diag, abs)
    denom = get_tot_var(af, sample_size)
    G = num / denom
    return G


def stats_from_matrices(covmat_nc, admix_cov, drift_err):
    k = covmat_nc.shape[0]
    G = []
    G_nc = []
    Ap = []
    totvar = []
    for i in range(1, k + 1):
        totvar.append(np.sum(covmat_nc[:i, :i]))
        G.append(
            get_matrix_sum(
                (covmat_nc - admix_cov - drift_err)[:i, :i],
                include_diag=False, abs=False
            ) / totvar[-1]
        )
        G_nc.append(
            get_matrix_sum(
                covmat_nc[:i, :i],
                include_diag=False, abs=False
            ) / totvar[-1]
        )
        Ap.append(
            get_matrix_sum(
                admix_cov[:i, :i],
                include_diag=True, abs=False
            ) / totvar[-1]
        )
    return (totvar, G_nc, G, Ap)


def get_ci(stat: np.ndarray, alpha=0.05, axis=0):
    qlower, qupper = (
        np.quantile(stat, alpha/2, axis=axis),
        np.quantile(stat, 1-alpha/2, axis=axis)
    )
    estimate = np.mean(stat, axis=axis)
    return (qlower, estimate, qupper)

# ================================================================
# Bootstrapping


def bootstrap_ci(estimate, straps, alpha=0.05, axis=0):
    # pivot method from Vince's cvtkpy
    qlower, qupper = (
        np.nanquantile(straps, alpha/2, axis=axis),
        np.nanquantile(straps, 1-alpha/2, axis=axis)
    )
    CIs = 2*estimate - qupper, estimate, 2*estimate - qlower
    return CIs


def weighted_mean(array, weights, axis=0):
    masked = np.ma.masked_invalid(array)
    mean = np.ma.average(masked, axis=axis, weights=weights)
    return mean.data if isinstance(mean, np.ma.MaskedArray) else mean


def bootstrap_stat(tiled_stat, weights, N_bootstrap, alpha=0.05, statistic=None):
    That = statistic
    rng = np.random.default_rng()
    L = len(tiled_stat)
    straps = []
    for _ in np.arange(N_bootstrap):
        bidx = rng.integers(0, L, size=L)
        straps.append(
            weighted_mean(
                tiled_stat[bidx],
                weights[bidx],
                axis=0,
            )
        )
    straps = np.array(straps)
    if That is None:
        That = np.mean(straps, axis=0)
    return bootstrap_ci(That, straps, alpha=alpha, axis=0)


def bootstrap_ratio(tiled_num, tiled_denom, weights, N_bootstrap, alpha=0.05, statistic=None):
    That = statistic
    assert tiled_num.shape[0] == tiled_denom.shape[0]
    rng = np.random.default_rng()
    L = len(tiled_num)
    straps = []
    for _ in np.arange(N_bootstrap):
        bidx = rng.integers(0, L, size=L)
        num = weighted_mean(
            tiled_num[bidx],
            weights[bidx],
            axis=0,
        )
        denom = weighted_mean(
            tiled_denom[bidx],
            weights[bidx],
            axis=0,
        )
        straps.append(num / denom)
    straps = np.array(straps)
    if That is None:
        That = np.mean(straps, axis=0)
    return bootstrap_ci(That, straps, alpha=alpha, axis=0)


def get_boot_average(bidx, tiled_stat, weights):
    masked = np.ma.masked_invalid(np.stack(tiled_stat)[bidx])
    strap = np.ma.average(masked, axis=0, weights=weights[bidx]).data
    return strap


def get_boot_average_ratio(bidx, tiled_stat_num, tiled_raw_tot_var, ragged_af, ragged_sample_size, weights):
    b = get_pseudohap_sampling_bias(
        np.concatenate(ragged_af[bidx], axis=1),
        np.concatenate(ragged_sample_size[bidx], axis=1),
    )
    strap_num = np.ma.average(
        np.ma.masked_invalid(np.stack(tiled_stat_num)[bidx]),
        axis=0, weights=weights[bidx]
    ).data
    strap_denom = np.ma.average(
        np.ma.masked_invalid(np.stack(tiled_raw_tot_var)[bidx]), 
        axis=0, weights=weights[bidx]
    ).data
    return strap_num / (strap_denom - b[-1] - b[0])


def bootstrap(
    tile_idxs,
    af,
    n_sample,
    Q,
    ref_af,
    n_sample_ref,
    alphas,
    N_bootstrap=5e3,
    full_output=False,
    bias=True,
    drift_err=True,
    abs_G=False,
    abs_Ap=False,
):
    n_loci = np.array([tile.size for tile in tile_idxs])

    tiled_af = [af[:, idx] for idx in tile_idxs]
    tiled_sample_size = [n_sample[:, idx] for idx in tile_idxs]

    assert af.shape == n_sample.shape
    tiled_cov = [
        get_covariance_matrix(a, bias=bias, sample_size=n)
        for a, n in zip(tiled_af, tiled_sample_size)
    ]

    assert ref_af.shape == n_sample_ref.shape
    tiled_admix_cov = [
        get_admix_covariance_matrix(
            Q, ref_af[:, idx], bias=bias,
            ref_sample_size=n_sample_ref[:, idx],
        )
        for idx in tile_idxs
    ]

    if drift_err:
        tiled_drift_err = [
            get_drift_err_matrix(
                solve_for_variances(np.diag(c - a), alphas),
                alphas,
            )
            for c, a in zip(tiled_cov, tiled_admix_cov)
        ]
    else:
        tiled_drift_err = [np.zeros(tiled_cov[0].shape)] * len(tiled_cov)

    tiled_corr_cov = [
        c - a - d for c, a, d in zip(tiled_cov, tiled_admix_cov, tiled_drift_err)
    ]

    tiled_num_G = [
        get_matrix_sum(cc, include_diag=False, abs=abs_G) 
        for cc in tiled_corr_cov
    ]
    if full_output:
        tiled_num_G_nc = [
            get_matrix_sum(c, include_diag=False, abs=abs_G) 
            for c in tiled_cov
        ]
    tiled_num_Ap = [
        get_matrix_sum(a, include_diag=True, abs=abs_Ap)
        for a in tiled_admix_cov
    ]
    tiled_raw_tot_var = [
        np.nanvar(a[-1] - a[0])
        for a in tiled_af
    ]

    weights = n_loci / np.sum(n_loci)

    rng = np.random.default_rng()
    L = len(tiled_af)
    ragged_af = np.empty((L,), dtype=object)
    ragged_af[:] = tiled_af
    ragged_sample_size = np.empty((L,), dtype=object)
    ragged_sample_size[:] = tiled_sample_size
    straps_cov = []
    straps_corr_cov = []
    straps_G = []
    straps_G_nc = []
    straps_Ap = []
    for _ in np.arange(N_bootstrap):
        straps_corr_cov.append(
            get_boot_average(
                rng.integers(0, L, size=L),
                tiled_corr_cov,
                weights,
            )
        )
        straps_G.append(
            get_boot_average_ratio(
                rng.integers(0, L, size=L),
                tiled_num_G, tiled_raw_tot_var,
                ragged_af, ragged_sample_size, weights)
        )
        straps_Ap.append(
            get_boot_average_ratio(
                rng.integers(0, L, size=L),
                tiled_num_Ap, tiled_raw_tot_var,
                ragged_af, ragged_sample_size, weights)
        )
        if full_output:
            straps_cov.append(
                get_boot_average(
                    rng.integers(0, L, size=L),
                    tiled_cov,
                    weights,
                )
            )
            straps_G_nc.append(
                get_boot_average_ratio(
                    rng.integers(0, L, size=L),
                    tiled_num_G_nc, tiled_raw_tot_var,
                    ragged_af, ragged_sample_size, weights)
            )

    straps = {
        'corr_cov': np.stack(straps_corr_cov),
        'G': np.stack(straps_G),
        'Ap': np.stack(straps_Ap),
    }
    if full_output:
        straps['cov'] = np.stack(straps_cov)
        straps['G_nc'] = np.stack(straps_G_nc)

    return straps
