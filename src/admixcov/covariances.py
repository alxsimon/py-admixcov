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
    sample_size: int|np.ndarray):
    hh = af * (1 - af)
    if isinstance(sample_size, np.ndarray):
        # meaning there can be NaNs
        # need to check this part
        tmp = (sample_size - 1).astype(float)
        tmp[tmp <= 0] = np.nan
        sample_correction = 1 / tmp
    else: # sample_size is single int
        sample_correction = 1 / (sample_size - 1)
    # array shape adjustment
    if isinstance(sample_correction, np.ndarray):
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
    has_NA = np.isnan(af).sum() > 0
    if has_NA:
        covmat = np.ma.cov(np.ma.masked_invalid(np.diff(af, axis=0))).data
    else:
        covmat = np.cov(np.diff(af, axis=0))
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
    # determine if ref_af has NAs
    has_NA = np.isnan(ref_af).sum() > 0
    admix_af = Q @ ref_af
    if has_NA:
        admix_cov = np.ma.cov(np.ma.masked_invalid(np.diff(admix_af, axis=0))).data
    else:
        admix_cov = np.cov(np.diff(admix_af, axis=0))
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


def get_matrix_sum(covmat: np.ndarray, k=0):
    S = np.sum(covmat - np.triu(covmat, k=k)) # removes upper triangle
    return S


def get_G(covmat, af, sample_size, include_diag=False):
    if include_diag:
        num = get_matrix_sum(covmat, k=1)
    else:
        num = get_matrix_sum(covmat, k=0)
    denom = get_tot_var(af, sample_size)
    G = num / denom
    return G


# ================================================================
# Bootstrapping


def run_bootstrap(tiled_stat, N_bootstrap, weights):
    straps = []
    for b in np.arange(N_bootstrap):
        bidx = np.random.randint(0, len(tiled_stat), size=len(tiled_stat))
        straps.append(
            np.average(np.stack(tiled_stat)[bidx], axis=0, weights=weights[bidx])
        )
    return np.stack(straps)


def run_bootstrap_ratio(tiled_stat_num, tiled_stat_denom, N_bootstrap, weights):
    straps_num = []
    straps_denom = []
    for b in np.arange(N_bootstrap):
        bidx = np.random.randint(0, len(tiled_stat_num), size=len(tiled_stat_num))
        straps_num.append(
            np.average(np.stack(tiled_stat_num)[bidx], axis=0, weights=weights[bidx])
        )
        straps_denom.append(
            np.average(np.stack(tiled_stat_denom)[bidx], axis=0, weights=weights[bidx])
        )
    return np.stack(straps_num) / np.stack(straps_denom)


def bootstrap(
    ts,
    af,
    n_sample,
    Q,
    ref_af,
    n_sample_ref,
    alphas,
    N_bootstrap=5e3,
    tile_size: int = int(1e6),
    bias=True,
    drift_err=True,
):

    tiles = [(i, i + tile_size) for i in range(0, int(ts.sequence_length), tile_size)]
    sites = ts.tables.sites
    tile_masks = [
        (start <= sites.position) & (sites.position < stop) for start, stop in tiles
    ]
    n_loci = np.array([np.sum(tile) for tile in tile_masks])

    tiled_af = [af[:, mask] for mask in tile_masks]

    tiled_cov = [
        get_covariance_matrix(af[:, mask], bias=bias, sample_size=n_sample)
        for mask in tile_masks
    ]

    # tiled_bias = [compute_bias_covmat(a, n_sample) for a in tiled_af]

    tiled_admix_cov = [
        get_admix_covariance_matrix(
            Q, ref_af[:, mask], bias=bias, ref_sample_size=n_sample_ref
        )
        for mask in tile_masks
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
        get_matrix_sum(c, k=0)
        for c in tiled_corr_cov
    ]
    tiled_num_Ap = [
        get_matrix_sum(c, k=1)
        for c in tiled_corr_cov
    ]
    tiled_tot_var = [
        get_tot_var(a, n_sample)
        for a in tiled_af
    ]

    weights = n_loci / np.sum(n_loci)
    straps_cov = run_bootstrap(tiled_cov, N_bootstrap, weights)
    # straps_bias = run_bootstrap(
    # 	[c - b for c, b in zip(tiled_cov, tiled_bias)],
    # 	N_bootstrap, weights
    # )
    straps_admix_cov = run_bootstrap(tiled_admix_cov, N_bootstrap, weights)
    straps_corr_cov = run_bootstrap(tiled_corr_cov, N_bootstrap, weights)
    straps_G = run_bootstrap_ratio(tiled_num_G, tiled_tot_var, N_bootstrap, weights)
    straps_Ap = run_bootstrap_ratio(tiled_num_Ap, tiled_tot_var, N_bootstrap, weights)

    return (straps_cov, straps_admix_cov, straps_corr_cov, straps_G, straps_Ap)
