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


def get_pseudohap_sampling_bias(af, sample_size):
    hh = af * (1 - af)
    if sample_size.ndim > 1:  # meaning there can be NaNs
        # need to check this part
        tmp = sample_size - 1
        tmp[tmp <= 0] = -99
        sample_correction = 1 / tmp
        sample_correction[sample_correction < 0] = np.nan
    else:
        sample_correction = 1 / (sample_size - 1)
    # array shape adjustment
    if sample_correction is not int:
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
