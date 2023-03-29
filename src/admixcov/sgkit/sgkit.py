import sgkit
import numpy as np
from ..covariances import *

def create_tile_idxs(ds, type: str, size: int=1000):
    if type == 'variant':
        wds = sgkit.window_by_variant(ds, size=size)
    elif type == 'position':
        wds = sgkit.window_by_position(ds, size=size)
    variant_index = [i for i, _ in enumerate(wds.variant_id.values)]
    tile_masks = [
        np.where(
            (start <= variant_index) & (variant_index < stop)
        )[0]
        for start, stop in zip(wds.window_start.values, wds.window_stop.values)
        if start != stop # filter out empty tiles
    ]
    return tile_masks


def ds2stats(ds, alpha_mask, tile_size_variant):
    times = [np.mean(ds.sample_date_bp.values[mask]) for mask in ds.mask_cohorts.values]
    geno = ds.call_genotype.values[:,:,0].T.astype(float)
    geno[geno == -1] = np.nan
    ref_af = np.stack([np.nanmean(geno[mask], axis=0) for mask in ds.mask_cohorts_ref.values])
    sample_size_ref = [np.sum(mask) for mask in ds.mask_cohorts_ref.values]
    def allele_freq(geno, mask):
        return np.nanmean(geno[mask], axis=0)
    
    af = np.stack([
        allele_freq(geno, mask)
        for mask in ds.mask_cohorts.values
    ])
    sample_size = np.array([
        np.sum(mask)
        for mask in ds.mask_cohorts.values
    ])
    Q = admix = np.stack([
        np.mean(ds.sample_admixture[mask], axis=0)
        for mask in ds.mask_cohorts.values
    ])
    covmat = get_covariance_matrix(
        af,
        bias=True,
        sample_size=ds.variant_count_nonmiss.values,
    )
    admix_cov = get_admix_covariance_matrix(
        Q,
        ref_af=ref_af,
        bias=True,
        ref_sample_size=ds.variant_count_nonmiss_ref.values,
    )
    alphas = q2a_simple(Q, alpha_mask)
    var_drift = solve_for_variances(
        np.diag(covmat - admix_cov),
        alphas,
    )
    drift_err = get_drift_err_matrix(var_drift, alphas)
    
    totvar = np.sum(covmat)
    G = get_matrix_sum(
            covmat - admix_cov - drift_err,
            include_diag=False, abs=False
        ) / totvar
    Ap = get_matrix_sum(
            admix_cov,
            include_diag=True, abs=False
        ) / totvar
    
    N_boot = 1e4
    tile_idxs = create_tile_idxs(ds, type='variant', size=tile_size_variant)
    sizes = [x.size for x in tile_idxs] # Number of SNPs in tiles

    n_sample = ds.variant_count_nonmiss.values
    n_sample_ref = ds.variant_count_nonmiss_ref.values
    tiled_af = [af[:, idx] for idx in tile_idxs]
    tiled_sample_size = [n_sample[:, idx] for idx in tile_idxs]

    assert af.shape == n_sample.shape
    tiled_cov = np.stack([
        get_covariance_matrix(a, bias=True, sample_size=n)
        for a, n in zip(tiled_af, tiled_sample_size)
    ])

    assert ref_af.shape == n_sample_ref.shape
    tiled_admix_cov = np.stack([
        get_admix_covariance_matrix(
            Q, ref_af[:, idx], bias=True,
            ref_sample_size=n_sample_ref[:, idx],
        )
        for idx in tile_idxs
    ])

    tiled_drift_err = [
        get_drift_err_matrix(
            solve_for_variances(np.diag(c - a), alphas),
            alphas,
        )
        for c, a in zip(tiled_cov, tiled_admix_cov)
    ]

    tiled_corr_cov = np.stack([
        c - a - d for c, a, d in zip(tiled_cov, tiled_admix_cov, tiled_drift_err)
    ])

    n_loci = np.array([tile.size for tile in tile_idxs])
    weights = n_loci / np.sum(n_loci)

    # do the bootstraps
    straps_cov = bootstrap_stat(tiled_corr_cov, weights, N_boot)

    tmp_totvar = np.sum(tiled_cov, axis=(1, 2))
    straps_totvar = bootstrap_stat(
            tmp_totvar,
            weights,
            N_boot,
        )
    straps_G = bootstrap_ratio(
            np.stack([get_matrix_sum(c) for c in tiled_corr_cov]),
            tmp_totvar,
            weights,
            N_boot,
            statistic=G,
        )
    straps_Ap = bootstrap_ratio(
            np.stack([get_matrix_sum(c, include_diag=True) for c in tiled_admix_cov]),
            tmp_totvar,
            weights,
            N_boot,
            statistic=Ap,
        )
    
    return (
        covmat,
        G,
        Ap,
        totvar,
        straps_cov,
        straps_G,
        straps_Ap,
        straps_totvar,
    )