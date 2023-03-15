import tskit
import numpy as np
from numpy.typing import NDArray
from ..covariances import *


def get_times(ts: tskit.TreeSequence):
    has_indiv = ts.tables.nodes.individual >= 0
    which_indiv = ts.tables.nodes.individual[has_indiv]
    individual_times = np.zeros(ts.num_individuals)
    individual_times[which_indiv] = ts.tables.nodes.time[has_indiv]
    return np.unique(individual_times)


def individuals_at(ts, time, population):
    has_indiv = ts.tables.nodes.individual >= 0
    which_indiv = ts.tables.nodes.individual[has_indiv]
    individual_times = np.zeros(ts.num_individuals)
    individual_times[which_indiv] = ts.tables.nodes.time[has_indiv]
    individual_populations = np.repeat(np.int32(-1), ts.num_individuals)
    individual_populations[which_indiv] = ts.tables.nodes.population[has_indiv]
    res_inds = np.where(
        (individual_times == time) & (individual_populations == population)
    )[0]
    return res_inds


def nodes_at(ts, time, population):
    is_sample = ts.tables.nodes.flags == 1
    at_time = ts.tables.nodes.time == time
    in_pop = ts.tables.nodes.population == population
    res_nodes = np.where(is_sample & at_time & in_pop)[0]
    return res_nodes


def draw_sample_set_at(
    ts: tskit.TreeSequence,
    time: int,
    rng,
    pop: int,
    n_sample: int,
):
    nodes = ts.tables.nodes
    all_samples = (nodes.population == pop) & (nodes.time == time) & (nodes.flags == 1)
    ind_id = np.unique(nodes[all_samples].individual)
    sample_set = np.where(
        all_samples
        & np.isin(nodes.individual, rng.choice(ind_id, size=n_sample, replace=False))
    )[0]
    # take only n_sample inds
    # take both nodes of each ind
    return sample_set


def draw_sample_sets(
    ts: tskit.TreeSequence,
    times: list[int],
    rng,
    pop: int,
    n_sample: int|list[int],
):
    if isinstance(n_sample, int):
        sample_sets = [draw_sample_set_at(ts, t, rng, pop, n_sample) for t in times]
    else:
        assert len(times) == len(n_sample)
        sample_sets = [
            draw_sample_set_at(ts, t, rng, pop, n)
            for t, n in zip(times, n_sample)
        ]
    return sample_sets


def get_allele_frequencies(
    ts: tskit.TreeSequence,
    sample_sets=None,
    flip=None,
):
    if sample_sets is None:
        sample_sets = [ts.samples()]
    n = np.array([len(x) for x in sample_sets])
    if flip is None:
        flip = np.zeros(ts.num_sites, dtype=bool)

    def f(x):
        return x / n

    res = ts.sample_count_stat(
        sample_sets,
        f,
        len(sample_sets),
        span_normalise=False,
        windows="sites",
        polarised=True,
        mode="site",
        strict=False,
    ).T

    res[:, flip] = 1 - res[:, flip]

    return res


def get_genotype_matrix_pseudohap(
    ts: tskit.TreeSequence,
    rng,
    N_samples: int,
    samples: NDArray,
    flip: None | NDArray[np.bool_] = None,  # wether to flip the allele
):
    # this function assumes sample nodes of an individual are adjacent.
    if flip is None:
        flip = np.zeros(ts.num_sites, dtype=bool)
    base_indices = np.array(range(0, N_samples * 2, 2))
    ret = np.zeros((ts.num_sites, N_samples), dtype=np.int8)
    for i, v in enumerate(ts.variants(samples=samples)):
        indices = rng.integers(0, 2, size=N_samples) + base_indices
        ret[i] = v.genotypes[indices]
        if flip[i]:
            ret[i] = 1 - ret[i]
    return ret


def get_pseudohap_genotypes(
    ts: tskit.TreeSequence,
    rng,
    samples: None | NDArray | list[NDArray] | tuple[list[NDArray]] = None,
    flip: None | NDArray[np.bool_] = None,  # wether to flip the allele
):
    # Deal with different ways of passing the samples
    if samples is None:
        N_samples = int(ts.num_samples / 2)
        passed_samples = ts.samples()
        gen = get_genotype_matrix_pseudohap(ts, rng, N_samples, passed_samples, flip)
        return gen
    elif isinstance(samples, np.ndarray):
        N_samples = int(samples.size / 2)
        gen = get_genotype_matrix_pseudohap(ts, rng, N_samples, samples, flip)
        return gen
    elif isinstance(samples, list):
        sizes = [int(s.size / 2) for s in samples]
        splits = np.cumsum(sizes)[:-1]
        passed_samples = np.concatenate(samples)
        N_samples = int(passed_samples.size / 2)
        gen = np.split(
            get_genotype_matrix_pseudohap(ts, rng, N_samples, passed_samples, flip),
            splits,
            axis=1,
        )
        return gen
    elif isinstance(samples, tuple):
        sizes = []
        for group in samples:
            sizes += [int(s.size / 2) for s in group]
        splits = np.cumsum(sizes)[:-1]
        passed_samples = np.concatenate([y for x in samples for y in x])  # unpack
        N_samples = int(passed_samples.size / 2)
        gen = np.split(
            get_genotype_matrix_pseudohap(ts, rng, N_samples, passed_samples, flip),
            splits,
            axis=1,
        )
        # repack
        tuple_gen = tuple()
        counter = 0
        for i in range(len(samples)):
            tuple_gen += (gen[counter : (counter + len(samples[i]))],)
            counter += len(samples[i])
        return tuple_gen


def get_admixture_proportions(
    ts: tskit.TreeSequence,
    admix_inds: list[int],
    ancestral_pops_nodes: list[np.ndarray],
):
    # need to check admix_inds is list of individual indices
    N_anc_pop = len(ancestral_pops_nodes)
    admix_proportions = np.zeros((len(admix_inds), N_anc_pop))
    admix_nodes = np.array([ts.individual(ind).nodes for ind in admix_inds]).flatten()
    edges = ts.tables.link_ancestors(admix_nodes, np.concatenate(ancestral_pops_nodes))
    for ix, ind in enumerate(admix_inds):
        child_nodes = ts.individual(ind).nodes
        anc_edges = [
            edges[np.isin(edges.child, child_nodes) & np.isin(edges.parent, pop_nodes)]
            for pop_nodes in ancestral_pops_nodes
        ]
        spans = [np.sum(pop_edges.right - pop_edges.left) for pop_edges in anc_edges]
        for j in range(N_anc_pop):
            admix_proportions[ix, j] = spans[j] / np.sum(spans)
    return admix_proportions


def create_tile_idxs(ts, tile_size):
    tiles = [(i, i + tile_size) for i in range(0, int(ts.sequence_length), tile_size)]
    sites = ts.tables.sites
    tile_idxs = [
        np.where(
            (start <= sites.position) & (sites.position < stop)
        )[0]
        for start, stop in tiles
    ]
    # no need to check empty tiles in simulations
    return tile_idxs


def analyze_trees(
    ts,
    times: np.ndarray | list[int],
    n_samples: list[int],
    admix_pop: int,
    refs: list[dict],
    alpha_mask: np.ndarray,
    rng=None,
):
    assert(list(refs[0].keys()) == ['pop', 'time', 'n'])
    if rng is None:
        rng = np.random.default_rng()
    
    samples_nodes = draw_sample_sets(ts, times, rng, admix_pop, n_samples)
    ref_nodes = [
        draw_sample_set_at(ts, r['time'], rng, r['pop'], r['n'])
        for r in refs
    ]

    ref_geno_sets, geno_sets = get_pseudohap_genotypes(
        ts, rng,
        samples=(ref_nodes, samples_nodes),
    )

    af = np.stack([
        np.mean(G, axis=1)
        for G in geno_sets
    ], axis=0)
    ref_af = np.stack([
        np.mean(G, axis=1)
        for G in ref_geno_sets
    ], axis=0)

    # admixture computations
    samples_inds = [np.unique([ts.node(u).individual for u in s]) for s in samples_nodes]
    ancestral_census_nodes = [
        np.where(
            (ts.tables.nodes.population == r['pop'])
            & (ts.tables.nodes.time == r['time'])
        )[0]
        for r in refs
    ]
    Q = np.stack([
        np.mean(
            get_admixture_proportions(ts, g, ancestral_census_nodes),
            axis=0,
        )
        for g in samples_inds
    ])

    alphas = q2a_simple(Q, alpha_mask)
    alphas[alphas < 0] = 0 # issues due to drift
    covmat = get_covariance_matrix(
        af, bias=True,
        sample_size=np.array(n_samples),
    )
    admix_cov = get_admix_covariance_matrix(
        Q, ref_af, bias=True,
        ref_sample_size=np.array([d['n'] for d in refs]),
    )
    var_drift = solve_for_variances(
        np.diag(covmat - admix_cov), alphas,
    )
    drift_err = get_drift_err_matrix(
        var_drift, alphas,
    )

    # pca correction
    gt = np.concatenate([
        np.concatenate(geno_sets, axis=1),
        np.concatenate(ref_geno_sets, axis=1),
    ], axis=1).T # (samples, variants)
    N = gt.shape[0]
    mean_gt = np.mean(gt, axis=0)
    p = ((1 + np.sum(gt, axis=0)) / (2 + 2 * N))

    X = ((gt - mean_gt) / ((p * (1 - p))**.5))
    Psi = np.cov(X)
    eval, evec = np.linalg.eigh(Psi)
    i = np.argsort(eval)
    i = i[::-1]
    evec = evec[:,i]
    eval = eval[i]

    N_vec = 2
    count_samples = np.sum(n_samples)
    gamma = np.zeros((gt.shape[1], N_vec))
    for i in range(N_vec):
        gamma[:,i] = np.nansum(evec[:count_samples, i] * gt[:count_samples].T, axis=1) / np.sum(evec[:count_samples, i]**2)

    gt_exp = np.dot(evec[:count_samples, :N_vec], gamma.T) # gamma_1*A1 + gamma_2*A2
    gt_adj = gt[:count_samples] - gt_exp
    af_adj = np.stack(
        [
            np.mean(gt_adj[int(np.sum(n_samples[:i])):int(np.sum(n_samples[:(i+1)]))], axis=0)
            for i, n in enumerate(n_samples)
        ]
    )

    covmat_pca = get_covariance_matrix(
        af_adj, bias=True,
        sample_size=np.array(n_samples),
    )

    return {
        'covmat': covmat,
        'admix_cov': admix_cov,
        'drift_err': drift_err,
        'Q': Q,
        'covmat_pca': covmat_pca,
        'evec': evec,
    }