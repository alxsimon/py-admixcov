import tskit
import numpy as np


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
    n_sample,
):
    nodes = ts.tables.nodes
    all_samples = (nodes.population == pop) & (nodes.time == time) & (nodes.flags == 1)
    ind_id = np.unique(nodes[all_samples].individual)
    # take only n_sample inds
    # take both nodes of each ind
    sample_set = np.where(
        all_samples
        & np.isin(nodes.individual, rng.choice(ind_id, size=n_sample, replace=False))
    )[0]
    return sample_set


def draw_sample_sets(
    ts: tskit.TreeSequence,
    times: list[int],
    rng,
    pop: int,
    n_sample,
):
    sample_sets = [draw_sample_set_at(ts, t, rng, pop, n_sample) for t in times]
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
    samples=None,
    flip=None,
):
    # this function assumes sample nodes of a same individual are adjacent.
    if samples is None:
        N_samples = int(ts.num_samples / 2)
        samples = ts.samples()
    else:
        N_samples = int(len(samples) / 2)
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
