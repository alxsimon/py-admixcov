import admixcov
import pytest
import numpy as np


class CovData:
    def __init__(self, seed=0):
        self.L = int(1e5)
        self.N_times = 5
        self.rng = np.random.default_rng(seed)
        self.sample_size = np.array([20, 10, 50, 100, 5])
        self.sample_size_md = np.array(
            [self.rng.integers(0, ss, size=self.L) for ss in self.sample_size]
        )
        self.af = self.rng.uniform(low=0, high=1.0, size=(self.N_times, self.L))
        self.af_md = self.af.copy()
        self.af_md[self.sample_size_md == 0] = np.nan
        self.ref_af = self.rng.uniform(low=0, high=1.0, size=(3, self.L))
        self.A = np.array(
            [
                [0.17, 0.05, 0.13],
                [0.10, 0.19, 0.09],
                [0.02, 0.06, 0.07],
                [0.03, 0.04, 0.02],
                [0.16, 0.09, 0.15],
            ]
        )
        self.Q = np.array(
            [
                [1.0, 0, 0],
                [
                    8.199999999999999512e-01,
                    5.000000000000000278e-02,
                    1.300000000000000044e-01,
                ],
                [
                    6.083999999999999408e-01,
                    2.210000000000000020e-01,
                    1.706000000000000016e-01,
                ],
                [
                    5.371399999999999508e-01,
                    2.478499999999999870e-01,
                    2.150100000000000067e-01,
                ],
                [
                    5.187973999999999641e-01,
                    2.655434999999999879e-01,
                    2.156590999999999925e-01,
                ],
                [
                    4.712784399999999652e-01,
                    2.493260999999999949e-01,
                    2.793954599999999844e-01,
                ],
            ]
        )
        # need to change bias computation
        self.bias_vector = np.mean(
            (self.af * (1 - self.af)) * (1 / (self.sample_size - 1))[:, np.newaxis],
            axis=1,
        )
        self.bias_vector_md = np.nanmean(
            (1 / (self.sample_size_md - 1)) * (self.af_md * (1 - self.af_md)),
            axis=1,
        )
        # ======
        self.cov = np.cov(np.diff(self.af, axis=0))
        self.cov_md = np.ma.cov(np.ma.masked_invalid(np.diff(self.af_md, axis=0))).data
        self.cov_bias = self.cov - admixcov.create_bias_correction_matrix(
            self.bias_vector
        )
        self.cov_bias_md = self.cov_md - admixcov.create_bias_correction_matrix(
            self.bias_vector_md
        )


@pytest.fixture(scope="module")
def cov_data():
    seed = 2273645
    yield CovData(seed=seed)


def test_bias(cov_data):
    assert (
        admixcov.get_bias_vector(cov_data.af, cov_data.sample_size)
        == cov_data.bias_vector
    ).all()
    assert (
        admixcov.get_bias_vector(cov_data.af_md, cov_data.sample_size_md)
        == cov_data.bias_vector_md
    ).all()
    assert (
        admixcov.get_bias_matrix(cov_data.af, cov_data.sample_size)
        == admixcov.create_bias_correction_matrix(cov_data.bias_vector)
    ).all()
    assert (
        admixcov.get_bias_matrix(cov_data.af_md, cov_data.sample_size_md)
        == admixcov.create_bias_correction_matrix(cov_data.bias_vector_md)
    ).all()


def test_covariance_matrix(cov_data):
    assert (
        admixcov.get_covariance_matrix(cov_data.af, bias=False) == cov_data.cov
    ).all()
    assert (
        admixcov.get_covariance_matrix(
            cov_data.af, bias=True, sample_size=cov_data.sample_size
        )
        == cov_data.cov_bias
    ).all()
    assert (
        admixcov.get_covariance_matrix(cov_data.af_md, bias=False) == cov_data.cov_md
    ).all()
    assert (
        admixcov.get_covariance_matrix(
            cov_data.af_md, bias=True, sample_size=cov_data.sample_size_md
        )
        == cov_data.cov_bias_md
    ).all()
