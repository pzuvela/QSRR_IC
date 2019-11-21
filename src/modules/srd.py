"""
Sum of ranking differences analysis
Validated by comparison of ranks with random numbers (CRNN)

References:
    1) Heberger, K. Sum of ranking differences compares methods or models fairly. TRAC 2010, 29, 101-109.
        (doi:10.1016/j.trac.2009.09.009)
    2) Heberger, K.; Kollar-Hunek, K. Sum of ranking differences for method discrimination and its validation:
        comparison of ranks with random numbers. J. Chemom. 2011, 25, 151-158.
        (doi:10.1002/cem.1320)
    3) Kalivas, J. H.; Heberger, K.; Andries, E. Sum of ranking differences (SRD) to ensemble multivariate
        calibration model merits for tuning parameter selection and comparing calibration methods, Anal. Chim. Acta
        2015, 869, 21-33.
        (doi:10.1016/j.aca.2014.12.056)
"""

# TODO: Parallelize the code
# TODO: Comment the code
# TODO: Add histogram & bar plotting


# Import numpy packages
from numpy import mean, size, ndarray, argsort, argwhere, arange, multiply, zeros
from numpy.random import permutation 


class SumOfRankingDiffs:

    def __init__(self, a, t=None):

        # Define input matrix & target
        self.A = a  # Matrix A [columns: models, methods; rows: samples]
        self.T = mean(a, axis=0) if t is None else t  # Target T [default: average]

        # Define size of A
        self.nrows, self.ncols = size(self.A, axis=0), size(self.A, axis=1)

        self.srd, self.srd_norm, self.srd_max, self.srd_rnd, self.srd_rnd_norm = [ndarray([])] * 5

    def compute_srd(self):

        # Define T & A indices, and initialize the rank of A
        t_index, a_index, a_ranking = argsort(self.T), argsort(self.A, axis=0), zeros((self.nrows, self.ncols))

        for i in range(self.nrows):
            row = argwhere(a_index == t_index[i])
            a_ranking[i, :] = row[:, 0].T

        ideal_rank = arange(self.nrows).reshape(-1, 1)
        self.srd = sum(abs(a_ranking - ideal_rank))

        return self

    def _srd_max(self):

        if self.nrows % 2 == 1:

            k = (self.nrows - 1) / 2
            self.srd_max = 2 * k * (k + 1)

        else:

            k = self.nrows / 2
            self.srd_max = 2 * (k ** 2)

        return self

    def srd_normalize(self):

        # Assertion to make sure that SRD is ran before normalization !
        assert self.srd.size > 0, '# You must run the SRD method before normalization !'

        # Normalization
        self.srd_norm = self.srd * (100 / self._srd_max().srd_max)

        return self

    @staticmethod
    def _srd_normalize_val(srd_vals, srd_max):
        return multiply(srd_vals, 100 / srd_max)

    # SRD validation using the distribution of SRD values of normally-distributed random numbers
    def srd_validate(self, n_rnd_vals=10000):

        # Assertion to make sure that SRD is ran before validation !
        assert self.srd.size > 0, '# You must run the SRD method before validation !'

        # Ideal ranking
        ideal_rank = arange(self.nrows).reshape(-1, 1)

        # Predefine "srd_rnd" and "srd_norm"
        srd_rnd, srd_rnd_norm = [zeros((n_rnd_vals, 1))] * 2

        # Compute SRD values for "n_rand_vals" random numbers
        for i in arange(n_rnd_vals):

            rnd_order = permutation(self.nrows).reshape(-1, 1)
            srd_rnd[i] = sum(abs(rnd_order - ideal_rank))
            srd_rnd_norm[i] = self._srd_normalize_val(srd_vals=srd_rnd[i], srd_max=self._srd_max().srd_max)

        self.srd_rnd, self.srd_rnd_norm = srd_rnd, srd_rnd_norm

        return self
