from typing import Tuple


class GlobalSearchSettings:
    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: Tuple[float, float] = (1.5, 1.9),
        n_jobs: int = -1
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.n_jobs = n_jobs
