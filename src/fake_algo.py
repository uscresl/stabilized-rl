from dataclasses import dataclass
import numpy as np


@dataclass
class FakeAlgo:

    perf_now: float = 0
    n_hparams: int = 2
    _hparam_dep: np.ndarray or None = None
    _improvement_noise: float = 1.0

    def __post_init__(self):
        self._hparam_dep = np.random.normal(size=(self.n_hparams * 3)) + 0.1

    def _featurize(self, hparams):
        return np.concatenate([hparams, hparams**2, hparams**3], axis=0)

    def step(self, hparams):
        featurized = self._featurize(hparams)
        perf_improvement = np.dot(self._hparam_dep, featurized) + np.random.normal(
            self._improvement_noise
        )
        self.perf_now += perf_improvement
        return self.perf_now
