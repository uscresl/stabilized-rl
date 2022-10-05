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
        self._phase = np.random.uniform(2 * np.pi)

    def _featurize(self, hparams):
        return np.concatenate([hparams, hparams**2, hparams**3], axis=0)

    def true_mean(self, hparams):
        featurized = self._featurize(hparams)
        return (np.dot(self._hparam_dep, featurized) +
                np.sin(self._phase + 5 * featurized[0]))

    def step(self, hparams):
        perf_improvement = self.true_mean(hparams) + np.random.normal(
            self._improvement_noise
        )
        self.perf_now += perf_improvement
        return self.perf_now
