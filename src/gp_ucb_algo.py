from collections import deque
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import numpy as np
from matplotlib import pyplot as plt

from dowel import logger, tabular
from garage.experiment.deterministic import get_seed
from garage.np.algos import RLAlgorithm

DEFAULT_GP_KERNEL = (Matern(length_scale=0.1, nu=2.5,
                            length_scale_bounds="fixed") +
                     (WhiteKernel(noise_level_bounds=(0.1, 1e5)) *
                      Matern(length_scale=0.1, nu=2.5,
                             length_scale_bounds="fixed")))


class GPUCBAlgo(RLAlgorithm):

    def __init__(self, inner_algo, perf_statistic='AverageReturn',
                 kernel=DEFAULT_GP_KERNEL, epoch_window_size=None,
                 min_epoch_window_size=None):
        self.inner_algo = inner_algo
        self.perf_changes = deque(maxlen=epoch_window_size)
        self.hparam_vecs = deque(maxlen=epoch_window_size)
        self.min_epoch_window_size = min_epoch_window_size
        self.regressor = GaussianProcessRegressor(kernel=kernel,
                                                  random_state=get_seed())
        self.prev_algo_perf = None
        self.perf_statistic = perf_statistic
        self.hparam_ranges = self.inner_algo.hparam_ranges()
        self.n_hparams = len(self.hparam_ranges)
        self.n_hparam_ucb_samples = 1000
        self.ucb_nu = 8
        self.ucb_delta = 0.5

    def _hparams_to_vec(self, hparams):
        vec = np.zeros((self.n_hparams,))
        for i, k in enumerate(self.hparam_ranges.keys()):
            vec[i] = hparams[k]
        return vec

    def _vec_to_hparams(self, vec):
        hparams = {}
        for i, k in enumerate(self.hparam_ranges.keys()):
            hparams[k] = vec[i]
        return hparams

    def step(self, trainer, epoch):
        if len(self.perf_changes) == 0:
            new_stats = self.inner_algo.step(trainer, epoch)
            hparams = self.inner_algo.get_hparams()
            vec = self._hparams_to_vec(hparams)
            tabular.record('ExpectedPerfChange', 0)
        else:
            self.regressor.fit(self.hparam_vecs, self.perf_changes)
            vec = self._select_ucb_hparams(epoch)
            hparams = self._vec_to_hparams(vec)
            self.inner_algo.set_hparams(hparams)
            new_stats = self.inner_algo.step(trainer, epoch)
        for key, value in hparams.items():
            with tabular.prefix('HParams/'):
                tabular.record(key, value)
        perf = new_stats[self.perf_statistic]
        if self.prev_algo_perf is not None:
            self.hparam_vecs.append(vec)
            self.perf_changes.append(perf - self.prev_algo_perf)
            for index, (key, value) in enumerate(self.hparam_ranges.items()):
                self.plot_gpr(f'{trainer.log_directory}/{key}{epoch:05}.png',
                              key, index)
        self.prev_algo_perf = perf
        return new_stats

    def _select_ucb_hparams(self, epoch):
        sampled_hparams = np.random.sample(size=(self.n_hparam_ucb_samples,
                                                 self.n_hparams))
        pred_mu, pred_std = self.regressor.predict(sampled_hparams,
                                                   return_std=True)
        tau_t = 2 * np.log(
            (1 + epoch) ** (self.n_hparams / 2 + 2) *
            np.pi**2 / (3 * self.ucb_delta)
        )  # 2*log(t^(d/2+2)π^2/(3δ))
        k_t = np.sqrt(self.ucb_nu * tau_t)
        ucb = pred_mu + k_t * pred_std
        max_idx = np.argmax(ucb)
        tabular.record('ExpectedPerfChange', pred_mu[max_idx])
        return sampled_hparams[max_idx]

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            dict[str, float]: Statistics

        """
        last_statistics = {}

        for epoch in trainer.step_epochs():
            last_statistics = self.step(trainer, epoch)
        return last_statistics

    def plot_gpr(self, filename, hparam, hparam_idx):
        sampled_hparams = np.random.sample(size=(10000, self.n_hparams))
        X_bel = sampled_hparams[:, hparam_idx]
        mean_bel, std_bel = self.regressor.predict(sampled_hparams,
                                                   return_std=True)

        plt.clf()
        # plt.ylim((-5, 5))
        plt.xlim((self.hparam_ranges[hparam][0],
                  self.hparam_ranges[hparam][1]))
        plt.scatter([vec[hparam_idx] for vec in self.hparam_vecs],
                    self.perf_changes, label="Observations")
        plt.plot(X_bel, mean_bel, label="Mean prediction")
        plt.fill_between(
            X_bel.ravel(),
            mean_bel - 1.96 * std_bel,
            mean_bel + 1.96 * std_bel,
            alpha=0.5,
            label=r"95% confidence interval",
        )
        plt.legend()
        plt.xlabel(f"{hparam}")
        plt.ylabel("Perf. Improvement")
        plt.savefig(filename)
