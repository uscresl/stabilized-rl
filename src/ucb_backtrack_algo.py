from collections import deque
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import numpy as np
from matplotlib import pyplot as plt
import cloudpickle

from dowel import logger, tabular
from garage.experiment.deterministic import get_seed
from garage.np.algos import RLAlgorithm

DEFAULT_GP_KERNEL = (Matern(length_scale=0.1, nu=1.5,
                            length_scale_bounds="fixed") +
                     ConstantKernel() +
                     (WhiteKernel() *
                      Matern(length_scale=0.1, nu=1.5,
                             length_scale_bounds="fixed")))
# DEFAULT_GP_KERNEL = (Matern(length_scale=0.05, nu=1.5,
                            # length_scale_bounds="fixed") +
                     # ConstantKernel())


class UCBBacktrackAlgo(RLAlgorithm):

    def __init__(self, inner_algo, perf_statistic='AverageReturn',
                 kernel=DEFAULT_GP_KERNEL, epoch_window_size=None,
                 min_epoch_window_size=8, offset_epoch=0, ucb_nu=1.0, ucb_delta=0.5):
        # UCB
        self.perf_changes = deque(maxlen=epoch_window_size)
        self.hparam_vecs = deque(maxlen=epoch_window_size)
        self.min_epoch_window_size = min_epoch_window_size
        self.regressor = GaussianProcessRegressor(kernel=kernel,
                                                  random_state=get_seed())
        self.hparam_ranges = self.inner_algo.hparam_ranges()
        self.n_hparams = len(self.hparam_ranges)
        self.n_hparam_ucb_samples = 1000
        self.ucb_nu = ucb_nu
        self.ucb_delta = ucb_delta

        # Backtracking
        self.prev_algo_states = deque(maxlen=offset_epoch + 1)
        self.offset_epoch = offset_epoch

        # Both
        self.inner_algo = inner_algo
        self.prev_algo_perf = None
        self.perf_statistic = perf_statistic


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
        saved_state = cloudpickle.dumps(self.inner_algo)
        if len(self.perf_changes) < self.min_epoch_window_size:
            vec = np.random.uniform(size=(self.n_hparams,))
            hparams = self._vec_to_hparams(vec)
            self.inner_algo.set_hparams(hparams)
            new_stats = self.inner_algo.step(trainer, epoch)
            tabular.record('ExpectedPerfChange', 0)
        else:
            vec = self._select_ucb_hparams(epoch)
            hparams = self._vec_to_hparams(vec)
            self.inner_algo.set_hparams(hparams)
            new_stats = self.inner_algo.step(trainer, epoch)
        for key, value in hparams.items():
            with tabular.prefix('HParams/'):
                tabular.record(key, value)
        perf = new_stats[self.perf_statistic]
        backtracked = False
        if self.prev_algo_perf is not None:
            # Update GP with perf change
            self.hparam_vecs.append(vec)
            self.perf_changes.append(perf - self.prev_algo_perf)
            perf_change_array = np.array(self.perf_changes)
            perf_array_scale = np.sqrt(np.mean(perf_change_array ** 2))
            perf_change_array /= perf_array_scale

            # Fit GP and plot
            self.regressor.fit(self.hparam_vecs, perf_change_array)
            for index, (key, value) in enumerate(self.hparam_ranges.items()):
                self.plot_gpr(f'{trainer.log_directory}/{key}_{epoch:05}.png',
                              key, index, epoch, perf_array_scale)

            # Backtrack if necessary
            if perf < self.prev_algo_perf and \
               len(self.prev_algo_states) > self.offset_epoch:
                prev_epoch, state = self.prev_algo_states[self.offset_epoch]
                self.inner_algo = cloudpickle.loads(state)
                logger.log(f'Backtracking at epoch {epoch} to epoch {prev_epoch}')
                backtracked = True
        if not backtracked:
            # Save the new state
            self.prev_algo_states.appendleft(
                (epoch, saved_state))
            self.prev_algo_perf = perf
        return new_stats

    def _select_ucb_hparams(self, epoch):
        sampled_hparams = np.random.uniform(size=(self.n_hparam_ucb_samples,
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

    def plot_gpr(self, filename, hparam, hparam_idx, epoch, perf_array_scale):
        N = 10000
        X_START = self.hparam_ranges[hparam][0]
        X_STOP = self.hparam_ranges[hparam][1]
        X_bel = np.linspace(X_START, X_STOP, num=N)
        sampled_hparams = np.random.uniform(size=(10000, self.n_hparams))
        sampled_hparams[:, hparam_idx] = X_bel
        mean_bel, std_bel = self.regressor.predict(sampled_hparams,
                                                   return_std=True)
        mean_bel *= perf_array_scale
        std_bel *= perf_array_scale
        tau_t = 2 * np.log(
            (1 + epoch) ** (self.n_hparams / 2 + 2) *
            np.pi**2 / (3 * self.ucb_delta)
        )  # 2*log(t^(d/2+2)π^2/(3δ))
        k_t = np.sqrt(self.ucb_nu * tau_t)
        tabular.record('UCB/kt', k_t)
        ucb = mean_bel + k_t * std_bel

        plt.clf()
        # plt.ylim((-5, 5))
        plt.xlim((X_START, X_STOP))
        plt.scatter([vec[hparam_idx] for vec in self.hparam_vecs],
                    self.perf_changes, label="Observations")
        plt.plot(X_bel, mean_bel, label="Mean prediction")
        plt.fill_between(
            X_bel,
            mean_bel - ucb,
            mean_bel + ucb,
            alpha=0.5,
            label=r"UCB Decision Surface",
        )
        # plt.fill_between(
            # X_bel,
            # mean_bel - 1.96 * std_bel,
            # mean_bel + 1.96 * std_bel,
            # alpha=0.5,
            # label=r"95% confidence interval",
        # )
        plt.legend()
        plt.xlabel(f"{hparam}")
        plt.ylabel("Perf. Improvement")
        plt.savefig(filename)
