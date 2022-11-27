from collections import deque
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import numpy as np
from matplotlib import pyplot as plt
import cloudpickle

from dowel import logger, tabular
from garage.experiment.deterministic import get_seed
from garage.np.algos import RLAlgorithm

DEFAULT_GP_KERNEL = (
    Matern(length_scale=0.1, nu=1.5, length_scale_bounds="fixed")
    + ConstantKernel()
    + (WhiteKernel() * Matern(length_scale=0.1, nu=1.5, length_scale_bounds="fixed"))
)
# DEFAULT_GP_KERNEL = (Matern(length_scale=0.05, nu=1.5,
# length_scale_bounds="fixed") +
# ConstantKernel())


class UCBBacktrackAlgo(RLAlgorithm):
    def __init__(
        self,
        inner_algo,
        perf_statistic="UndiscountedReturns",
        kernel=DEFAULT_GP_KERNEL,
        epoch_window_size=None,
        min_epoch_window_size=8,
        offset_epoch=0,
        ucb_nu=1.0,
        ucb_delta=0.5,
        backtrack_decrease_prob=0.8,
    ):

        # Both
        self.inner_algo = inner_algo
        self.prev_algo_perf = None
        self.perf_statistic = perf_statistic

        # UCB
        self.perf_changes = deque(maxlen=epoch_window_size)
        self.hparam_vecs = deque(maxlen=epoch_window_size)
        self.min_epoch_window_size = min_epoch_window_size
        self.regressor = GaussianProcessRegressor(
            kernel=kernel, random_state=get_seed()
        )
        self.hparam_ranges = self.inner_algo.hparam_ranges()
        self.n_hparams = len(self.hparam_ranges)
        self.n_hparam_ucb_samples = 1000
        self.ucb_nu = ucb_nu
        self.ucb_delta = ucb_delta
        self.prev_hparam_vec = None

        # Backtracking
        self.prev_algo_states = deque(maxlen=offset_epoch + 1)
        self.offset_epoch = offset_epoch
        self.backtrack_decrease_prob = backtrack_decrease_prob
        self.total_backtracks = 0

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

    def _should_backtrack(self, perf, prev_perf):
        if self.backtrack_decrease_prob is None:
            return np.mean(perf) < np.mean(prev_perf)
        else:
            n_samples = 1000
            perf_samples = np.random.choice(perf, n_samples, replace=True)
            prev_perf_samples = np.random.choice(prev_perf, n_samples, replace=True)
            prob_perf_decrease = np.mean(perf_samples < prev_perf_samples)
            return prob_perf_decrease > self.backtrack_decrease_prob

    def step(self, trainer, epoch):
        # Because we save the state _here_ before running step (and hence
        # getting perf), offset_epoch == 0.
        saved_state = cloudpickle.dumps(self.inner_algo)
        if len(self.perf_changes) < self.min_epoch_window_size:
            hvec = np.random.uniform(size=(self.n_hparams,))
            hparams = self._vec_to_hparams(hvec)
            self.inner_algo.set_hparams(hparams)
            new_stats = self.inner_algo.step(trainer, epoch)
            tabular.record("ExpectedPerfChange", 0)
        else:
            hvec = self._select_ucb_hparams(epoch)
            hparams = self._vec_to_hparams(hvec)
            self.inner_algo.set_hparams(hparams)
            new_stats = self.inner_algo.step(trainer, epoch)
        for key, value in hparams.items():
            with tabular.prefix("HParams/"):
                tabular.record(key, value)
        perf = new_stats[self.perf_statistic]
        backtracked = False
        if self.prev_algo_perf is not None:
            assert self.prev_hparam_vec is not None
            # Update GP with perf change
            self.hparam_vecs.append(self.prev_hparam_vec)
            self.perf_changes.append(np.mean(perf) - np.mean(self.prev_algo_perf))
            perf_change_array = np.array(self.perf_changes)
            perf_array_scale = np.sqrt(np.mean(perf_change_array**2))
            perf_change_array /= perf_array_scale

            # Fit GP and plot
            self.regressor.fit(self.hparam_vecs, perf_change_array)
            for index, (key, value) in enumerate(self.hparam_ranges.items()):
                self.plot_gpr(
                    f"{trainer.log_directory}/{key}_{epoch:05}.png",
                    key,
                    index,
                    epoch,
                    perf_array_scale,
                )

            # Backtrack if necessary
            if (
                self._should_backtrack(perf, self.prev_algo_perf)
                and len(self.prev_algo_states) > self.offset_epoch
            ):
                prev_epoch, state = self.prev_algo_states[self.offset_epoch]
                self.inner_algo = cloudpickle.loads(state)
                logger.log(f"Backtracking at epoch {epoch} to epoch {prev_epoch}")
                backtracked = True
                self.total_backtracks += 1
        if not backtracked:
            # Save the state from before we did "this step"
            # Save the performance from this step, which was an evaluation of that state
            self.prev_algo_states.appendleft((epoch, saved_state))
            self.prev_algo_perf = perf
            self.prev_hparam_vec = hvec
        tabular.record("backtracked_now", backtracked)
        tabular.record("total_backtracks", self.total_backtracks)
        return new_stats

    def _select_ucb_hparams(self, epoch):
        sampled_hparams = np.random.uniform(
            size=(self.n_hparam_ucb_samples, self.n_hparams)
        )
        pred_mu, pred_std = self.regressor.predict(sampled_hparams, return_std=True)
        tau_t = 2 * np.log(
            (1 + epoch) ** (self.n_hparams / 2 + 2) * np.pi**2 / (3 * self.ucb_delta)
        )  # 2*log(t^(d/2+2)π^2/(3δ))
        k_t = np.sqrt(self.ucb_nu * tau_t)
        ucb = pred_mu + k_t * pred_std
        max_idx = np.argmax(ucb)
        tabular.record("ExpectedPerfChange", pred_mu[max_idx])
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
        X_START = [x[0] for x in self.hparam_ranges.values()]
        X_STOP = [x[1] for x in self.hparam_ranges.values()]
        X_bel = np.linspace(X_START, X_STOP, num=N)
        # sampled_hparams = np.random.uniform(size=(N, self.n_hparams))
        # sampled_hparams[:, hparam_idx] = X_bel
        mean_bel, std_bel = self.regressor.predict(X_bel, return_std=True)
        mean_bel *= perf_array_scale
        std_bel *= perf_array_scale
        tau_t = 2 * np.log(
            (1 + epoch) ** (self.n_hparams / 2 + 2) * np.pi**2 / (3 * self.ucb_delta)
        )  # 2*log(t^(d/2+2)π^2/(3δ))
        k_t = np.sqrt(self.ucb_nu * tau_t)
        tabular.record("UCB/kt", k_t)
        ucb = mean_bel + k_t * std_bel

        plt.clf()
        # plt.ylim((-5, 5))
        plt.xlim((X_START[hparam_idx], X_STOP[hparam_idx]))
        plt.scatter(
            [vec[hparam_idx] for vec in self.hparam_vecs],
            self.perf_changes,
            label="Observations",
        )
        plt.plot(X_bel[:, hparam_idx], mean_bel, label="Mean prediction")
        plt.fill_between(
            X_bel[:, hparam_idx],
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
