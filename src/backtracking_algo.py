from collections import deque
import cloudpickle

from dowel import logger
from garage.np.algos import RLAlgorithm

class BacktrackingAlgo(RLAlgorithm):

    def __init__(self, inner_algo, perf_statistic='AverageReturn',
                 offset_epoch=0):
        self.inner_algo = inner_algo
        self.prev_algo_states = deque(maxlen=offset_epoch + 1)
        self.prev_algo_perf = None
        self.perf_statistic = perf_statistic
        self.offset_epoch = offset_epoch

    def step(self, trainer, epoch):
        saved_state = cloudpickle.dumps(self.inner_algo)
        new_stats = self.inner_algo.step(trainer, epoch)

        if self.prev_algo_perf is not None and \
           new_stats[self.perf_statistic] < self.prev_algo_perf and \
           len(self.prev_algo_states) > self.offset_epoch:
            prev_epoch, state = self.prev_algo_states[self.offset_epoch]
            self.inner_algo = cloudpickle.loads(state)
            logger.log(f'Backtracking at epoch {epoch} to epoch {prev_epoch}')
        else:
            self.prev_algo_states.appendleft(
                (epoch, saved_state))
            self.prev_algo_perf = new_stats[self.perf_statistic]
        return new_stats

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

