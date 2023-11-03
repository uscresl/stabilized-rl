from stick import OutputEngine, declare_output_engine
import dowel
from dowel.tabular_input import TabularInput


@declare_output_engine
class DowelOutputEngine(OutputEngine):
    def __init__(self, epoch_step_key, dowel_logger, tabular=None):
        self._epoch_step_key = epoch_step_key
        self._tabular = tabular or TabularInput()
        self._logger = dowel_logger

    def log(self, prefix, key, value):
        self._tabular.record("/".join(prefix + [key]), value)
        if self._epoch_step_ is not None and key == self._epoch_step_key:
            self._logger.log(self._tabular)
            self._logger.dump_all()

    def close(self):
        self._file.close()


def global_dowel_output():
    return DowelOutputEngine(None, dowel.logger, dowel.tabular)
