from stick import OutputEngine, declare_output_engine

SummaryWriter = None


@declare_output_engine
class TensorBoardOutput(OutputEngine):
    def __init__(self, log_dir, run_name, flush_secs=120, histogram_samples=1e2):
        global SummaryWriter
        try:
            if SummaryWriter is None:
                from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            pass
        try:
            if SummaryWriter is None:
                from tensorboardX import SummaryWriter
        except ImportError:
            pass
        try:
            if SummaryWriter is None:
                from tf.summary import SummaryWriter
        except ImportError:
            pass

        if SummaryWriter is None:
            raise ImportError("Could not find tensorboard API")

        self.writer = SummaryWriter(log_dir, flush_secs=flush_secs)
        self._histogram_samples = int(histogram_samples)
        self.run_name = run_name

    def log_row(self, row):
        if row.table_name == "hparams":
            flat_dict = row.as_flat_dict()
            hparams = {
                k: v for (k, v) in flat_dict.items() if not k.startswith("hparam")
            }
            metrics = {k: v for (k, v) in flat_dict.items() if k.startswith("hparam")}
            self.writer.add_hparams(hparams, metrics, run_name=self.run_name)
        for k, v in row.as_flat_dict().items():
            self.writer.add_scalar(k, v, row.step)
        self.writer.flush()

    def close(self):
        """Flush all the events to disk and close the file."""
        self.writer.close()
