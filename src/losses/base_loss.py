import torch.nn as nn

class BaseLoss:
    # TODO: actually reset_metrics here :D
    def reset_metrics(self):
        for metric in self.metrics.values():
            if hasattr(metric, "reset"):
                metric.reset()

    def on_start_epoch(self):
        self.reset_metrics()

    def on_end_epoch(self):
        pass
