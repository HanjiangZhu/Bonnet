
class BaseMetric:
    def reset(self):
        raise NotImplementedError

    def update(self, value, count):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def as_wandb_metric(self):
        raise NotImplementedError