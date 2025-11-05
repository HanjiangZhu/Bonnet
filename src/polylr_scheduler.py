from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, ema_loss=None, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        if current_step >= self.max_steps:
            return
        
        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

if __name__ == "__main__":
    import torch
    import torch.optim as optim

    steps_per_epoch = 100
    max_steps = 250_000
    epochs_to_compute = 1000

    model = torch.nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = PolyLRScheduler(optimizer, 0.01, max_steps, 0.9)

    lrs = []

    for i in range(epochs_to_compute):
        for j in range(250):
            scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])
    
    import matplotlib.pyplot as plt
    plt.plot(lrs)
    plt.savefig("polylr_new.png")