import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
import logging
from polylr_scheduler import PolyLRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1,
                 reduce_lr_on_plateau = False,
                 reduce_min_lr_on_plateau = False,
                 plateau_factor = 0.2,
                 plateau_patience = 30
        ):
        assert warmup_steps < first_cycle_steps
        
        self.reduce_min_lr_on_plateau = reduce_min_lr_on_plateau
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        self.best_ema_loss = None
        self.no_improvement_counter = 0
        self.delta_improv = 5 * 1e-3
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def on_end_epoch(self, ema_loss):
        if self.reduce_lr_on_plateau:
            if self.best_ema_loss is not None:
                if ema_loss - self.best_ema_loss > -self.delta_improv:
                    self.no_improvement_counter += 1
                else:
                    self.no_improvement_counter = 0
                    self.best_ema_loss = ema_loss
                
                if self.no_improvement_counter > self.plateau_patience:
                    new_max_lr = self.base_max_lr * self.plateau_factor
                    new_max_lr = max(new_max_lr, self.min_lr)
                    self.no_improvement_counter = 0

                    if new_max_lr < self.base_max_lr:
                        self.base_max_lr = new_max_lr
                        logging.info(f"Plateau reached! Decreasing max_lr to {self.max_lr:.4f}")
            else:
                self.best_ema_loss = ema_loss

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.max_lr = max(self.max_lr, self.min_lr)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

if __name__ == "__main__":
    import torch
    import torch.optim as optim

    steps_per_epoch = 250
    epochs_to_compute = 1000
    max_steps = 250_000

    model = torch.nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, 250, 1.25, 0.01, 0.000001, 125, 0.9)

    lrs = []

    for i in range(epochs_to_compute):
        for j in range(steps_per_epoch):
            scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])
    
    lrs2 = []

    poly_scheduler = PolyLRScheduler(optimizer, 0.01, max_steps, 0.9)
    for i in range(epochs_to_compute):
        for j in range(steps_per_epoch):
            poly_scheduler.step()
        lrs2.append(optimizer.param_groups[0]['lr'])

    # Plot but make the y-axis log scale
    # Don't include 0 in the y (only the y range)
    import matplotlib.pyplot as plt
    plt.plot(lrs, label="CosineAnnealingWarmupRestarts")
    plt.plot(lrs2, label="PolyLRScheduler")
    plt.yscale("log")
    plt.ylim(1e-7, 0.01)
    plt.legend()
    plt.savefig("PolyLR_vs_Cosine.png")