import torch

class StepsPerEpochDataLoader:
    def __init__(self, dataloader, n_step):
        self.num_steps = n_step
        self.idx = 0
        self.loader = dataloader
        self.iter_loader = iter(dataloader)
    
    def __iter__(self):
        return self

    def __len__(self):
        return self.num_steps

    def __next__(self):
        # if reached number of steps desired, stop
        if self.idx == self.num_steps:
            self.idx = 0
            raise StopIteration
        else:
            self.idx += 1
        # while True
        try:
            return next(self.iter_loader)
        except StopIteration:
            # reinstate iter_loader, then continue
            self.iter_loader = iter(self.loader)
            return next(self.iter_loader)
        
class DataLoaderCUDABus:
    """
    Responsible for sending batches of data to the GPU.
    """

    def __init__(self, loader, enable_prefetch: bool = True):
        self.loader = loader
        self.iter_loader = iter(loader)
        self.enable_prefetch = enable_prefetch
        self.end_of_dataloader = False
        if self.enable_prefetch:
            self.stream = torch.cuda.Stream()

    def on_start_epoch(self):
        self.end_of_dataloader = False
        self.preload()
        self.iter_loader = iter(self.loader)

    def preload(self):
        if not self.enable_prefetch:
            return

        try:
            self.next_batch_list = next(self.iter_loader)
            while self.next_batch_list is None:
                self.next_batch_list = next(self.iter_loader)
        except StopIteration:
            self.next_batch_list = None
            self.end_of_dataloader = True
            return
 
        with torch.cuda.stream(self.stream):
            for i in range(len(self.next_batch_list)):
                self.next_batch_list[i] = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in self.next_batch_list[i].items()}

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.loader)

    def __next__(self):
        if self.end_of_dataloader:
            return None

        if self.enable_prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)

            batch_list = self.next_batch_list
            if batch_list is not None:
                for i in range(len(batch_list)):
                    for k in batch_list[i].keys():
                        if isinstance(batch_list[i][k], torch.Tensor):
                            batch_list[i][k].record_stream(torch.cuda.current_stream())

            self.preload()
            return batch_list
        else:
            batch_list = next(self.iter_loader)
            if batch_list is not None:
                for i in range(len(batch_list)):
                    batch_list[i] = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_list[i].items()}

            return batch_list