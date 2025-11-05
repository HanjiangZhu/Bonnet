import torch
import torch.nn as nn

def create_normalization_from_name(name: str, num_features: int):
    if name == 'batch':
        return nn.BatchNorm1d(num_features)
    elif name == 'instance':
        return nn.InstanceNorm1d(num_features)
    elif name == 'layer':
        return nn.LayerNorm(num_features)
    elif name == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=num_features)
    elif name is not None:
        raise NotImplementedError(f'Normalization {name} not implemented')
    else:
        return None


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None, normalization: str = None, dropout: float = 0.0):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        
        self.norm = create_normalization_from_name(normalization, size_out)

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)

        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        result = x_s + dx

        if hasattr(self, 'dropout'):
            result = self.dropout(result)

        if self.norm is not None:
            result = self.norm(result)
        
        return result