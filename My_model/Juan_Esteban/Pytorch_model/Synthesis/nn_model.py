from torch import nn

class InvModel(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_units):
        super().__init__()
        padding = "valid"
        self.simple_conv = nn.Sequential(nn.Conv1d(in_channels=in_shape, out_channels=512, kernel_size = 2, stride=1, padding=padding),
        nn.ReLU(),
        nn.Conv1d(in_channels=512, out_channels=256, kernel_size = 2, stride=1, padding=padding),
        nn.ReLU(),
        nn.Conv1d(in_channels=256, out_channels=64, kernel_size = 2, stride=1, padding=padding),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features = 64, out_features = out_shape))
    def forward(self, x):
        return self.simple_conv(x)