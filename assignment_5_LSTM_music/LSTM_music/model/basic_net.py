from torch import nn

class BasicNet(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, num_layers=2, dropout=(0.3, 0.3)):
        super(BasicNet, self).__init__()
        self.backbone = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout[0])
        # Batchnorm helps avoid the 'single-note' prediction problem.
        # Check https://github.com/Skuldur/Classical-Piano-Composer/issues/31
        self.head = nn.Sequential(
        nn.BatchNorm1d(hidden_size),
        nn.Linear(hidden_size, int(out_size * 1.5)),
        nn.ReLU(),
        nn.BatchNorm1d(int(out_size * 1.5)),
        nn.Linear(int(out_size * 1.5), out_size))

    def forward(self, x):
        back_out = self.backbone(x.unsqueeze(dim=1))
        head_out = self.head(back_out[0].squeeze(dim=1))
        return head_out