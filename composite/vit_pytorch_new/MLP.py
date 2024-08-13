from torch import nn

class MLP(nn.Module):
    def __init__(self, input_features, num_classes):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_features),
            nn.Linear(input_features, 2000),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(1000, num_classes),
            # nn.ReLU(),
        )




    def forward(self, x):
        x = self.layers(x)
        return x