from torch import nn

class MLP(nn.Module):
    def __init__(self, input_features, num_classes):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_features),
            nn.Linear(input_features, 1000),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1000, num_classes)
        )




    def forward(self, x):
        x = self.layers(x)
        return x