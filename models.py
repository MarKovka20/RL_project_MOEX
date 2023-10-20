from torch import nn


class Traider(nn.Module):

    def __init__(self, in_dim, out_dim, num_hidden_layers):

        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.Dropout1d(p=0.5),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(in_dim * 2, in_dim * 2),
                    nn.Dropout1d(p=0.5),
                    nn.ReLU()
                )
            for _ in range(num_hidden_layers - 2)
            ],
            nn.Linear(in_dim * 2, out_dim),
            # nn.Softmax()
        )

    def forward(self, x):

        return self.model(x)