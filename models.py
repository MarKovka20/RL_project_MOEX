from torch import nn


class Traider(nn.Module):

    def __init__(self, 
                 in_dim, 
                 hidden_dim,
                 out_dim, 
                 num_hidden_layers,
                 use_softmax = True,
                 dropout = 0.5):

        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout1d(p=dropout),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout1d(p=dropout)
                )
            for _ in range(num_hidden_layers - 2)
            ],
            nn.Linear(hidden_dim, out_dim) 
        )

        self.use_softmax = use_softmax
        if use_softmax == True:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        out =  self.model(x)
        if self.use_softmax:
            out = self.softmax(out)
        return out