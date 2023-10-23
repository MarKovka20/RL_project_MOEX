import argparse
import random, os

import numpy as np
import torch

from env import Environment
from models import Traider
from trainer import Trainer


def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

parser = argparse.ArgumentParser()

parser.add_argument("--observation_dim", default=80, type=int)
parser.add_argument("--num_hidden_layers", default=3, type=int)
parser.add_argument("--hidden_dim", default=500, type=int)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--use_softmax", default=True, type=bool)
parser.add_argument("--n_actions", default=2, type=int)
parser.add_argument("--levels", default=40, type=int)
parser.add_argument("--learning_rate", default=0.0001, type=float)
parser.add_argument("--epochs_limit", default=50, type=int)
parser.add_argument("--update_online_model_step", default=5, type=int)
parser.add_argument("--update_target_model_step", default=250)
parser.add_argument("--epsilon", default=0.15, type=float)
parser.add_argument("--transaction", default=100.0)
parser.add_argument("--batch_size", default=64)
parser.add_argument("--gamma", default=0.9)
parser.add_argument("--memory_limit", default=10000)
parser.add_argument("--train_data_path", default="data/train.data")
parser.add_argument("--test_data_path", default="data/test.data")
parser.add_argument("--save_path", default="traider.pt")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--seed", default=42)

args = parser.parse_args()
seed_everything(args.seed)

online_model = Traider(in_dim=args.observation_dim, 
                       hidden_dim=args.hidden_dim,
                       out_dim=args.n_actions, 
                       num_hidden_layers=args.num_hidden_layers,
                       dropout = args.dropout,
                       use_softmax = args.use_softmax)
target_model = Traider(in_dim=args.observation_dim, 
                       hidden_dim=args.hidden_dim,
                       out_dim=args.n_actions, 
                       num_hidden_layers=args.num_hidden_layers,
                       dropout = args.dropout,
                       use_softmax = args.use_softmax)

train_environment = Environment(args.train_data_path, args.levels, args.observation_dim)

test_environment = Environment(args.test_data_path, 1, args.observation_dim)

loss_fn = torch.nn.HuberLoss()

optimizer = torch.optim.Adam(online_model.parameters(), lr=args.learning_rate)

save_path = "traider__" + "__".join([f'{k}={v}' for k, v in args._get_kwargs()]) + '.pt'

# args.save_path = "traider__" + "__".join([f'{k}={v}' for k, v in args._get_kwargs()]) + '.pt'
print(args.save_path)
model = torch.nn.Linear(1, 1)
print(args.save_path)
torch.save(model.state_dict(), args.save_path)
trainer = Trainer(
    train_environment=train_environment,
    test_environment=test_environment,
    online_model=online_model,
    target_model=target_model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    cfg=args
)

# trainer.train()