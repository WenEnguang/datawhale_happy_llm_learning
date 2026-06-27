import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.emb_dim = config['emb_dim']
        self.hidden_dim = config['hidden_dim']

        self.fc1 = nn.Linear(self.emb_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.emb_dim)
        self.dropout = nn.Dropout(config['drop_rate'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x