import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import gc
import numpy as np

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, neg_mask: torch.Tensor, pos_mask: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        x1 = between_sim.diag() + \
            (refl_sim * neg_mask.to(torch.float64).fill_diagonal_(0)).sum(1) + \
                            (between_sim * neg_mask.to(torch.float64).fill_diagonal_(0)).sum(1) 
        
        numerator = f((self.sim(z1, z1) * pos_mask.to(torch.float64).fill_diagonal_(0)).sum(1)) + \
                                      f((self.sim(z1, z2) * pos_mask.to(torch.float64)).sum(1))
        
        loss = -torch.log(numerator / x1)
        return loss

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, neg_mask: torch.Tensor,pos_mask: torch.Tensor,
                          batch_size: int):
        
        #print ('batched_semi_loss')
        f = lambda x: torch.exp(x / self.tau)
        num_nodes = z1.size(0)
        mask = torch.tensor(np.random.choice(np.arange(0, num_nodes), size = min(batch_size,num_nodes), replace = False))
        
        refl_sim = f(self.sim(z1[mask], z1))
        between_sim = f(self.sim(z1[mask], z2))

        x1 = (refl_sim * neg_mask[mask]).sum(1) + \
                    (between_sim * neg_mask[mask]).sum(1) - \
                        (refl_sim * neg_mask[mask])[:, mask].diag() -\
                            (between_sim * neg_mask[mask])[:, mask].diag() +\
                                   between_sim[:, mask].diag()
      
        numerator = f((self.sim(z1[mask], z1) * pos_mask[mask]).sum(1)) + \
                    f((self.sim(z1[mask], z2) * pos_mask[mask]).sum(1)) -\
                        f((self.sim(z1[mask], z1) * pos_mask[mask])[:, mask].diag()) 

        loss = -torch.log(numerator / x1)
        return loss
        

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, neg_mask: torch.Tensor, pos_mask: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            print ('normal loss used')
            l1 = self.semi_loss(h1, h2, neg_mask, pos_mask)
            l2 = self.semi_loss(h2, h1, neg_mask, pos_mask)
        else:
            print ('batched loss used')
            l1 = self.batched_semi_loss(h1, h2, neg_mask, pos_mask, batch_size)
            l2 = self.batched_semi_loss(h2, h1, neg_mask, pos_mask,batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x