import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
from eval import label_classification

import datetime
import pandas as pd
import numpy as np
import os
from propagate import negative_sample_idx,  postive_sample_idx
torch.autograd.set_detect_anomaly(True)

def train(model: Model, x, edge_index, neg_mask, pos_mask, batch_size):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    loss = model.loss(z1, z2, neg_mask, pos_mask, batch_size = batch_size)
    loss.backward()
    optimizer.step()

    return loss.item()

def test(model: Model, x, edge_index, y,train_indices, valid_indices,  test_indices, final=False):
    model.eval()
    z = model(x, edge_index)
    return label_classification(z, y, train_indices, valid_indices, test_indices,ratio=0.1)

def embed(model: Model, x, edge_index, y,train_indices, valid_indices,  test_indices, final=False):
    model.eval()
    z = model(x, edge_index)
    return z


def seed_set(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--rd_seed', type=int, default=0)
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    tau_neg_prop = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]['neg_tau']
    tau_pos_prop = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]['pos_tau']
    f_weight = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]['f_weight']
    seed_set(args.rd_seed)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']
    tau = config['tau']
    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']

    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    def get_dataset(path, name):
        assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
        name = 'dblp' if name == 'DBLP' else name

        return (CitationFull if name == 'dblp' else Planetoid)(
            path,
            name,
            transform=T.NormalizeFeatures())

    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]
    
    labels = data.y.clone()
    idx = np.arange(0,len(labels))
    np.random.shuffle(idx)
    indices = torch.as_tensor(idx)
    train_indices = indices[: int(len(labels)*0.1)]
    valid_indices = indices[int(len(labels)*0.1) : int(len(labels)*0.2)]
    test_indices = indices[int(len(labels)*0.2):]
    train_mask = torch.zeros_like(labels)
    train_mask [train_indices]= 1
    train_mask = train_mask.bool() 
    
    print ('train_indices', train_indices)
    print ('valid_indices',  valid_indices)
    print ('test_indices', test_indices)
    print ('train #:', len(train_indices),'valid #:', len( valid_indices),' test #:', len( test_indices ))

    weight_dir = './weight'
    pos_mask = postive_sample_idx(data.edge_index, labels, float(tau_pos_prop), feature = data.x.clone(), \
                                  k=20,  alpha =0.1, beta= f_weight, ppr_dir =weight_dir, data_name = args.dataset)
    neg_mask = negative_sample_idx(data.edge_index, labels, float(tau_neg_prop),feature = data.x.clone(),\
                                   k = 20,  alpha = 0.1, beta= f_weight,ppr_dir = weight_dir, data_name = args.dataset)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    neg_mask = neg_mask.to(device)
    pos_mask = pos_mask.to(device)
    data = data.to(device)

    encoder = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    
    if args.dataset == 'PubMed' or args.dataset == 'DBLP':
        batch_size =5000
    else:
        batch_size = 0
    print ('batch_size: ',  batch_size)     
    for epoch in range(1, num_epochs + 1):
       
        loss = train(model, data.x, data.edge_index,neg_mask,pos_mask, batch_size)
        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    print("=== Final ===")
    result = test(model, data.x, data.edge_index, data.y, train_indices, valid_indices, test_indices, final=True)
    file_name = f"{args.dataset}_{drop_edge_rate_1}_{drop_edge_rate_2}_{drop_feature_rate_1}_{drop_feature_rate_2}_{tau_pos_prop}_{tau_neg_prop}.csv"
    pd.DataFrame({'Time':[str(datetime.datetime.now())], 'Valid_Accuracy':[result['valid']['mean']], 'Test_Accuracy':[result['test']['mean']]}).to_csv(file_name, mode='a')
