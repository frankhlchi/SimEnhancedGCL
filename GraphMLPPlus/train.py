from __future__ import division
from __future__ import print_function
import random
import time
import argparse
import numpy as np
import torch
import os
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import yaml
from yaml import SafeLoader
from pathlib import Path


from models import GMLP
from utils import load_citation, accuracy, get_A_r
import warnings
from propagate import negative_sample_idx, positive_sample_idx
from propagate_global import negative_sample_idx_g, positive_sample_idx_g
warnings.filterwarnings('ignore')

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--data', type=str, default='cora',
                    help='dataset to be used')
parser.add_argument('--alpha', type=float, default=2.0,
                    help='To control the ratio of Ncontrast loss')
parser.add_argument('--batch_size', type=int, default=2048,
                    help='batch size')
parser.add_argument('--order', type=int, default=2,
                    help='to compute order-th power of adj')
parser.add_argument('--tau', type=float, default=1.0,
                    help='temperature for Ncontrast loss')
parser.add_argument('--rand_seed', type=int, default=1, help='Random seed')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def seed_set(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
## get data
seed_set(args.rand_seed)
data_name = args.data
print (data_name, ' random seed:',args.rand_seed)
adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.data, 'AugNormAdj',True)
feature_input = features.clone()

config_g = yaml.load(open('config.yaml'), Loader=SafeLoader)
tau_neg_prop = config_g [data_name]['neg_tau']
tau_pos_prop = config_g [data_name]['pos_tau']
f_weight = config_g[data_name]['f_weight']
f_sim_matrix = torch.matmul(F.normalize(feature_input), F.normalize(feature_input).T) 

if data_name != 'pubmed':
    pos_label = positive_sample_idx(adj.coalesce().indices(), labels,  tau_pos_prop , f_sim_matrix, args.data, beta = f_weight)
    neg_label = negative_sample_idx(adj.coalesce().indices(), labels,  tau_neg_prop , f_sim_matrix, args.data, beta = f_weight)
else:
    print ('global')
    pos_label = positive_sample_idx_g(adj.coalesce().indices(), labels,  tau_pos_prop , f_sim_matrix, args.data, beta = f_weight)
    neg_label = negative_sample_idx_g(adj.coalesce().indices(), labels,  tau_neg_prop , f_sim_matrix, args.data, beta = f_weight)

    
## Model and optimizer
model = GMLP(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            )
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


if args.cuda:
    model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    pos_label = pos_label.cuda()
    neg_label = neg_label.cuda()

def Ncontrast(x_dis, pos_label,neg_label, tau = 1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp( tau * x_dis)
    x_dis_sum = torch.sum(x_dis*neg_label, 1)
    x_dis_sum_pos = torch.sum(x_dis*pos_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def get_batch(batch_size):
    """
    get a batch of feature & adjacency matrix
    """
    all_index = np.array(list(set(np.arange(len(labels))) - set(idx_train.cpu().numpy())))
    rand_indx = np.random.choice(all_index , batch_size - len(idx_train), replace=False)
    rand_indx = torch.tensor(np.concatenate((idx_train.cpu(), rand_indx))).cuda()
    
    features_batch = features[rand_indx]
    pos_label_batch = pos_label[rand_indx,:][:,rand_indx]
    neg_label_batch = neg_label[rand_indx,:][:,rand_indx]
    return features_batch, pos_label_batch, neg_label_batch

def train():
    features_batch, pos_label_batch, neg_label_batch = get_batch(batch_size=args.batch_size)
    model.train()
    optimizer.zero_grad()
    output, x_dis = model(features_batch)
    loss_train_class = F.nll_loss(output[idx_train], labels[idx_train])
    loss_Ncontrast = Ncontrast(x_dis, pos_label_batch, neg_label_batch, tau = args.tau)
    loss_train = loss_train_class + loss_Ncontrast * args.alpha
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return 

def test():
    model.eval()
    output = model(features)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    return acc_test, acc_val

best_accu = 0
best_val_acc = 0
print('\n'+'training configs', args)
for epoch in tqdm(range(args.epochs)):
    train()
    tmp_test_acc, val_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc


log_file = open(r"log_%s.txt"%data_name, encoding="utf-8",mode="a+")  
with log_file as file_to_be_write:  
    print('tau', 'order', \
            'batch_size', 'hidden', \
                'alpha', 'lr', \
                    'weight_decay', 'data', \
                         'vali_acc', 'test_acc', file=file_to_be_write, sep=',')
    print(args.tau, args.order, \
         args.batch_size, args.hidden, \
             args.alpha, args.lr, \
                 args.weight_decay, args.data, \
                      best_val_acc.item(),test_acc.item(), file=file_to_be_write, sep=',')


