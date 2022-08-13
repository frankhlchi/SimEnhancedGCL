from torch_geometric.nn import APPNP
import torch as th
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian
from scipy.stats import rankdata
from scipy.special import softmax
from pathlib import Path

def negative_sample_idx(edge_index, labels, tau, feature, k=20,  alpha =0.1, beta = 0.5,  ppr_dir = './weight', data_name ='Cora'):
    n = len(labels)
    m = len(labels.unique())
    ppr_file_path = "%s/ppr_%s_%s_%s.pt"%(ppr_dir, data_name, k, alpha)
    print ('prop negative tau:', tau, ' k:',k, ' alpha:',alpha, ' beta:', beta)
    
    if Path(ppr_file_path).is_file():
        res_ = th.load(ppr_file_path)
    else:
        labels_one_hot = F.one_hot(th.tensor([i for i in range(len(labels))]))
        LP = APPNP(k, alpha)
        res_ = LP(labels_one_hot.float(), edge_index = edge_index)
        th.save(res_, ppr_file_path)
      
    feature_file_path = "%s/feature_%s.pt"%(ppr_dir, data_name)
    if Path(feature_file_path).is_file():
        f_sim_matrix = th.load(feature_file_path)
    else:
        f_sim_matrix =  th.matmul(F.normalize(feature), F.normalize(feature).T)
        th.save(f_sim_matrix, feature_file_path)
        
    rescale = res_.mean()/f_sim_matrix.mean()
    res_ = ((1- beta)*res_ + f_sim_matrix * rescale * beta)

    res_ = th.nan_to_num(th.exp(-res_/tau)).fill_diagonal_(0)
    res_mean = res_.sum(axis=1).reshape(-1,1)/(res_.shape[0] - 1)
    res_ = res_/res_mean
    return  res_


def postive_sample_idx(edge_index, labels,  tau, feature, k=20,  alpha =0.1,  beta = 0.5, ppr_dir = './weight', data_name ='Cora'):
    n = len(labels)
    m = len(labels.unique())
    ppr_file_path = "%s/ppr_%s_%s_%s.pt"%(ppr_dir, data_name, k, alpha)
    print ('prop negative tau:', tau, ' k:',k, ' alpha:',alpha, ' beta:', beta)
    
    if Path(ppr_file_path).is_file():
        res_ = th.load(ppr_file_path)
    else:
        labels_one_hot = F.one_hot(th.tensor([i for i in range(len(labels))]))
        LP = APPNP(k, alpha)
        res_ = LP(labels_one_hot.float(), edge_index = edge_index)
        th.save(res_, ppr_file_path)
    
    feature_file_path = "%s/feature_%s.pt"%(ppr_dir, data_name)
    if Path(feature_file_path).is_file():
        f_sim_matrix = th.load(feature_file_path)
    else:
        f_sim_matrix =  th.matmul(F.normalize(feature), F.normalize(feature).T)
        th.save(f_sim_matrix, feature_file_path)
        
    rescale = res_.mean()/f_sim_matrix.mean()
    res_ = ((1- beta)*res_ + f_sim_matrix * rescale * beta)
    
    size = res_.shape[0]
    upper_limit = th.nan_to_num(th.tensor(float('inf')))/(size*10)
    res_ = th.clamp(th.nan_to_num(th.exp(res_/tau), posinf = upper_limit), max= upper_limit) - 1
    res_[range(size), range(size)] = th.clamp(res_.diag(), min = 1)
    res_ = res_/th.nan_to_num(res_.mean(axis=1).reshape(-1,1))  
    res_ = res_/res_.shape[0]

    return  res_