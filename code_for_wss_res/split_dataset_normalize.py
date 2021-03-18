from torch_geometric.data import DataLoader 
from MyOwnDataset_normalize_train import MyOwnDataset_normalize_train
from MyOwnDataset_normalize_val import MyOwnDataset_normalize_val
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
def split_dataset_normalize(device,my_path_t,my_path_v,batch_size):
    dataset_train = MyOwnDataset_normalize_train(my_path_t)
    dataset_val = MyOwnDataset_normalize_val(my_path_v)
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_train.data=dataset_train.data.to(device)
    dataset_val.data=dataset_val.data.to(device)

   
    #LOADING BOTH DATASET
    train_loader = DataLoader(dataset_train, batch_size=batch_size)
    validation_loader = DataLoader(dataset_val, batch_size=batch_size)
    
    return {'train':train_loader,'val':validation_loader}