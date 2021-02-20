import torch 
import matplotlib.pyplot as plt
from split_dataset_normalize import split_dataset_normalize as split_dataset
import numpy as np
from model_normalize import Feast_GCN as GCN
import pyvista as pv
from torch_geometric.transforms import FaceToEdge,RandomRotate,Compose,GenerateMeshNormals
from torch_geometric.data import Data,DataLoader,InMemoryDataset
from losses import nmse,NMAE
from torch.optim.lr_scheduler import ReduceLROnPlateau


def denormalize_wss(point_array,maxm,minm):
    #maxm=point_array.max()
    #minm=point_array.min()
    # print("OLD MAX: ",maxm)
    # print("OLD MIN: ",minm)
    #print(maxm)
    maxm=maxm[0].detach().numpy()
    minm=minm[0].detach().numpy()
    
    new_array=((point_array)*(maxm-minm))+minm
    # print("NEW MAX: ",new_array.max())
    # print("NEW MIN: ",new_array.min())
    return new_array
#%% SETTING PARAMS
meshes_path='1cm_edge_asc/whole_dataset'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device='cpu'
#dataset=my_train_fn()
#loader=DataLoader(dataset,batch_size=1)
#dataset=MyOwnDataset(root='../Meshes_vtp',)
hyperParams={
    "lr": 0.0001,
    "epochs":1000,
    "batch_size":1,
    "val_split":0.05,
    "loss":torch.nn.MSELoss(),
    "weight_decay":5e-6
    
    }

# #WEIGHT INITIALIZATION 
# def weights_init_uniform_rule(m):
#         classname = m.__class__.__name__
#         # for every Linear layer in a model..
#         if classname.find('FeaStConv') != -1:
#             # get the number of the inputs
#             n = m.in_channels
#             y = 1.0/np.sqrt(n)
#             m.weight.data.uniform_(-y, y)
#             #m.bias.data.fill_(0)
#         if classname.find('Linear') != -1:
#             # get the number of the inputs
#             n = m.in_features
#             y = 1.0/np.sqrt(n)
#             m.weight.data.uniform_(-y, y)
#     # create a new model with these weights
model = GCN()

#model.apply(weights_init_uniform_rule)
#%% SETTING FOR TRAINING

#loader=DataLoader(dataset,batch_size=1)
data_loaders=split_dataset(device,
                           meshes_path, 
                           hyperParams['batch_size'], 
                           hyperParams['val_split']
                           )

print('-'*40+'\nDATASET CREATED')



optimizer_1 = torch.optim.Adam(model.parameters(), 
                             lr=hyperParams['lr'], 
                             weight_decay=hyperParams['weight_decay']
                             )

scheduler_1 = ReduceLROnPlateau(
        optimizer_1,
        mode='min',
        factor=0.1,
        patience=100,
        verbose=True
    )

criterion = hyperParams['loss']

model=model.to(device)  

#%% TRAINING

from training_code import training
model,saved_loss,train_metr,val_metr,cos_simil=training(hyperParams,model,data_loaders,optimizer_1,scheduler_1)
#%% NMSE PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),saved_loss[0],label='Train')
ax.plot(range(hyperParams['epochs']),saved_loss[1],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('NMSE')
plt.yscale("log")
plt.show()
#%% NMSE PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),saved_loss[0],label='Train')
ax.plot(range(hyperParams['epochs']),saved_loss[1],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('NMSE')
#plt.yscale("log")
plt.show()
#%% COSINE SIMILARITY PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),cos_simil[0],label='Train')
ax.plot(range(hyperParams['epochs']),cos_simil[1],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('COSINE SIMILARITY')
#plt.yscale("log")
plt.show()
#%%
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),cos_simil[0],label='Train')
ax.plot(range(hyperParams['epochs']),cos_simil[1],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('COSINE SIMILARITY')
plt.yscale("log")
plt.show()

#%% NMAE PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),np.sum(train_metr,axis=0),label='Train')
ax.plot(range(hyperParams['epochs']),np.sum(val_metr,axis=0),label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('NMAE')
plt.yscale("log")
plt.show()
#%% NMAE PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),np.sum(train_metr,axis=0),label='Train')
ax.plot(range(hyperParams['epochs']),np.sum(val_metr,axis=0),label='Val')
ax.legend()
ax.set_xlabel('Epochs')
#ax.set_ylabel('NMAE')
#plt.yscale("log")
plt.show()
#%%
#from results_for_norm import apply_model_on_mesh,predict_on_dataloader
from results_normalize import apply_model_on_mesh,predict_on_dataloader
#wss_maxm,wss_minm,vrtx_maxm,vrtx_minm=predict_on_dataloader(model,data_loaders)
predict_on_dataloader(meshes_path,model,data_loaders,data_loaders_training=None)
