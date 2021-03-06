import torch 
import matplotlib.pyplot as plt
from split_dataset import split_dataset
from split_dataset_normalize import split_dataset_normalize
import numpy as np
from model import GCN
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
meshes_path='new_mesh'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device='cpu'
#dataset=my_train_fn()
#loader=DataLoader(dataset,batch_size=1)
#dataset=MyOwnDataset(root='../Meshes_vtp',)
hyperParams={
    "lr": 0.001,
    "epochs": 1000,
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


#model.apply(weights_init_uniform_rule)
#%% SETTING FOR TRAINING
model = GCN()
#loader=DataLoader(dataset,batch_size=1)
data_loaders=split_dataset(device,
                           meshes_path, 
                           hyperParams['batch_size'], 
                           hyperParams['val_split']
                           )


print('-'*40+'\nDATASET CREATED NORMALIZED [0,1]')



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
#norm=GenerateMeshNormals()
from training_code import training
model,saved_loss,train_metr,val_metr=training(hyperParams,model,data_loaders,optimizer_1,scheduler_1)
#%% LOSS PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),saved_loss[0],label='Train')
ax.plot(range(hyperParams['epochs']),saved_loss[1],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
plt.yscale("log")
plt.show()
#%% METRIC PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),train_metr[0],label='Train')
ax.plot(range(hyperParams['epochs']),val_metr[0],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('Metric on X')
plt.yscale("log")
plt.show()
#%% Training with different normalization
from model_normalize import Feast_GCN
norm_path='diff_norm_mesh'
model_abs=Feast_GCN()
model_abs=model_abs.to(device) 
data_loaders_norm_abs=split_dataset_normalize(device,
                                    norm_path, 
                                    hyperParams['batch_size'], 
                                    hyperParams['val_split']
                                    )
optimizer_2 = torch.optim.Adam(model_abs.parameters(), 
                             lr=hyperParams['lr'], 
                             weight_decay=hyperParams['weight_decay']
                             )

scheduler_2 = ReduceLROnPlateau(
        optimizer_2,
        mode='min',
        factor=0.1,
        patience=100,
        verbose=True
    )
print('-'*40+'\nDATASET CREATED NORMALIZED [-1, 1]')
#%%
from training_code import training


model_abs,saved_loss_abs,train_metr_abs,val_metr_abs=training(hyperParams,model_abs,data_loaders_norm_abs,optimizer_2,scheduler_2)
#%% LOSS PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),saved_loss_abs[0],label='Train')
ax.plot(range(hyperParams['epochs']),saved_loss_abs[1],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss_abs')
plt.yscale("log")
plt.show()
#%% METRIC PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),train_metr_abs[0],label='Train')
ax.plot(range(hyperParams['epochs']),val_metr_abs[0],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('Metric_abs on X')
plt.yscale("log")
plt.show()

#%% METRIC PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),train_metr[0],label='0->1')
ax.plot(range(hyperParams['epochs']),train_metr_abs[0],label='-1->1')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('Metric on X')
plt.yscale("log")
plt.show()
#%%
from results import apply_model_on_mesh,predict_on_dataloader

#wss_maxm,wss_minm,vrtx_maxm,vrtx_minm=predict_on_dataloader(model,data_loaders)
#predict_on_dataloader(model,data_loaders,data_loaders_training=None)
#%%
# #%%
# file_name='../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated/Clipped/aorta_0_dec_cl2.vtp'
# # out_name='../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated/Predicted/aorta_0_pred.vtp'
# value = input("Do you want to make a prediction on "+file_name+"? [y/n]\n")
# if value=='y':
#     value = input("Choose a name for the prediction file:\n")
    
#     out_name='../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated/Predicted/'+value+'.vtp'
#my_path='../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated/Caso sano'
#apply_model_on_mesh(my_path,model,device,data_loaders,known=True)

