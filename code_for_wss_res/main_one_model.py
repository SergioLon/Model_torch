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
import os
import matplotlib.font_manager
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
meshes_path_t='dataset/training'
meshes_path_v='dataset/val'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device='cpu'
#dataset=my_train_fn()
#loader=DataLoader(dataset,batch_size=1)
#dataset=MyOwnDataset(root='../Meshes_vtp',)
hyperParams={
    "lr": 0.01,
    "epochs":600,
    "batch_size":1,
    "val_split":0.05,
    "loss":torch.nn.MSELoss(),
    "weight_decay":5e-6
    
    }

#WEIGHT INITIALIZATION 
def weights_init_uniform_rule(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('FeaStConv') != -1:
            # get the number of the inputs
            # n = m.in_channels
            # y = 1.0/np.sqrt(n)
            #m.weight.data.uniform_(-y, y)
            torch.nn.init.normal_(m.weight,mean=0,std=0.3)
            torch.nn.init.zeros_(m.bias)
            #m.bias.data.fill_(0)
        if classname.find('Conv1d') != -1:
            # get the number of the inputs
            # n = m.in_features
            # y = 1.0/np.sqrt(n)
            # m.weight.data.uniform_(-y, y)
            torch.nn.init.normal_(m.weight,mean=0,std=0.3)
            torch.nn.init.zeros_(m.bias)
    # create a new model with these weights
model = GCN()

model.apply(weights_init_uniform_rule)
#%% SETTING FOR TRAINING

#loader=DataLoader(dataset,batch_size=1)
data_loaders=split_dataset(device,
                           meshes_path_t,
                           meshes_path_v,
                           hyperParams['batch_size'], 
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

#%% SAVING SETTINGS
res_dir = input("Create a directory where saving the results:\n")
try:
    os.mkdir('dataset/results/'+res_dir)
except OSError:
    print("DIRECTORY NOT CREATED")
else:
    print("DIRECTORY SUCCESSFULLY CREATED")
res_dir='dataset/results/'+res_dir

f= open(res_dir+'/notes.txt','w+')
note=input("NOTE:\n")
f.write("NOTE:\n")
f.write(note+'\n')     
f.close()
#%% TRAINING
from training_code import training
model,saved_loss,nmae_metr,cos_simil,mre_saved=training(hyperParams,model,data_loaders,optimizer_1,scheduler_1,res_dir)
plt.rcParams['font.family']='DeJavu Serif'
plt.rcParams['font.serif']=['Times New Roman']

#%% NMSE PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),saved_loss[0],label='Train')
ax.plot(range(hyperParams['epochs']),saved_loss[1],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('NMSE')
plt.yscale("log")
plt.show()
plt.savefig(res_dir+'/nmse_log.png')
#%% NMSE PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),saved_loss[0],label='Train')
ax.plot(range(hyperParams['epochs']),saved_loss[1],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('NMSE')
#plt.yscale("log")
plt.show()
plt.savefig(res_dir+'/nmse.png')
#%% COSINE SIMILARITY PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),cos_simil[0],label='Train')
ax.plot(range(hyperParams['epochs']),cos_simil[1],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('COSINE SIMILARITY')
plt.yscale("log")
plt.show()
plt.savefig(res_dir+'/cos_sim_log.png')
#%%
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),cos_simil[0],label='Train')
ax.plot(range(hyperParams['epochs']),cos_simil[1],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('COSINE SIMILARITY')
#plt.yscale("log")
plt.show()
plt.savefig(res_dir+'/cos_sim.png')
#%% NMAE PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),nmae_metr[0],label='Train')
ax.plot(range(hyperParams['epochs']),nmae_metr[1],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('NMAE')
plt.yscale("log")
plt.show()
plt.savefig(res_dir+'/nmae_log.png')
#%% NMAE PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),nmae_metr[0],label='Train')
ax.plot(range(hyperParams['epochs']),nmae_metr[1],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
#ax.set_ylabel('NMAE')
#plt.yscale("log")
plt.show()
plt.savefig(res_dir+'/nmae.png')
#%% MRE PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),mre_saved[0],label='Train')
ax.plot(range(hyperParams['epochs']),mre_saved[1],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('MRE')
plt.yscale("log")
plt.show()
plt.savefig(res_dir+'/mre_log.png')
#%% MRE PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),mre_saved[0],label='Train')
ax.plot(range(hyperParams['epochs']),mre_saved[1],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('MRE')
#plt.yscale("log")
plt.show()
plt.savefig(res_dir+'/mre.png')
#%%
#from results_for_norm import apply_model_on_mesh,predict_on_dataloader
from results_normalize import predict_on_dataloader
#wss_maxm,wss_minm,vrtx_maxm,vrtx_minm=predict_on_dataloader(model,data_loaders)
meshes_path=res_dir
predict_on_dataloader(meshes_path,model,data_loaders)
#%%
plt.close('all')
