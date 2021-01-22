import torch 
import matplotlib.pyplot as plt
from split_dataset import split_dataset
import numpy as np
from model import GCN
import pyvista as pv
from torch_geometric.transforms import FaceToEdge,RandomRotate,Compose
from torch_geometric.data import Data,DataLoader,InMemoryDataset
from losses import nmse
#%% SETTING PARAMS
meshes_path='../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device='cpu'
#dataset=my_train_fn()
#loader=DataLoader(dataset,batch_size=1)
#dataset=MyOwnDataset(root='../Meshes_vtp',)
hyperParams={
    "lr": 0.001,
    "epochs": 10000,
    "batch_size":1,
    "val_split":0.1,
    "loss":torch.nn.MSELoss(),
    "weight_decay":5e-6
    
    }

#WEIGHT INITIALIZATION 
def weights_init_uniform_rule(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('FeaStConv') != -1:
            # get the number of the inputs
            n = m.in_channels
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            #m.bias.data.fill_(0)
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
    # create a new model with these weights
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



optimizer = torch.optim.Adamax(model.parameters(), 
                             lr=hyperParams['lr'], 
                             weight_decay=hyperParams['weight_decay']
                             )
criterion = hyperParams['loss']

model=model.to(device)  
saved_loss=np.zeros((2,hyperParams["epochs"]),dtype=np.dtype('float32'))


#%% TRAINING
print('-'*40+'\nTRAINING  STARTED\n'+'-'*40)
for epoch in range(hyperParams['epochs']):
    try:
        
        print(f'Epoch: {epoch:3d}')
        train_loss=0.0
        val_loss=0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
                
            
            #running_loss = 0.0
            for ii,batch in enumerate(data_loaders[phase]):
                # if ii==0 and phase=='val':
                #     print(batch.vrtx_max)
                #     #print(batch.wss_max)
                #     maxm=batch.wss_max
                #     minm=batch.wss_min
                    
                #print(batch.pos.size(0))
                out = model(batch)  # Perform a single forward pass.
          
    
                loss = nmse(out, batch.wss)  # Compute the loss solely based on the training nodes.
                #loss_x = nmse(out[:,0], batch.wss[:,0])
                #loss_abs = nmse(out[:,1], batch.wss[:,3])
                #loss=loss_x+loss_abs
                optimizer.zero_grad()
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()
                    train_loss+=loss.data
                else:
                    val_loss+=loss.data
                
    
        
        train_loss /= len(data_loaders['train'])
        val_loss /=len(data_loaders['val'])
        #save the loss
        saved_loss[0][epoch]=train_loss
        saved_loss[1][epoch]=val_loss
        #print loss for each epoch
        print('{} Loss: {:.4f}; {} Loss: {:.4f}'.format('Train', train_loss,'Val',val_loss))
    except KeyboardInterrupt:
        break
#%% LOSS PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),saved_loss[0],label='Train')
ax.plot(range(hyperParams['epochs']),saved_loss[1],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
plt.show()

#%%
from results import apply_model_on_mesh,predict_on_dataloader

wss_maxm,wss_minm,vrtx_maxm,vrtx_minm=predict_on_dataloader(model,data_loaders)

#%%
file_name='../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated/Clipped/aorta_0_dec_cl2.vtp'
# out_name='../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated/Predicted/aorta_0_pred.vtp'
value = input("Do you want to make a prediction on "+file_name+"? [y/n]\n")
if value=='y':
    value = input("Choose a name for the prediction file:\n")
    
    out_name='../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated/Predicted/'+value+'.vtp'
    apply_model_on_mesh(file_name,out_name,model, device, vrtx_maxm, vrtx_minm, wss_maxm, wss_minm)

#%% RESULTS PLOT
# model.eval()
# for idx,m in enumerate(data_loaders['val']):
#     if idx==0:
#         wss_maxm=m.wss_max
#         wss_minm=m.wss_min
        
#         vrtx_maxm=m.vrtx_max
#         vrtx_minm=m.vrtx_min
        
#         out=model(m)
#         # a=torch.sqrt(out[:,0]**2+out[:,1]**2+out[:,2]**2).unsqueeze(1)
#         # fig, ax = plt.subplots()
#         # ax.plot(m.wss[:,3].cpu(),label='Real')
#         # ax.plot(a.cpu().detach().numpy(),label='Pred')
#         # ax.legend()
#         # #ax.title('One Val sample')
#         # ax.set_xlabel('Vertx')
#         # ax.set_ylabel('WSS_ABS normalized')
#         plt.show()
#         fig, ax = plt.subplots()
#         ax.plot(m.wss[:,0].cpu(),label='Real')
#         ax.plot(out[:,0].cpu().detach().numpy(),label='Pred')
#         ax.legend()
#         #ax.title('One Val sample')
#         ax.set_xlabel('Vertx')
#         ax.set_ylabel('WSS_X normalized')
#         plt.show()
#         fig, ax = plt.subplots()
#         ax.plot(m.wss[:,1].cpu(),label='Real')
#         ax.plot(out[:,1].cpu().detach().numpy(),label='Pred')
#         ax.legend()
#         #ax.title('One Val sample')
#         ax.set_xlabel('Vertx')
#         ax.set_ylabel('WSS_Y normalized')
#         plt.show()
#         fig, ax = plt.subplots()
#         ax.plot(m.wss[:,2].cpu(),label='Real')
#         ax.plot(out[:,2].cpu().detach().numpy(),label='Pred')
#         ax.legend()
#         #ax.title('One Val sample')
#         ax.set_xlabel('Vertx')
#         ax.set_ylabel('WSS_Z normalized')
#         plt.show()
#         break

#%% Predict on a mesh

# model.eval()
# a=pv.read('../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated/aorta_0_dec.vtp')
# #preprocess the input
# faces=a.faces.reshape((-1,4))[:, 1:4].T
# pos=torch.tensor(a.points,dtype=torch.float)

# faces=torch.LongTensor(faces)
# # mean = ( vrtx_maxm + vrtx_minm ) / 2.

# data=Data(
#                 pos=pos,
#                 face=faces,
#         )

# f2e=FaceToEdge(remove_faces=(True))
# data=f2e(data)
# data=data.to(device)
# mean = ( vrtx_maxm + vrtx_minm ) / 2.
# data.pos = (data.pos - mean) / ( (vrtx_maxm - vrtx_minm)/2)
# out=model(data)
# #plt.show()
# #bring the output in the original range

# out=out*(wss_maxm[:,:-1]-wss_minm[:,:-1])+wss_minm[:,:-1]
# #save the outpunt as mesh attributes
# a.point_arrays["wss_x_pred"]=out[:,0].cpu().detach().numpy()
# a.point_arrays["wss_y_pred"]=out[:,1].cpu().detach().numpy()
# a.point_arrays["wss_z_pred"]=out[:,2].cpu().detach().numpy()
# a.point_arrays["wss_abs_pred"]=torch.sqrt(out[:,0]**2+out[:,1]**2+out[:,2]**2).unsqueeze(1).cpu().detach().numpy()
# #visualize the difference between the real and the predicted one
# # X COMPONENT
# fig, ax = plt.subplots()

# ax.plot(np.abs(a.point_arrays["wss_x_pred"]-a.point_arrays["wss_x"]))
# ax.legend()
# #ax.title('One Val sample')
# ax.set_xlabel('Vertx')
# ax.set_ylabel('|WSS_X-WSS_X_PRE|')
# plt.show()
# # Y COMPONENT
# fig, ax = plt.subplots()

# ax.plot(np.abs(a.point_arrays["wss_y_pred"]-a.point_arrays["wss_y"]))
# ax.legend()
# #ax.title('One Val sample')
# ax.set_xlabel('Vertx')
# ax.set_ylabel('|WSS_Y-WSS_Y_PRED|')
# plt.show()
# #Z COMPONENT
# fig, ax = plt.subplots()

# ax.plot(np.abs(a.point_arrays["wss_z_pred"]-a.point_arrays["wss_z"]))
# ax.legend()
# #ax.title('One Val sample')
# ax.set_xlabel('Vertx')
# ax.set_ylabel('|WSS_Z-WSS_Z_PRED|')
# plt.show()
# #ABS
# fig, ax = plt.subplots()

# ax.plot(np.abs(a.point_arrays["wss_abs_pred"]-a.point_arrays["wss_abs"]))
# ax.legend()
# #ax.title('One Val sample')
# ax.set_xlabel('Vertx')
# ax.set_ylabel('|WSS_ABS-WSS_ABS_PRED|')
# plt.show()
# a.save('../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated/Predicted/aorta_0_pred.vtp')
    