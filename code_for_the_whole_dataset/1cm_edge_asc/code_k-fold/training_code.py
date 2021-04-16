def denormalize_std_1_wss(point_array,std,mean):
    std=std[0].detach().numpy()
    
    mean=mean.detach().numpy()
    new_point_array=(point_array*std)+mean
    return np.expand_dims(new_point_array,axis=-1)

import numpy as np
import pyvista as pv
from losses import nmse,NMAE,Cos_sim,mre
import torch
def training(hyperParams,model,data_loaders,optimizer,scheduler,mesh_path):
    f_nmse= open(mesh_path+'/nmse.txt','w+')
    f_nmae= open(mesh_path+'/nmae.txt','w+')
    saved_loss=np.zeros((2,hyperParams["epochs"]),dtype=np.dtype('float32'))
    train_metr=np.zeros((3,hyperParams["epochs"]),dtype=np.dtype('float32'))
    val_metr=np.zeros((3,hyperParams["epochs"]),dtype=np.dtype('float32'))
    cosine_simil=np.zeros((2,hyperParams["epochs"]),dtype=np.dtype('float32'))
    mre_saved=np.zeros((2,hyperParams["epochs"]),dtype=np.dtype('float32'))
    print('-'*40+'\nTRAINING  STARTED\n'+'-'*40)
    for epoch in range(hyperParams['epochs']):
        try:
            
            print(f'Epoch: {epoch:3d}')
            train_loss=0.0
            val_loss=0.0
            cos_train_loss=0.0
            cos_val_loss=0.0
            mre_train_loss=0.0
            mre_val_loss=0.0
            train_metric=np.zeros((3,1),dtype=float)
            val_metric=np.zeros((3,1),dtype=float)
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
                    #batch=norm(batch)
                    out = model(batch)  # Perform a single forward pass.
                    #print(ii)
                    on_train=batch.wss_coord
                    loss = nmse(out, on_train)  # Compute the loss solely based on the training nodes.
                    nmae=NMAE(out,on_train)
                    mre_on=mre(out,on_train)
                    cos_loss=Cos_sim(out,on_train)
                    # loss = nmse(out, batch.wss_coord)  # Compute the loss solely based on the training nodes.
                    # nmae=NMAE(out,batch.wss_coord)
                    #print(out.size())
                    #loss_x = nmse(out[:,0], batch.wss[:,0])
                    #loss_abs = nmse(out[:,1], batch.wss[:,3])
                    #loss=loss_x+loss_abs
                    optimizer.zero_grad()
                    if phase == 'train':
                        loss.backward()
                        #cos_loss.backward()
                        #torch.sum(nmae).backward()
                        # update the weights
                        optimizer.step()
                        train_loss+=loss.data
                        train_metric=train_metric+nmae.detach().numpy()
                        cos_train_loss+=cos_loss
                        mre_train_loss+=mre_on
                        if epoch==hyperParams['epochs']-1:
                            
                            data=batch.to('cpu')
                            nodes=data.pos.numpy()
                            nodes[:,0]=nodes[:,0]*data.std_x.numpy()
                            nodes[:,1]=nodes[:,1]*data.std_y.numpy()
                            nodes[:,2]=nodes[:,2]*data.std_z.numpy()
                            #
                            cells=data.face.numpy()
                            temp=np.array([3]*cells.shape[1])
                            cells=np.c_[temp,cells.T].ravel()
                            mesh=pv.PolyData(nodes,cells)
                            wss_x_p=denormalize_std_1_wss(out[:,0].cpu().detach().numpy(),batch.wss_std_x.cpu(),batch.wss_mean_x.cpu())
                            wss_y_p=denormalize_std_1_wss(out[:,1].cpu().detach().numpy(),batch.wss_std_y.cpu(),batch.wss_mean_y.cpu())
                            wss_z_p=denormalize_std_1_wss(out[:,2].cpu().detach().numpy(),batch.wss_std_z.cpu(),batch.wss_mean_z.cpu())
                            mesh.point_arrays["wss_pred"]=np.concatenate([wss_x_p,wss_y_p,wss_z_p],1)
                            wss_x=denormalize_std_1_wss(batch.wss_coord[:,0].cpu().detach().numpy(),batch.wss_std_x.cpu(),batch.wss_mean_x.cpu())
                            wss_y=denormalize_std_1_wss(batch.wss_coord[:,1].cpu().detach().numpy(),batch.wss_std_y.cpu(),batch.wss_mean_y.cpu())
                            wss_z=denormalize_std_1_wss(batch.wss_coord[:,2].cpu().detach().numpy(),batch.wss_std_z.cpu(),batch.wss_mean_z.cpu())
                            mesh.point_arrays["wss"]=np.concatenate([wss_x,wss_y,wss_z],1)
                            out_name=mesh_path+'/'+'mesh_train'+str(ii)+'.vtp'
                            target=batch.wss_coord.cpu().detach().numpy()
                            out=out.cpu().detach().numpy()
                            err_x=np.abs((out[:,0]-target[:,0]))/max(abs(target[:,0]))
                            err_y=np.abs((out[:,1]-target[:,1]))/max(abs(target[:,1]))
                            err_z=np.abs((out[:,2]-target[:,2]))/max(abs(target[:,2]))
                            mesh.point_arrays["err"]=np.concatenate([np.expand_dims(err_x,-1),np.expand_dims(err_y,-1),np.expand_dims(err_z,-1)],1)
                            mesh.save(out_name)
                    else:
                        val_loss+=loss.data
                        val_metric=val_metric+nmae.detach().numpy()
                        cos_val_loss+=cos_loss
                        mre_val_loss+=mre_on
                        #scheduler.step(val_loss)
            scheduler.step(val_loss)
            train_loss /= len(data_loaders['train'])
            val_loss /=len(data_loaders['val'])
            train_metric /= len(data_loaders['train'])
            val_metric /=len(data_loaders['val'])
            cos_train_loss /= len(data_loaders['train'])
            cos_val_loss /=len(data_loaders['val'])
            mre_train_loss /= len(data_loaders['train'])
            mre_val_loss /=len(data_loaders['val'])
            #save the loss
            saved_loss[0][epoch]=train_loss
            saved_loss[1][epoch]=val_loss
            # Cosine similarity
            cosine_simil[0][epoch]=cos_train_loss
            cosine_simil[1][epoch]=cos_val_loss
            # MEAN RELATIVE ERROR
            mre_saved[0][epoch]=mre_train_loss
            mre_saved[1][epoch]=mre_val_loss
            for j in range(3):
                train_metr[j][epoch]=train_metric[j]
                val_metr[j][epoch]=val_metric[j]
            #print loss for each epoch
            f_nmse.write( train_loss+"    "+ val_loss)
            f_nmae.write( np.sum(train_metric)+"    "+np.sum(val_metric))
            print('{} COSINE SIMILARITY: {:.4f}; {} COSINE SIMILARITY: {:.4f}'.format('Train', cos_train_loss,'Val',cos_val_loss))
            print('{} NMSE: {:.4f}; {} NMSE: {:.4f}'.format('Train', train_loss,'Val',val_loss))
            print('{} NMAE: {:.4f}; {} NMAE: {:.4f}'.format('Train', np.sum(train_metric),'Val',np.sum(val_metric)))
            print('{} MRE: {:.4f}; {} MRE: {:.4f}'.format('Train', mre_train_loss,'Val',mre_val_loss))
        except KeyboardInterrupt:
            break
    return model,saved_loss,train_metr,val_metr,cosine_simil,mre_saved
