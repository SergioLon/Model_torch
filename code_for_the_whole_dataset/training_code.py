import numpy as np
from losses import nmse,NMAE

def training(hyperParams,model,data_loaders,optimizer,scheduler):
    saved_loss=np.zeros((2,hyperParams["epochs"]),dtype=np.dtype('float32'))
    train_metr=np.zeros((3,hyperParams["epochs"]),dtype=np.dtype('float32'))
    val_metr=np.zeros((3,hyperParams["epochs"]),dtype=np.dtype('float32'))
    print('-'*40+'\nTRAINING  STARTED\n'+'-'*40)
    for epoch in range(hyperParams['epochs']):
        try:
            
            print(f'Epoch: {epoch:3d}')
            train_loss=0.0
            val_loss=0.0
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
                    
                    loss = nmse(out, batch.wss_coord)  # Compute the loss solely based on the training nodes.
                    nmae=NMAE(out,batch.wss_coord)
                    #print(out.size())
                    #loss_x = nmse(out[:,0], batch.wss[:,0])
                    #loss_abs = nmse(out[:,1], batch.wss[:,3])
                    #loss=loss_x+loss_abs
                    optimizer.zero_grad()
                    if phase == 'train':
                        loss.backward()
                        # update the weights
                        optimizer.step()
                        train_loss+=loss.data
                        train_metric=train_metric+nmae.detach().numpy()
                    else:
                        val_loss+=loss.data
                        val_metric=val_metric+nmae.detach().numpy()
                        scheduler.step(val_loss)
        
            
            train_loss /= len(data_loaders['train'])
            val_loss /=len(data_loaders['val'])
            train_metric /= len(data_loaders['train'])
            val_metric /=len(data_loaders['val'])
            
            #save the loss
            saved_loss[0][epoch]=train_loss
            saved_loss[1][epoch]=val_loss
            for j in range(3):
                train_metr[j][epoch]=train_metric[j]
                val_metr[j][epoch]=val_metric[j]
            #print loss for each epoch
            print('{} NMSE Loss: {:.4f}; {} NMSE Loss: {:.4f}'.format('Train', train_loss,'Val',val_loss))
            print('{} NMAE: {:.4f}; {} NMAE: {:.4f}'.format('Train', np.mean(train_metric),'Val',np.mean(val_metric)))
        except KeyboardInterrupt:
            break
    return model,saved_loss,train_metr,val_metr
