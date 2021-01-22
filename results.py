import torch 
import matplotlib.pyplot as plt
#from split_dataset import split_dataset
import numpy as np
#from model import GCN
import pyvista as pv
from torch_geometric.transforms import FaceToEdge
from torch_geometric.data import Data

def normalize_wss(point_array):
    maxm=point_array.max()
    minm=point_array.min()
    # print("OLD MAX: ",maxm)
    # print("OLD MIN: ",minm)
    new_array=(point_array-minm)/(maxm-minm)
    # print("NEW MAX: ",new_array.max())
    # print("NEW MIN: ",new_array.min())
    return new_array


"""
This function use the model given to make a prediction on a mesh,
'known' is a bool, if True it means that the  original wss values are known, so
the prediction and the original can be compared
"""
def apply_model_on_mesh(file_name,out_name,model,device,vrtx_maxm,vrtx_minm,wss_maxm,wss_minm,known=True):
    model.eval()
    mesh=pv.read(file_name)
    #preprocess the input
    faces=mesh.faces.reshape((-1,4))[:, 1:4].T
    pos=torch.tensor(mesh.points,dtype=torch.float)
    
    faces=torch.LongTensor(faces)
    # mean = ( vrtx_maxm + vrtx_minm ) / 2.
    
    data=Data(
                    pos=pos,
                    face=faces,
            )
    
    f2e=FaceToEdge(remove_faces=(True))
    data=f2e(data)
    data=data.to(device)
    mean = ( vrtx_maxm + vrtx_minm ) / 2.
    data.pos = (data.pos - mean) / ( (vrtx_maxm - vrtx_minm)/2)
    out=model(data)
    #plt.show()
    
    #bring the output in the original range
    
    out=out*(wss_maxm[:,:-1]-wss_minm[:,:-1])+wss_minm[:,:-1]
    out_abs=torch.sqrt(out[:,0]**2+out[:,1]**2+out[:,2]**2).unsqueeze(1)
    #save the outpunt as mesh attributes
    mesh.point_arrays["wss_x_pred"]=out[:,0].cpu().detach().numpy()
    mesh.point_arrays["wss_y_pred"]=out[:,1].cpu().detach().numpy()
    mesh.point_arrays["wss_z_pred"]=out[:,2].cpu().detach().numpy()
    mesh.point_arrays["wss_abs_pred"]=out_abs.cpu().detach().numpy()
    
    #visualize the difference between the predicted and the real one, if known
    if known==True:    
        
        # X COMPONENT
        fig, ax = plt.subplots()
        
        ax.plot(np.abs(mesh.point_arrays["wss_x_pred"]-mesh.point_arrays["wss_x"]))
        #ax.legend()
        #ax.title('One Val sample')
        ax.set_xlabel('Vertx')
        ax.set_ylabel('|WSS_X-WSS_X_PRE|')
        plt.show()
        # Y COMPONENT
        fig, ax = plt.subplots()
        
        ax.plot(np.abs(mesh.point_arrays["wss_y_pred"]-mesh.point_arrays["wss_y"]))
        #ax.legend()
        #ax.title('One Val sample')
        ax.set_xlabel('Vertx')
        ax.set_ylabel('|WSS_Y-WSS_Y_PRED|')
        plt.show()
        #Z COMPONENT
        fig, ax = plt.subplots()
        
        ax.plot(np.abs(mesh.point_arrays["wss_z_pred"]-mesh.point_arrays["wss_z"]))
        #ax.legend()
        #ax.title('One Val sample')
        ax.set_xlabel('Vertx')
        ax.set_ylabel('|WSS_Z-WSS_Z_PRED|')
        plt.show()
        #ABS
        fig, ax = plt.subplots()
        
        ax.plot(np.abs(mesh.point_arrays["wss_abs_pred"]-mesh.point_arrays["wss_abs"]))
        #ax.legend()
        #ax.title('One Val sample')
        ax.set_xlabel('Vertx')
        ax.set_ylabel('|WSS_ABS-WSS_ABS_PRED|')
        plt.show()
        ##
        value = input("Normalize the outputs in range 0-1? [y/n]\n")
        if value=='y':    
            mesh.point_arrays["wss_abs_pred_norm"]=normalize_wss(mesh.point_arrays["wss_abs_pred"])
            mesh.point_arrays["wss_abs_norm"]=normalize_wss(mesh.point_arrays["wss_abs"])
            mesh.point_arrays["wss_x_pred_norm"]=normalize_wss(mesh.point_arrays["wss_x_pred"])
            mesh.point_arrays["wss_x_norm"]=normalize_wss(mesh.point_arrays["wss_x"])
            mesh.point_arrays["wss_y_pred_norm"]=normalize_wss(mesh.point_arrays["wss_y_pred"])
            mesh.point_arrays["wss_y_norm"]=normalize_wss(mesh.point_arrays["wss_y"])
            mesh.point_arrays["wss_z_pred_norm"]=normalize_wss(mesh.point_arrays["wss_z_pred"])
            mesh.point_arrays["wss_z_norm"]=normalize_wss(mesh.point_arrays["wss_z"])
    ##
    mesh.save(out_name)
    
def predict_on_dataloader(model,data_loaders):
    model.eval()
    for idx,m in enumerate(data_loaders['train']):
        if m.wss_max[0,0]!=0:
            wss_maxm=m.wss_max
            wss_minm=m.wss_min
            vrtx_maxm=m.vrtx_max
            vrtx_minm=m.vrtx_min
            
    for idx,m in enumerate(data_loaders['val']):
        if idx==0:
            if m.wss_max[0,0]!=0:
                wss_maxm=m.wss_max
                wss_minm=m.wss_min
                vrtx_maxm=m.vrtx_max
                vrtx_minm=m.vrtx_min
            
            out=model(m)
            # a=torch.sqrt(out[:,0]**2+out[:,1]**2+out[:,2]**2).unsqueeze(1)
            # fig, ax = plt.subplots()
            # ax.plot(m.wss[:,3].cpu(),label='Real')
            # ax.plot(a.cpu().detach().numpy(),label='Pred')
            # ax.legend()
            # #ax.title('One Val sample')
            # ax.set_xlabel('Vertx')
            # ax.set_ylabel('WSS_ABS normalized')
            plt.show()
            fig, ax = plt.subplots()
            ax.plot(m.wss[:,0].cpu(),label='Real')
            ax.plot(out[:,0].cpu().detach().numpy(),label='Pred')
            ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('WSS_X normalized')
            plt.show()
            fig, ax = plt.subplots()
            ax.plot(m.wss[:,1].cpu(),label='Real')
            ax.plot(out[:,1].cpu().detach().numpy(),label='Pred')
            ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('WSS_Y normalized')
            plt.show()
            fig, ax = plt.subplots()
            ax.plot(m.wss[:,2].cpu(),label='Real')
            ax.plot(out[:,2].cpu().detach().numpy(),label='Pred')
            ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('WSS_Z normalized')
            plt.show()
            break
    return wss_maxm,wss_minm,vrtx_maxm,vrtx_minm