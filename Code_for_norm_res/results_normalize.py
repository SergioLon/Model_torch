import torch 
import matplotlib.pyplot as plt
#from split_dataset import split_dataset
import numpy as np
#from model import GCN
import pyvista as pv

from losses import nmse,NMAE,Cos_sim,mre
def relative_error(out,target):
    
    r_err=np.zeros((len(out),1),dtype=float)
    for i in range(len(out)):
        #print("OUT ",out[i])
        #print("TARGET",target[i])
        r_err[i]=(out[i]-target[i])**2
        #print("ERRORE QUADRO",r_err[i])
        r_err[i]=np.sqrt(r_err[i])
        #print("ERRORE RADICE",r_err[i])
        r_err[i]/=np.sqrt(target[i]**2)
        #print("ERRORE NORMALIZZATO",r_err[i])
        # if r_err[i]>10:
        #     print("ERROR BIGGER THAN 1")
        #     print("OUT ",out[i])
        #     print("TARGET",target[i])
            
    return r_err
def denormalize_min_max_wss(point_array,maxm,minm):
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
    return np.expand_dims(new_array,axis=-1)

def denormalize_std_1_wss(point_array,std,mean):
    std=std[0].detach().numpy()
    
    mean=mean.detach().numpy()
    new_point_array=(point_array*std)+mean
    return np.expand_dims(new_point_array,axis=-1)

def denormalize_max_abs_wss(point_array,maxm):
    #maxm=point_array.max()
    #minm=point_array.min()
    # print("OLD MAX: ",maxm)
    # print("OLD MIN: ",minm)
    #print(maxm)
    maxm=maxm[0].detach().numpy()
    #minm=minm[0].detach().numpy()
    
    new_array=(point_array)*maxm
    # print("NEW MAX: ",new_array.max())
    # print("NEW MIN: ",new_array.min())
    return np.expand_dims(new_array,axis=-1)


"""
This function use the model given to make a prediction on a mesh,
'known' is a bool, if True it means that the  original wss values are known, so
the prediction and the original can be compared
"""

    
def predict_on_dataloader(mesh_path,model,data_loaders):
    model.eval()
    f= open(mesh_path+'/errors.txt','w+')
    # note=input("NOTE:\n")
    # f.write("NOTE:\n")
    # f.write(note+'\n')        
    for idx,m in enumerate(data_loaders['val']):
        
        #if idx==0:
            # if m.wss_max[0,0]!=0:
            #     wss_maxm=m.wss_max
            #     wss_minm=m.wss_min
            #     vrtx_maxm=m.vrtx_max
            #     vrtx_minm=m.vrtx_min
            target=m.norm
            out=model(m)
            f.write("MESH : %d \n" %idx)
            f.write("NMSE: %f \n" %nmse(out, target).cpu().detach().numpy())
            f.write("NMAE: %f \n" %np.sum(NMAE(out, target).cpu().detach().numpy()))
            f.write("COSINE SIMILARITY: %f \n" %Cos_sim(out, target).cpu().detach().numpy())
            f.write("MRE: %f \n" %mre(out, target).cpu().detach().numpy())
            print("NMSE: ",nmse(out, target).cpu().detach().numpy())
            print("NMAE: ",np.sum(NMAE(out, target).cpu().detach().numpy()))
            print("COSINE SIMILARITY: ",Cos_sim(out, target).cpu().detach().numpy())
            print("MRE: ",mre(out, target).cpu().detach().numpy())
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
            ax.plot(m.norm[:,0].cpu(),label='Real')
            ax.plot(out[:,0].cpu().detach().numpy(),label='Pred')
            ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('MESH '+str(idx)+' NORM_X ')
            plt.show()
            plt.savefig(mesh_path+'/MESH_' +str(idx)+'_x.png')
            fig, ax = plt.subplots()
            ax.plot(m.norm[:,1].cpu(),label='Real')
            ax.plot(out[:,1].cpu().detach().numpy(),label='Pred')
            ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('MESH '+str(idx)+' NORM_Y')
            plt.show()
            plt.savefig(mesh_path+'/MESH_' +str(idx)+'_y.png')
            fig, ax = plt.subplots()
            ax.plot(m.norm[:,2].cpu(),label='Real')
            ax.plot(out[:,2].cpu().detach().numpy(),label='Pred')
            ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('MESH '+str(idx)+' NORM_Z')
            plt.show()
            plt.savefig(mesh_path+'/MESH_' +str(idx)+'_z.png')
            # fig, ax = plt.subplots()
            # ax.plot(m.wss_coord[:,3].cpu(),label='Real')
            # ax.plot(out[:,3].cpu().detach().numpy(),label='Pred')
            # ax.legend()
            # #ax.title('One Val sample')
            # ax.set_xlabel('Vertx')
            # ax.set_ylabel('WSS_abs normalized')
            # plt.show()
            #creating the predicted mesh
            data=m.to('cpu')
            nodes=data.pos.numpy()
            # WHEN POS NORMALIZED STD=1
            nodes[:,0]=nodes[:,0]*data.std_x.numpy()
            nodes[:,1]=nodes[:,1]*data.std_y.numpy()
            nodes[:,2]=nodes[:,2]*data.std_z.numpy()
            #
            cells=data.face.numpy()
            temp=np.array([3]*cells.shape[1])
            cells=np.c_[temp,cells.T].ravel()
            mesh=pv.PolyData(nodes,cells)
            # print("MAX: ",m.wss_max)
            # print("MIN: ",m.wss_min)
            mesh.point_arrays["norm"]=data.norm.numpy()
            mesh.point_arrays["norm_pred"]=out.cpu().detach().numpy()
          
            out_name=mesh_path+'/'+'mesh'+str(idx)+'.vtp'
            
            
            ##
             # X COMPONENT
            fig, ax = plt.subplots()
            
            ax.plot(np.abs(mesh.point_arrays["norm_pred"][:,0]-mesh.point_arrays["norm"][:,0]))
            #ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('MESH' +str(idx)+' |NORM_X-NORM_X_PRE|')
            plt.show()
            plt.savefig(mesh_path+'/MESH_' +str(idx)+'_dx.png')
            # Y COMPONENT
            fig, ax = plt.subplots()
            
            ax.plot(np.abs(mesh.point_arrays["norm_pred"][:,1]-mesh.point_arrays["norm"][:,1]))
            #ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('MESH' +str(idx)+' |NORM_Y-NORM_Y_PRED|')
            plt.show()
            plt.savefig(mesh_path+'/MESH_' +str(idx)+'_dy.png')
            #Z COMPONENT
            fig, ax = plt.subplots()
            
            ax.plot(np.abs(mesh.point_arrays["norm_pred"][:,2]-mesh.point_arrays["norm"][:,2]))
            #ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('MESH' +str(idx)+' NORM_Z-NORM_Z_PRED|')
            plt.show()
            plt.savefig(mesh_path+'/MESH_' +str(idx)+'_dz.png')
            #ABS
            # fig, ax = plt.subplots()
            
            # ax.plot(np.abs(mesh.point_arrays["wss_abs_pred"]-mesh.point_arrays["wss_abs"]))
            # #ax.legend()
            # #ax.title('One Val sample')
            # ax.set_xlabel('Vertx')
            # ax.set_ylabel('|WSS_ABS-WSS_ABS_PRED|')
            # plt.show()
            #ERRORE PERCHENTUALE
            # X COMPONENT
            fig, ax = plt.subplots()
            #err_x=np.abs((mesh.point_arrays["wss_x_pred"]-mesh.point_arrays["wss_x"])/mesh.point_arrays["wss_x"])*100
            err_x=np.abs((mesh.point_arrays["norm_pred"][:,0]-mesh.point_arrays["norm"][:,0]))/max(abs(mesh.point_arrays["norm"][:,0]))
            #print("err_x.size(0)")
            ax.plot(err_x)
            #ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('MESH' +str(idx)+' Error NORM_X')
            plt.show()
            plt.savefig(mesh_path+'/MESH_' +str(idx)+'_ex.png')
            # Y COMPONENT
            fig, ax = plt.subplots()
            err_y=np.abs((mesh.point_arrays["norm_pred"][:,1]-mesh.point_arrays["norm"][:,1]))/max(abs(mesh.point_arrays["norm"][:,1]))
            ax.plot(err_y)
            #ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('MESH' +str(idx)+' Error NORM_Y')
            plt.show()
            plt.savefig(mesh_path+'/MESH_' +str(idx)+'_ey.png')
            #Z COMPONENT
            fig, ax = plt.subplots()
            err_z=np.abs((mesh.point_arrays["norm_pred"][:,2]-mesh.point_arrays["norm"][:,2]))/max(abs(mesh.point_arrays["norm"][:,2]))
            ax.plot(err_z)
            #ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('MESH' +str(idx)+' Error NORM_Z')
            plt.show()
            plt.savefig(mesh_path+'/MESH_' +str(idx)+'_ez.png')
            # #ABS
            # fig, ax = plt.subplots()
            # err_abs=np.abs((mesh.point_arrays["wss_abs_pred"]-mesh.point_arrays["wss_abs"]))/max(abs(mesh.point_arrays["wss_abs"]))
            # ax.plot(err_abs)
            # #ax.legend()
            # #ax.title('One Val sample')
            # ax.set_xlabel('Vertx')
            # ax.set_ylabel('% Error WSS_ABS')
            # plt.show()
            
            # mesh.point_arrays["err_z"]=err_z
            # mesh.point_arrays["err_x"]=err_x
            # mesh.point_arrays["err_y"]=err_y
            # print(len(err_x))
            # print(len(err_y))
            # print(len(err_z))
            mesh.point_arrays["err"]=np.concatenate([np.expand_dims(err_x,-1),np.expand_dims(err_y,-1),np.expand_dims(err_z,-1)],1)
            print("Mean Error X: ",np.mean(err_x))
            print("Mean Error Y: ",np.mean(err_y))
            print("Mean Error Z: ",np.mean(err_z))
            f.write("Mean Error X: %f \n" %np.mean(err_x))
            f.write("Mean Error Y: %f \n"%np.mean(err_y))
            f.write("Mean Error Z: %f \n" %np.mean(err_z))
            #mesh.point_arrays["err_abs"]=err_abs
            #mesh.point_arrays["err"]=np.concatenate([err_x,err_y,err_z],1)
            ##
            fig, ax = plt.subplots()
            #err_x=np.abs((mesh.point_arrays["wss_x_pred"]-mesh.point_arrays["wss_x"])/mesh.point_arrays["wss_x"])*100
            #r_err_x=np.abs((mesh.point_arrays["wss_pred"][:,0]-mesh.point_arrays["wss"][:,0]))/np.abs(mesh.point_arrays["wss"][:,0])
            r_err_x=relative_error(mesh.point_arrays["norm_pred"][:,0],mesh.point_arrays["norm"][:,0])
            #print("err_x.size(0)")
            ax.plot(r_err_x)
            #ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('MESH' +str(idx)+' Relative Error NORM_X')
            plt.show()
            plt.savefig(mesh_path+'/MESH_' +str(idx)+'_rex.png')
            # Y COMPONENT
            fig, ax = plt.subplots()
            #r_err_y=np.abs((mesh.point_arrays["wss_pred"][:,1]-mesh.point_arrays["wss"][:,1]))/np.abs(mesh.point_arrays["wss"][:,1])
            r_err_y=relative_error(mesh.point_arrays["norm_pred"][:,1],mesh.point_arrays["norm"][:,1])
            ax.plot(r_err_y)
            #ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('MESH' +str(idx)+' Relative Error NORM_Y')
            plt.show()
            plt.savefig(mesh_path+'/MESH_' +str(idx)+'_rey.png')
            #Z COMPONENT
            fig, ax = plt.subplots()
            #r_err_z=np.abs((mesh.point_arrays["wss_pred"][:,2]-mesh.point_arrays["wss"][:,2]))/np.abs(mesh.point_arrays["wss"][:,2])
            r_err_z=relative_error(mesh.point_arrays["norm_pred"][:,2],mesh.point_arrays["norm"][:,2])
            ax.plot(r_err_z)
            #ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('MESH' +str(idx)+' Relative Error NORM_Z')
            plt.show()
            plt.savefig(mesh_path+'/MESH_' +str(idx)+'_rez.png')
            # #ABS
            # fig, ax = plt.subplots()
            # err_abs=np.abs((mesh.point_arrays["wss_abs_pred"]-mesh.point_arrays["wss_abs"]))/max(abs(mesh.point_arrays["wss_abs"]))
            # ax.plot(err_abs)
            # #ax.legend()
            # #ax.title('One Val sample')
            # ax.set_xlabel('Vertx')
            # ax.set_ylabel('% Error WSS_ABS')
            # plt.show()
            
            # mesh.point_arrays["err_z"]=err_z
            # mesh.point_arrays["err_x"]=err_x
            # mesh.point_arrays["err_y"]=err_y
            # print(len(err_x))
            # print(len(err_y))
            # print(len(err_z))
            #print("FIRST VALUE",r_err_x/mesh.n_points)
            mesh.point_arrays["r_err"]=np.concatenate([np.expand_dims(r_err_x,-1),np.expand_dims(r_err_y,-1),np.expand_dims(r_err_z,-1)],1)
            print("Mean Relative Error X: ",np.sum(r_err_x)/mesh.n_points)
            print("Mean Relative Error Y: ",np.sum(r_err_y)/mesh.n_points)
            print("Mean Relative Error Z: ",np.sum(r_err_z)/mesh.n_points)
            f.write("Mean Relative Error X: %f \n" %(np.sum(r_err_x)/mesh.n_points))
            f.write("Mean Relative Error Y: %f \n"%(np.sum(r_err_y)/mesh.n_points))
            f.write("Mean Relative Error Z: %f \n"%(np.sum(r_err_z)/mesh.n_points))
            
            mesh.save(out_name)
    f.close()
            #break
