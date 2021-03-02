import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import glob
import pyvista as pv
import numpy as np
#import trimesh as tr

#import matplotlib.pyplot as plt
import torch
from MyOwnDataset_normalize_train import MyOwnDataset_normalize_train
from torch_geometric.transforms import Compose,KNNGraph,FaceToEdge,GenerateMeshNormals
from new_random_rotate import RandomRotate
from torch_geometric.data import Data,DataLoader,InMemoryDataset

class Normilize_WSS(object):
    def __call__(self,data):
        
        maxm = data.wss_coord.max(dim=-2).values
        minm = data.wss_coord.min(dim=-2).values
        
        data.wss_min[:]=minm.min()
        data.wss_max[:]=minm.max()
        print("OLD WSS MAX: ",maxm)
        print("OLD WSS MIN: ",minm)
        # print("OLD POS_X MAX: ",data.pos_x.max())
        # print("OLD POS_X MIN: ",data.pos_x.min())
        #mean = ( maxm.max() + minm.min() ) / 2.
        ## NORMALIZE [-1,1]
        maxm_abs = data.wss_coord.abs().max(dim=-2).values
        
        data.wss_max_abs[:]=maxm_abs.max()
        #data.wss_coord = (data.wss_coord)/maxm_abs.max()
        ## NORMALIZE [0,1]
        #data.wss_coord = (data.wss_coord - minm.min()) / ( (maxm.max() - minm.min()))
        
        ## NORMALIZE STD=1
        ##LOAD TRAINING MEAN AND STD
        dataset=MyOwnDataset_normalize_train(root='dataset/training',)
        train_loader = DataLoader(dataset, batch_size=1)
        for batch in train_loader:
            
            std_x=batch.wss_std_x
            std_y=batch.wss_std_y
            std_z=batch.wss_std_z
            wss_mean_x = batch.wss_mean_x
            wss_mean_y = batch.wss_mean_y
            wss_mean_z = batch.wss_mean_z
            break
        data.wss_std_x[:]=std_x
        data.wss_std_y[:]=std_y
        data.wss_std_z[:]=std_z
        
        data.wss_mean_x[:]=wss_mean_x
        data.wss_mean_y[:]=wss_mean_y
        data.wss_mean_z[:]=wss_mean_z
        print("WSS MEAN: ",wss_mean_x,wss_mean_y,wss_mean_z)
        print("WSS STD X: ",std_x)
        print("WSS STD Y: ",std_y)
        print("WSS STD Z: ",std_z)
        data.wss_coord[:,0] = (data.wss_coord[:,0]-wss_mean_x) /std_x
        data.wss_coord[:,1] = (data.wss_coord[:,1]-wss_mean_y) /std_y
        data.wss_coord[:,2] = (data.wss_coord[:,2]-wss_mean_z) /std_z
        maxm = data.wss_coord.max(dim=-2).values
        minm = data.wss_coord.min(dim=-2).values
        wss_mean = ( maxm + minm ) / 2.
        
        std_x=torch.std(data.wss_coord[:,0])
        std_y=torch.std(data.wss_coord[:,1])
        std_z=torch.std(data.wss_coord[:,2])
        # print("NEW MEAN X: ",mean_x)
        # print("NEW MEAN Y: ",mean_y)
        # print("NEW MEAN Z: ",mean_z)
        print("NEW WSS MEAN: ",wss_mean)
        print("NEW WSS STD X: ",std_x)
        print("NEW WSS STD Y: ",std_y)
        print("NEW WSS STD Z: ",std_z)
        #data.wss_coord = (data.wss_coord - mean) / ( (maxm.max() - minm.min())/2.)
        #data.pos_x=((data.pos_x - minm[0]) / ( (maxm[0] - minm[0])))
        #data.pos_y=torch.tensor(np.expand_dims(data.pos[:,1].detach().numpy(),axis=-1))
        #data.pos_z=torch.tensor(np.expand_dims(data.pos[:,2].detach().numpy(),axis=-1))
        #print(data.pos_x.size())
        print("NEW WSS MAX: ",data.wss_coord.max(dim=-2).values)
        print("NEW WSS MIN: ",data.wss_coord.min(dim=-2).values)
        return data
class Normilize_Norm(object):
    def __call__(self,data):
        
        maxm = data.norm.max(dim=-2).values
        minm = data.norm.min(dim=-2).values
        
        #data.wss_min[:]=minm.min()
        print("OLD NORM MAX: ",maxm)
        print("OLD NORM MIN: ",minm)
        # print("OLD POS_X MAX: ",data.pos_x.max())
        # print("OLD POS_X MIN: ",data.pos_x.min())
        #mean = ( maxm.max() + minm.min() ) / 2.
        #maxm_abs = data.norm.abs().max(dim=-2).values
        data.norm_max[:]=maxm.max()
        data.norm_min[:]=minm.min()
        #data.wss_coord = (data.wss_coord)/maxm_abs.max()
        data.norm = (data.norm - minm.min()) / ( (maxm.max() - minm.min()))
        #data.wss_coord = (data.wss_coord - mean) / ( (maxm.max() - minm.min())/2.)
        #data.pos_x=((data.pos_x - minm[0]) / ( (maxm[0] - minm[0])))
        #data.pos_y=torch.tensor(np.expand_dims(data.pos[:,1].detach().numpy(),axis=-1))
        #data.pos_z=torch.tensor(np.expand_dims(data.pos[:,2].detach().numpy(),axis=-1))
        #print(data.pos_x.size())
        print("NEW NORM MAX: ",data.norm.max(dim=-2).values)
        print("NEW NORM MIN: ",data.norm.min(dim=-2).values)
        return data
    
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class Normalize_vertx(object):
    def __call__(self,data):
        maxm = data.pos.max(dim=-2).values
        minm = data.pos.min(dim=-2).values
        

        #data.vrtx_max[0,:]=maxm[:]
        #data.vrtx_min[0,:]=minm[:]
        print("OLD VERTX MAX: ",maxm)
        print("OLD VERTX MIN: ",minm)
        # print("OLD POS_X MAX: ",data.pos_x.max())
        # print("OLD POS_X MIN: ",data.pos_x.min())
        
        #NORMALIZE [-1,1]
        maxm_abs = data.pos.abs().max(dim=-2).values
        #data.pos =data.pos/maxm_abs.max()
        
        #NORMALIZE [0,1]
        #data.pos=(data.pos- minm.min())/ (maxm.max() - minm.min())
        
        ##LOAD TRAINING MEAN AND STD
        dataset=MyOwnDataset_normalize_train(root='dataset/training',)
        train_loader = DataLoader(dataset, batch_size=1)
        for batch in train_loader:
            
            std_x=batch.std_x
            std_y=batch.std_y
            std_z=batch.std_z
            mean_x = batch.mean_x
            mean_y = batch.mean_y
            mean_z = batch.mean_z
            break
        
        data.mean_x[:]=mean_x
        data.mean_y[:]=mean_y
        data.mean_z[:]=mean_z
        
        data.std_x[:]=std_x
        data.std_y[:]=std_y
        data.std_z[:]=std_z
        # print("MEAN X: ",mean_x)
        # print("MEAN Y: ",mean_y)
        # print("MEAN Z: ",mean_z)
        print("MEAN: ",mean_x,mean_y,mean_z)
        print("STD X: ",std_x)
        print("STD Y: ",std_y)
        print("STD Z: ",std_z)
        # NORMALIZE MEAN=0 STD=1
        data.pos[:,0] = (data.pos[:,0]-mean_x) /std_x
        data.pos[:,1] = (data.pos[:,1]-mean_y) /std_y
        data.pos[:,2] = (data.pos[:,2]-mean_z) /std_z
        ##
        #data.pos = (data.pos - mean) / ( (maxm - minm)/2)
        #data.pos =data.pos/maxm.max()
        #data.pos=(data.pos- minm.min())/ (maxm.max() - minm.min())
        #data.pos_x=((data.pos_x - minm[0]) / ( (maxm[0] - minm[0])))
        #data.pos_y=torch.tensor(np.expand_dims(data.pos[:,1].detach().numpy(),axis=-1))
        #data.pos_z=torch.tensor(np.expand_dims(data.pos[:,2].detach().numpy(),axis=-1))
        #print(data.pos_x.size())
        # mean_x = torch.mean(data.pos[:,0])
        # mean_y = torch.mean(data.pos[:,1])
        # mean_z = torch.mean(data.pos[:,2])
        maxm = data.pos.max(dim=-2).values
        minm = data.pos.min(dim=-2).values
        mean = ( maxm + minm ) / 2.
        
        std_x=torch.std(data.pos[:,0])
        std_y=torch.std(data.pos[:,1])
        std_z=torch.std(data.pos[:,2])
        # print("NEW MEAN X: ",mean_x)
        # print("NEW MEAN Y: ",mean_y)
        # print("NEW MEAN Z: ",mean_z)
        print("NEW MEAN: ",mean)
        print("NEW STD X: ",std_x)
        print("NEW STD Y: ",std_y)
        print("NEW STD Z: ",std_z)
        print("NEW VRTX MAX: ",data.pos.max(dim=-2).values)
        print("NEW VRTX MIN: ",data.pos.min(dim=-2).values)
        
        # print("NEW POS_X MAX: ",data.pos_x.max())
        # print("NEW POS_X MIN: ",data.pos_x.min())
        return data
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
    

#p_trans= [Normalize_vertx(),Normilize_Norm(),]
p_trans= [Normalize_vertx(),Normilize_WSS()]
pre_trans=Compose(p_trans)

trans=[
       RandomRotate(90,axis=0),
       RandomRotate(90,axis=2),
       RandomRotate(90,axis=1),
       ]

pos_trans=Compose(trans)
pos_trans_0=RandomRotate(90,axis=0)
pos_trans_1=RandomRotate(90,axis=1)
pos_trans_2=RandomRotate(90,axis=2)

class MyOwnDataset_normalize_val(InMemoryDataset):
     def __init__(self, 
                  root, 
                  #transform=pos_trans,
                  transform=None,
                  pre_transform=pre_trans):
                  #pre_transform=None):    
        super(MyOwnDataset_normalize_val, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        #self.transform=transform
        self.pre_transform=pre_transform

     @property
     def raw_file_names(self):
         file_name=glob.glob(os.path.join(self.root,'*.vtp'))
         return file_name

     @property
     def processed_file_names(self):
        return ['data.pt']


     def process(self):
        # Read data into huge `Data` list.
        data_list = []
        f2e=FaceToEdge(remove_faces=(False))
        norm_calculate=GenerateMeshNormals()
        knn_g=KNNGraph(k=12)
        for ii,name in enumerate(self.raw_file_names):
            # print(name)
            mesh=pv.read(name)
            
            faces=mesh.faces.reshape((-1,4))[:, 1:4].T
            pos=torch.tensor(mesh.points,dtype=torch.float)
            vrtx_max=pos.max(dim=-2).values
            vrtx_min=pos.min(dim=-2).values
            pos=pos-((vrtx_max+vrtx_min)/2.)
            
            faces=torch.LongTensor(faces)
            #wss=torch.tensor(np.expand_dims(mesh.point_arrays["WSS magnitude"],axis=-1),dtype=torch.float)
            # wss_x=torch.tensor(np.expand_dims(mesh.point_arrays["wss_x"],axis=-1),dtype=torch.float)
            # wss_y=torch.tensor(np.expand_dims(mesh.point_arrays["wss_y"],axis=-1),dtype=torch.float)
            # wss_z=torch.tensor(np.expand_dims(mesh.point_arrays["wss_z"],axis=-1),dtype=torch.float)
            #wss=mesh.point_arrays["wss"]
            #print(wss.size())
            #wss_abs=torch.tensor(np.expand_dims(np.sqrt(wss[:,0]**2+wss[:,1]**2+wss[:,2]**2),axis=-1),dtype=torch.float)
            #wss=torch.tensor(np.expand_dims(mesh.point_arrays["wss"][:,0],axis=-1),dtype=torch.float)
            #wss=torch.tensor(mesh.point_arrays["wss"],dtype=torch.float)
            #print(wss.size())
            
            #wss_coord=torch.cat([wss,wss_abs],dim=1)
            
            wss_coord=torch.tensor(mesh.point_arrays["wss"],dtype=torch.float)
            norm=torch.tensor(mesh.point_arrays["norm"],dtype=torch.float)
            # wss_max=torch.zeros((1,4))
            # wss_min=torch.zeros((1,4))
            # vrtx_max=torch.zeros((1,3))
            # vrtx_min=torch.zeros((1,3))
            #print(wss.size(0))
            ##
            # prova=np.zeros((pos.size(0),1))
            # prova[:int(pos.size(0)/2)]=0.5
            
            # prova=torch.tensor(prova)
            
            ##
            data=Data(
                pos=pos,
                #pos_x=pos_x,
                # pos_y=None,
                # pos_z=None,
                face=faces,
                #wss=wss,
                # wss_x=wss_x,
                # wss_y=wss_y,
                # wss_z=wss_z,
                wss_coord=wss_coord,
                #wss_abs=wss_abs,
                wss_max_abs=0.,
                wss_max=0.,
                wss_min=0.,
                wss_std_x=0.,
                wss_std_y=0.,
                wss_std_z=0.,
                wss_mean_x=0.,
                wss_mean_y=0.,
                wss_mean_z=0.,
                norm_max=0.,
                norm_min=0.,
                norm=norm,
                std_x=0.,
                std_y=0.,
                std_z=0.,
                mean_x=0.,
                mean_y=0.,
                mean_z=0.,
                # vrtx_max=vrtx_max,
                # vrtx_min=vrtx_min,
                # wss_abs=wss_abs,
                #prova=prova,
                )
            #data=knn_g(data)
            data=f2e(data)
            #data=m.to('cpu')
            # nodes=data.pos.numpy()
            # #
            # cells=data.face.numpy()
            # temp=np.array([3]*cells.shape[1])
            # cells=np.c_[temp,cells.T].ravel()
            # mesh=pv.PolyData(nodes,cells)
            # mesh.point_arrays['norm']=data.norm.numpy()
            # mesh.point_arrays['wss']=data.wss_coord.numpy()
            # mesh.save(self.root+'/rotated_001/aorta_'+str(ii)+'.vtp')
            
            # data_aug_1=pos_trans(data)
            # data_aug_2=pos_trans(data)
            # nodes_a=data_aug_1.pos.numpy()
            # #
            # cells_a=data_aug_1.face.numpy()
            # temp_a=np.array([3]*cells_a.shape[1])
            # cells_a=np.c_[temp_a,cells_a.T].ravel()
            # mesh_a=pv.PolyData(nodes_a,cells_a)
            
            # mesh_a.point_arrays['norm']=data_aug_1.norm.numpy()
            # mesh_a.point_arrays['wss']=data_aug_1.wss_coord.numpy()
            # mesh_a.save(self.root+'/rotated_001/rot_aorta_'+str(ii)+'.vtp')
            #data=norm(data)
            #data_aug=norm_calculate(data_aug)
            #print(data)
            data_list.append(data)
            
            # data_list.append(data_aug_1)
            # data_list.append(data_aug_2)
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            #data_list = [self.pre_transform(data) for data in data_list]
            data, slices = self.collate(data_list)
            data=self.pre_transform(data)
        else:
            data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])


#dataset=MyOwnDataset_normalize_val(root='dataset/val',)
