import torch
#import numpy as np
def nmse(output, target):
    if len(output.size())>1:
        loss=torch.zeros((output.size(1),1))
        #N=output.size(0)
        for i in range (output.size(1)):
            
            loss[i] = torch.sum((output[:,i] - target[:,i])**2) / torch.sum(target[:,i]**2)
            #mse
            #loss[i] = torch.sum((output[:,i] - target[:,i])**2)
            #print(loss)
        return torch.sum(loss)
    else:
        loss = torch.sum((output - target)**2) / torch.sum(target**2)
        return loss

def nl1(output, target):
    loss = torch.sum(torch.abs(output - target)) / torch.sum(torch.abs(target))
    return loss

def NMAE(output,target):
    if len(output.size())>1:
        nmae=torch.zeros((output.size(1),1))
        N=output.size(0)
        for i in range (output.size(1)):
            nmae[i]=torch.sum((output[:,i] - target[:,i]).abs())/(N*((target[:,i]).abs().max())-(target[:,i].abs().min()))
        return torch.sum(nmae)
    else:
        nmae=torch.sum((output - target).abs())/(N*((target).abs().max())-(target.abs().min()))
        return nmae

from torch.nn import CosineSimilarity 

def Cos_sim(output,target):
    cos=CosineSimilarity(dim=0)
    if len(output.size())>1:
        cos_sim=torch.zeros((output.size(1),1))
        
        for i in range (output.size(1)):
            cos_sim[i]=1-cos(output[:,i], target[:,i])
        #print("CALCOLO COS: ", cos_sim)
        return torch.sum(cos_sim)
    else:
        cos_sim=1-cos(output, target)
        return cos_sim


def mre(output,target):
    N=output.size(0)
    if len(output.size())>1:
        mre=torch.zeros((output.size(1),1))
        N=output.size(0)
        for i in range (output.size(1)):
            #print("DIM OUT",output[:,i].size())
            #print("DIM TAR",target[:,i].size())
            mre[i]=torch.sum(torch.sqrt((output[:,i] - target[:,i])**2)/(torch.sqrt(target[:,i]**2)))
        return torch.sum(mre/N)
    else:
        mre=torch.sum(torch.sqrt((output - target)**2)/(torch.sqrt(target**2)))
        return mre/N