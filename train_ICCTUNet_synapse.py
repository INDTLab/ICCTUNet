from ICCTUNet_synapse import ICCTUNet
from medpy import metric
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import random
import time
from PIL import Image
import h5py
from functools import partial
from  swin_transformer import PatchMerging,SwinTransformerBlock,window_partition,Mlp,window_reverse
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import argparse
import os.path as osp
parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id', type=int, default=0,help="id of cuda device,default:0")
parser.add_argument('--num_epochs', type=int, default=450,help="training epochs")
parser.add_argument('--batch_size', type=int, default=4,help="train batch_size")
parser.add_argument('--drop_path', type=float, default=0.0,help="drop path rate")
parser.add_argument('--lr', type=float, default=5e-4,help="learning rate")
parser.add_argument('--train_path', type=str, default='./data/Synapse/train_npz',help="path to train set")
config = parser.parse_args()

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

snapshot_path = f'./ICCTUNet_batch{config.batch_size}_{config.num_epochs}epoch_weights/'
log_path = f'./ICCTUNet_batch{config.batch_size}_{config.num_epochs}epoch_log/'

if not osp.exists(snapshot_path):
    os.mkdir(snapshot_path)

if not osp.exists(log_path):
    os.mkdir(log_path)



device = torch.device(f'cuda:{config.cuda_id}')



def data_aug(image,label,re_size=None):
    if re_size:
        image = TF.resize(image,(re_size,re_size))
        label = TF.resize(label,(re_size,re_size),interpolation=TF.InterpolationMode.NEAREST)
    if random.random() > 0.5:
        angle = random.randint(-30, 30)
        image = TF.rotate(image,angle)
        label = TF.rotate(label,angle)
    if random.random() > 0.5:
        image = TF.hflip(image)
        label = TF.hflip(label)
    if random.random() > 0.5:
        image = TF.vflip(image)
        label = TF.vflip(label)
    return image,label

def val_data_aug(image,label,re_size=None):
    if re_size:
        image = TF.resize(image,(re_size,re_size))
        label = TF.resize(label,(re_size,re_size),interpolation=TF.InterpolationMode.NEAREST)
    return image,label

def get_file_lst(file_path):
    
    file_list = os.listdir(file_path)
    num_train = len(file_list)
    path_lst = []
    for f in file_list:
        f_path = os.path.join(file_path,f)
        path_lst.append(f_path)
    train_lst = path_lst
    return train_lst

train_lst = get_file_lst(config.train_path)

class SynapseDataset_train(Dataset):
    def __init__(self,train_lst,image_size):
        self.path_lst =train_lst
        self.image_size = image_size
        print(f"got {len(self.path_lst)} images,{len(self.path_lst)} masks")
    def __getitem__(self,index):
        data = np.load(self.path_lst[index])
        img = data['image']
        mask_ = data['label']
        
        image = TF.to_tensor(img).float().contiguous()
        mask = TF.to_tensor(mask_).long().contiguous()
        
        image,mask = data_aug(image,mask,self.image_size)
        
        return {
            'image': image,
            'mask': mask
        }
    def __len__(self):
        return len(self.path_lst)
train_set = SynapseDataset_train(train_lst,image_size=224)

train_bs = config.batch_size

train_loader = DataLoader(train_set,batch_size = train_bs,shuffle=True,num_workers=8,pin_memory=True)

num_epochs = config.num_epochs

lr = config.lr
weight_decay = 0.05

ICCTUNet = ICCTUNet().to(device)

optimizer = torch.optim.AdamW(ICCTUNet.parameters(),lr = lr,weight_decay=weight_decay)
ce_criterion = nn.CrossEntropyLoss()

ce_loss_lst = []
trans_ce_loss_lst = []
fuse_ce_loss_lst = []

dice_loss_lst = []
mse_loss_lst = []

max_iters = num_epochs * len(train_loader)
dice_lst = []
trans_dice_lst = []
fuse_dice_lst = []

f = open(log_path+'save_log.txt','w')
def train(net,num_epochs):
    num_iter = 0
  
    for epoch in range(num_epochs):
        net.train()
        tic = time.time()
        total_num = 0
        total_ce_loss = 0
        total_trans_ce_loss = 0
        total_fuse_ce_loss = 0
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            image = data['image'].to(device)
            label = data['mask'].squeeze(1).to(device)
            pred,swin_pred,fuse_pred = net(image)
            loss_ce = ce_criterion(pred,label)
            trans_loss_ce = ce_criterion(swin_pred,label)
            fuse_loss_ce = ce_criterion(fuse_pred,label)
            total_ce_loss += loss_ce.item()*len(label)
            total_trans_ce_loss += trans_loss_ce.item()*len(label)
            total_fuse_ce_loss += fuse_loss_ce.item()*len(label)

            loss = loss_ce + trans_loss_ce + fuse_loss_ce# + loss_mse
            total_loss += loss.item() * len(data)
            total_num += len(label)
            

            loss.backward()
            optimizer.step()
            
            lr_ = lr * (1-num_iter/max_iters) ** 0.9
            num_iter += 1
            for param in optimizer.param_groups:
                param['lr'] = lr_            
        toc = time.time()
        print(f"epoch:{epoch+1}/{num_epochs}, ce loss{total_ce_loss/total_num:.5f} trans ce loss{total_trans_ce_loss/total_num:.5f}  fuse ce loss{total_fuse_ce_loss/total_num:.5f} time:{toc-tic}")
        
        ce_loss_lst.append(total_ce_loss/total_num)
        trans_ce_loss_lst.append(total_trans_ce_loss/total_num)
        fuse_ce_loss_lst.append(total_fuse_ce_loss/total_num)
        
        np.save(log_path+"ce_loss.npy",np.array(ce_loss_lst))
        np.save(log_path+"trans_ce_loss.npy",np.array(trans_ce_loss_lst))
        np.save(log_path+"fuse_ce_loss.npy",np.array(fuse_ce_loss_lst))
            
        if (epoch+1)%10==0:
            state_dict = {'net':net.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
            torch.save(state_dict, snapshot_path + f'ICCTUNet_epoch{epoch}.pth')

train(ICCTUNet,num_epochs)

