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
import os.path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id', type=int, default=0,help="id of cuda device,default:0")
config = parser.parse_args()

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
device = torch.device(f'cuda:{config.cuda_id}')
ICCTUNet = ICCTUNet()
checkpoint = torch.load('./checkpoint_synapse.pth',map_location=device)
ICCTUNet.load_state_dict(checkpoint['net'])
ICCTUNet.to(device)
print('load weights done')

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

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


class SynapseDataset_test(Dataset):
    def __init__(self,data_list,re_size=None):
        super(SynapseDataset_test,self).__init__()
        self.data_list = data_list
        self.re_size = re_size
    def __getitem__(self,index):
        file = self.data_list[index]
        file_name = file.split('/')[-1][:-7]
        data = h5py.File(file)
        image, label = data['image'][:], data['label'][:]
        image = torch.tensor(image).contiguous()
        label = torch.tensor(label).contiguous()
        if self.re_size:
            image,label = val_data_aug(image,label,self.re_size)
        sample = {'image': image, 'label': label,'name':file_name}
        return sample
    def __len__(self):
        return len(self.data_list)

test_path = './data/Synapse/test_vol_h5/'
test_list = os.listdir(test_path)
test_list = [os.path.join(test_path,file) for file in test_list]
test_dataset = SynapseDataset_test(test_list,224)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

num_class=9

def evaluate(net):
    print("start evaluate!")
    net.eval()
    test_dice_list = 0.0
    trans_dice_list = 0.0
    fuse_dice_list = 0.0

    with torch.no_grad():
        for data in test_loader:
            metric_lst = []
            trans_metric_lst = []
            fuse_metric_lst = []
            image,label = data['image'][0],data['label'][0]
            name = data['name'][0]
            print(f'case:{name}')
            label_tensor = label
            label_tensor = label_tensor.to(device)

            label = label.detach().numpy()
            prediction = np.zeros_like(label)
            trans_prediction = np.zeros_like(label)
            fuse_prediction = np.zeros_like(label)
            for index in range(image.shape[0]):
                slice_ = image[index,:,:]
                mask_ = label_tensor[index,:,:].to(device)

                input_tensor = slice_.unsqueeze(0).unsqueeze(0).float().to(device)
                pred,swin_pred,fuse_pred = net(input_tensor)

                out = torch.argmax(torch.softmax(pred, dim=1), dim=1).squeeze(0)
                trans_out = torch.argmax(torch.softmax(swin_pred, dim=1), dim=1).squeeze(0)
                fuse_out = torch.argmax(torch.softmax(fuse_pred, dim=1), dim=1).squeeze(0)

                out = out.cpu().detach().numpy()
                trans_out = trans_out.cpu().detach().numpy()
                fuse_out = fuse_out.cpu().detach().numpy()
                prediction[index] = out
                trans_prediction[index] = trans_out
                fuse_prediction[index] = fuse_out

            for i in range(1,num_class):
                metric_lst.append(calculate_metric_percase(prediction==i,label==i))
                trans_metric_lst.append(calculate_metric_percase(trans_prediction==i,label==i))
                fuse_metric_lst.append(calculate_metric_percase(fuse_prediction==i,label==i))

            test_dice_list += np.array(metric_lst)
            trans_dice_list += np.array(trans_metric_lst)
            fuse_dice_list += np.array(fuse_metric_lst)
    test_dice_list /= len(test_dataset)
    trans_dice_list /= len(test_dataset)
    fuse_dice_list /= len(test_dataset)

    print(f'mean dice on test set: {np.mean(test_dice_list,axis=0)[0]} mean hd95: {np.mean(test_dice_list,axis=0)[1]}')
    print(f'trans mean dice on test set: {np.mean(trans_dice_list,axis=0)[0]} mean hd95: {np.mean(trans_dice_list,axis=0)[1]}')
    print(f'fuse mean dice on test set: {np.mean(fuse_dice_list,axis=0)[0]} mean hd95: {np.mean(fuse_dice_list,axis=0)[1]}')
                

evaluate(ICCTUNet)